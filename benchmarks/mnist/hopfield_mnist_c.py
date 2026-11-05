"""MNIST-C: the frozen L8 Baseline and Hopfield networks on 15 unseen corruptions.

Same networks as ``benchmarks.mnist.hopfield`` (L8 placement, frozen memory),
evaluated on the MNIST-C test corruptions (Mu & Gilmer 2019). beta* is read
from ``benchmarks/mnist/tuned_params.json`` (selected on validation data by
``benchmarks.mnist.tuning``), so MNIST-C is never used to tune anything.

Run from the repository root (tune once first):

    python -m benchmarks.mnist.tuning
    python -m benchmarks.mnist.hopfield_mnist_c            # 10 seeds, reuses cached backbones
    python -m benchmarks.mnist.hopfield_mnist_c --quick    # smoke run
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.mnist.baseline import (
    MNIST_C_CORRUPTION_GROUPS,
    MNIST_C_UNCORRUPTED_NAME,
    BaselineRun,
    EmbeddingClassifier,
    apply_baseline_params,
    build_arg_parser,
    config_from_args,
    configure_threads,
    encode,
    ensure_mnist_c_downloaded,
    environment_info,
    forward_embeddings,
    load_mnist,
    load_mnist_c_test_set,
    mean_std,
    nearest_prototype,
    prepare_baseline_run,
    require_tuned,
    resolve_device,
    summarize_outputs,
    train_size,
)
from benchmarks.mnist.hopfield import (
    ALPHA,
    BASELINE_COLOR,
    HOPFIELD_COLOR,
    WINNER_DELTA,
    HopfieldRetrievalLayer,
    _import_pyplot,
    _ms,
    _p,
    _signed,
    _style,
    build_embedding_net,
    decide_winner,
    holm_adjusted_p_values,
    mcnemar_test,
    memory_fingerprint,
    seed_stats,
    selection_label,
    shared_weights_equal,
    spearman_correlation,
    table_header,
)

CORRUPTION_READOUTS = ("baseline", "hopfield", "nearest_prototype")


def evaluate_readouts_on_test_set(
    baseline_net: EmbeddingClassifier,
    hopfield_net: EmbeddingClassifier,
    query_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    prototypes: torch.Tensor,
    memory_labels: torch.Tensor,
    beta: float,
    hard_readout_beta: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Paired metrics of all three readouts on one embedded test set; dataset-independent.

    ``nearest_prototype`` is the Hopfield layer at a very large beta, where the
    softmax becomes a hard choice (check S1 verifies this equivalence). Each
    non-baseline readout also carries its McNemar counts against the baseline.
    """
    bridge = hopfield_net.bridge
    assert isinstance(bridge, HopfieldRetrievalLayer)

    def run(net: EmbeddingClassifier) -> dict[str, torch.Tensor]:
        return forward_embeddings(net, query_embeddings, batch_size=batch_size, device=device)

    outputs = {"baseline": run(baseline_net)}
    bridge.set_retrieval(beta=hard_readout_beta)
    outputs["nearest_prototype"] = run(hopfield_net)
    bridge.set_retrieval(beta=beta)
    outputs["hopfield"] = run(hopfield_net)

    evaluated: dict[str, dict[str, Any]] = {}
    correct: dict[str, torch.Tensor] = {}
    for readout in CORRUPTION_READOUTS:
        metrics, per_sample, _ = summarize_outputs(outputs[readout], labels, prototypes, memory_labels)
        correct[readout] = per_sample["head"]
        evaluated[readout] = {**metrics, "mcnemar_b": None, "mcnemar_c": None, "mcnemar_p": None}
        if readout != "baseline":
            test = mcnemar_test(correct["baseline"], correct[readout])
            evaluated[readout].update(mcnemar_b=test["b"], mcnemar_c=test["c"], mcnemar_p=test["p"])
    return evaluated


def _paired_summary(
    first_by_seed: Sequence[float],
    second_by_seed: Sequence[float],
) -> dict[str, Any]:
    """Mean ± std of both sides, Δ (first − second) and across-seed statistics."""
    deltas = [a - b for a, b in zip(first_by_seed, second_by_seed)]
    return {
        "first": mean_std(first_by_seed),
        "second": mean_std(second_by_seed),
        "delta": sum(deltas) / len(deltas),
        "delta_std": mean_std(deltas)[1],
        **seed_stats(deltas),
    }


def summarize_corruption_suite(
    rows: Sequence[dict[str, Any]],
    corruption_groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Overall, per-group and per-corruption comparisons from flat per-seed rows.

    The main result is the mean accuracy over all corruptions (Hopfield −
    Baseline, Wilcoxon across seeds). Group and corruption results are
    secondary, so their Wilcoxon p-values are Holm-adjusted within each family
    and the winner is decided on the adjusted p-value.
    """
    seeds = sorted({r["seed"] for r in rows})
    by_key = {(r["seed"], r["corruption"], r["readout"]): r for r in rows}
    corruptions = [name for names in corruption_groups.values() for name in names]

    def accuracy_by_seed(readout: str, names: Sequence[str]) -> list[float]:
        return [
            sum(by_key[(s, name, readout)]["head_acc"] for name in names) / len(names)
            for s in seeds
        ]

    def compare(first: str, second: str, names: Sequence[str]) -> dict[str, Any]:
        return _paired_summary(accuracy_by_seed(first, names), accuracy_by_seed(second, names))

    def hopfield_mean(name: str, metric: str) -> float:
        return sum(by_key[(s, name, "hopfield")][metric] for s in seeds) / len(seeds)

    overall = {
        "hopfield_vs_baseline": compare("hopfield", "baseline", corruptions),
        "nearest_prototype_vs_baseline": compare("nearest_prototype", "baseline", corruptions),
        "hopfield_vs_nearest_prototype": compare("hopfield", "nearest_prototype", corruptions),
    }
    for comparison in overall.values():
        comparison["winner"] = None
        if comparison["wilcoxon_p"] is not None:
            significant = abs(comparison["delta"]) >= WINNER_DELTA and comparison["wilcoxon_p"] < ALPHA
            comparison["significant"] = significant
            comparison["winner"] = "first" if significant and comparison["delta"] > 0 else (
                "second" if significant else "tie"
            )

    per_group = [{"group": g, **compare("hopfield", "baseline", names)}
                 for g, names in corruption_groups.items()]
    per_corruption = []
    for group, names in corruption_groups.items():
        for name in names:
            comparison = compare("hopfield", "baseline", [name])
            tests = [by_key[(s, name, "hopfield")] for s in seeds]
            comparison.update(
                corruption=name,
                group=group,
                nearest_prototype=mean_std(accuracy_by_seed("nearest_prototype", [name])),
                mcnemar_p=max(t["mcnemar_p"] for t in tests),
                seed_signs=[(t["mcnemar_c"] > t["mcnemar_b"]) - (t["mcnemar_c"] < t["mcnemar_b"])
                            for t in tests],
                embedding_shift=hopfield_mean(name, "embedding_shift"),
                entropy_norm=hopfield_mean(name, "entropy_norm"),
                true_proto_weight=hopfield_mean(name, "true_proto_weight"),
            )
            per_corruption.append(comparison)

    for family in (per_group, per_corruption):
        have_wilcoxon = all(item["wilcoxon_p"] is not None for item in family)
        adjusted = (holm_adjusted_p_values([item["wilcoxon_p"] for item in family])
                    if have_wilcoxon else [None] * len(family))
        for item, holm_p in zip(family, adjusted):
            item["holm_p"] = holm_p
            if holm_p is not None:
                item["winner"] = decide_winner(item["delta"], holm_p, (), holm_p)
            elif "mcnemar_p" in item:
                item["winner"] = decide_winner(item["delta"], item["mcnemar_p"], item["seed_signs"])
            else:
                item["winner"] = None

    return {
        "overall": overall,
        "per_group": per_group,
        "per_corruption": per_corruption,
        "shift_gain_spearman": spearman_correlation(
            [c["embedding_shift"] for c in per_corruption], [c["delta"] for c in per_corruption]
        ),
    }


MNIST_C_SECTION = "hopfield_mnist_c"


def mnist_c_fingerprint(cfg: dict[str, Any], num_train: int) -> dict[str, Any]:
    return memory_fingerprint(cfg, num_train, cfg["mnist_c_beta_values"])


def hopfield_mnist_c_params(cfg: dict[str, Any], data: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Stored beta* on the wider MNIST-C grid and its mean validation curve."""
    return require_tuned(cfg, MNIST_C_SECTION, mnist_c_fingerprint(cfg, train_size(cfg, data)))


def prepare_mnist_c_nets(
    cfg: dict[str, Any],
    *,
    retrain: bool = False,
) -> tuple[torch.device, list[BaselineRun], list[EmbeddingClassifier], float, list[float]]:
    """Frozen backbones and Hopfield nets at the stored beta*.

    beta* was selected on the validation sets only, by ``benchmarks.mnist.tuning``;
    MNIST-C is never used to select or tune anything. Backbones come from the
    same cache as the main experiment, so nothing is retrained when that cache
    matches the config.
    """
    threads = configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    print(f"Device: {device}  threads: {threads}  torch {torch.__version__}")
    data = load_mnist(cfg["data_dir"])
    apply_baseline_params(cfg, data)
    tuned = hopfield_mnist_c_params(cfg, data)
    beta_star: float = tuned["beta_star"]
    ensure_mnist_c_downloaded(cfg["data_dir"])

    per_class, center = cfg["prototypes_per_class"], cfg["center_prototypes"]
    runs: list[BaselineRun] = []
    hopfield_nets: list[EmbeddingClassifier] = []
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        runs.append(run)
        hopfield_nets.append(build_embedding_net(run, per_class, center, beta=beta_star))
    print(f"beta* = {beta_star:g} (stored; validation head_acc on "
          f"{selection_label(cfg['beta_selection_sets'])}, tuned on seeds {tuned['seeds']})")
    return device, runs, hopfield_nets, beta_star, tuned["mean_val_curve"]


def run_mnist_c_experiment(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """Evaluate the frozen Baseline and Hopfield nets on every MNIST-C test set."""
    device, runs, hopfield_nets, beta_star, mean_validation_curve = prepare_mnist_c_nets(
        cfg, retrain=retrain
    )
    betas = cfg["mnist_c_beta_values"]
    selection_sets = cfg["beta_selection_sets"]
    rows: list[dict[str, Any]] = []
    uncorrupted_accuracy_gaps: list[float] = []
    hard_readout_agreement: list[float] = []
    shared_weights_ok: list[bool] = []
    clean_images = runs[0].splits.test_images
    uncorrupted_images, uncorrupted_labels = load_mnist_c_test_set(
        cfg["data_dir"], MNIST_C_UNCORRUPTED_NAME, test_subset=cfg["test_subset"]
    )
    images_match = torch.equal(uncorrupted_images, clean_images)
    labels_match = torch.equal(uncorrupted_labels, runs[0].splits.test_labels)
    if not labels_match:
        raise ValueError("MNIST-C labels do not match the MNIST test labels; results would be invalid.")

    set_groups = {MNIST_C_UNCORRUPTED_NAME: "uncorrupted"}
    set_groups.update({n: g for g, names in MNIST_C_CORRUPTION_GROUPS.items() for n in names})
    for run, hopfield_net in zip(runs, hopfield_nets):
        print(f"  seed {run.seed}: evaluating {len(set_groups)} MNIST-C test sets")
        bridge = hopfield_net.bridge
        assert isinstance(bridge, HopfieldRetrievalLayer)
        bridge.set_retrieval(beta=beta_star, num_updates=1)
        offset = bridge.center.cpu() if bridge.center is not None else torch.zeros(run.z_train.shape[1])
        clean_embeddings = run.test_embeddings("gaussian", 0.0)
        shared_weights_ok.append(shared_weights_equal(run.net, hopfield_net))

        for corruption, group in set_groups.items():
            images, labels = load_mnist_c_test_set(
                cfg["data_dir"], corruption, test_subset=cfg["test_subset"]
            )
            if not torch.equal(labels, run.splits.test_labels):
                raise ValueError(f"MNIST-C {corruption} labels do not match the MNIST test labels.")
            query_embeddings = encode(run.net.encoder, images, batch_size=run.batch_size, device=device)
            shift = 1.0 - F.cosine_similarity(
                query_embeddings - offset, clean_embeddings - offset, dim=-1
            ).mean().item()
            evaluated = evaluate_readouts_on_test_set(
                run.net, hopfield_net, query_embeddings, labels,
                prototypes=run.prototypes, memory_labels=run.memory_labels,
                beta=beta_star, hard_readout_beta=cfg["s1_beta"],
                batch_size=run.batch_size, device=device,
            )
            for readout, metrics in evaluated.items():
                rows.append({"seed": run.seed, "corruption": corruption, "group": group,
                             "readout": readout, "embedding_shift": shift, **metrics})
            if corruption == MNIST_C_UNCORRUPTED_NAME:
                clean_out = forward_embeddings(
                    run.net, clean_embeddings, batch_size=run.batch_size, device=device
                )
                clean_acc = (clean_out["head_pred"] == labels).float().mean().item()
                uncorrupted_accuracy_gaps.append(abs(evaluated["baseline"]["head_acc"] - clean_acc))
                bridge.set_retrieval(beta=cfg["s1_beta"])
                hard_out = forward_embeddings(
                    hopfield_net, query_embeddings, batch_size=run.batch_size, device=device
                )
                bridge.set_retrieval(beta=beta_star)
                memory = bridge.prototypes.cpu()
                nearest = bridge.memory_labels.cpu()[
                    nearest_prototype(query_embeddings - offset, memory - offset)
                ]
                hard_readout_agreement.append((hard_out["proto_pred"] == nearest).float().mean().item())

    return {
        "config": cfg,
        "environment": environment_info(device),
        "beta_star": beta_star,
        "beta_values": betas,
        "mean_validation_curve": mean_validation_curve,
        "beta_selection": selection_label(selection_sets),
        "rows": rows,
        "n_test": int(labels.shape[0]),
        "num_cached_backbones": sum(run.from_cache for run in runs),
        "sanity": {
            "uncorrupted_images_match_mnist": images_match,
            "labels_match_mnist": labels_match,
            "uncorrupted_accuracy_gap": max(uncorrupted_accuracy_gaps),
            "hard_readout_agreement": min(hard_readout_agreement),
            "shared_weights_identical": all(shared_weights_ok),
        },
        "summary": summarize_corruption_suite(rows, MNIST_C_CORRUPTION_GROUPS),
    }


def format_mnist_c_report(raw: dict[str, Any]) -> str:
    """Render the MNIST-C result tables, the sanity report and the verdict."""
    summary = raw["summary"]
    sanity = raw["sanity"]
    cfg = raw["config"]
    n_seeds = len(cfg["seeds"])
    beta_star = raw["beta_star"]
    context = (f"seeds={n_seeds}  n_test={raw['n_test']}  β*={beta_star:g}"
               f" (selected on {raw['beta_selection']})  backbones reused from cache:"
               f" {raw['num_cached_backbones']}/{n_seeds}")
    lines = table_header(
        "MNIST-C", "Main result: mean head_acc over the 15 corruptions", "readout",
        "encoder and head weights, clean training, β*", context,
    )

    def seeds(item: dict[str, Any]) -> str:
        return f"{item['n_pos']}/{item['n_neg']}"

    lines.append(f"  {'Comparison':<36} │ {'First':>17} │ {'Second':>17} │ {'Δ':>9} │"
                 f" {'Seeds +/−':>9} │ {'Wilcoxon p':>10}")
    titles = {
        "hopfield_vs_baseline": "Hopfield vs Baseline (main)",
        "nearest_prototype_vs_baseline": "Nearest prototype vs Baseline",
        "hopfield_vs_nearest_prototype": "Hopfield vs Nearest prototype",
    }
    for key, title in titles.items():
        c = summary["overall"][key]
        lines.append(f"  {title:<36} │ {_ms(c['first']):>17} │ {_ms(c['second']):>17} │"
                     f" {_signed(c['delta']):>9} │ {seeds(c):>9} │ {_p(c['wilcoxon_p']):>10}")

    lines += ["", *table_header(
        "MNIST-C", "Hopfield vs Baseline per corruption group", "corruption group",
        "as above", "Winner uses Holm-adjusted Wilcoxon p across the 6 groups",
    )]
    lines.append(f"  {'Group':<21} │ {'Baseline':>17} │ {'Hopfield':>17} │ {'Δ (H−B)':>9} │"
                 f" {'Seeds +/−':>9} │ {'Holm p':>8} │ Winner")
    for g in summary["per_group"]:
        lines.append(f"  {g['group']:<21} │ {_ms(g['second']):>17} │ {_ms(g['first']):>17} │"
                     f" {_signed(g['delta']):>9} │ {seeds(g):>9} │ {_p(g['holm_p']):>8} │"
                     f" {g['winner'] or '—'}")

    lines += ["", *table_header(
        "MNIST-C", "Hopfield vs Baseline per corruption", "corruption",
        "as above", "McNemar p = max over seeds; winner uses Holm-adjusted Wilcoxon p across 15 corruptions",
    )]
    lines.append(f"  {'Corruption':<14} │ {'Baseline':>7} │ {'Hopfield':>8} │ {'Nearest':>7} │ {'Δ (H−B)':>8} │"
                 f" {'McNemar p':>9} │ {'Seeds +/−':>9} │ {'Holm p':>7} │ {'Winner':<8} │"
                 f" {'Shift':>6} │ {'Entropy':>7} │ {'True wt':>7}")
    for c in summary["per_corruption"]:
        lines.append(
            f"  {c['corruption']:<14} │ {c['second'][0]:>7.4f} │ {c['first'][0]:>8.4f} │"
            f" {c['nearest_prototype'][0]:>7.4f} │ {_signed(c['delta']):>8} │ {_p(c['mcnemar_p']):>9} │"
            f" {seeds(c):>9} │ {_p(c['holm_p']):>7} │ {c['winner']:<8} │ {c['embedding_shift']:>6.3f} │"
            f" {c['entropy_norm']:>7.3f} │ {c['true_proto_weight']:>7.3f}"
        )
    correlation = summary["shift_gain_spearman"]
    lines.append("  Shift = 1 − cosine(corrupted z, clean z) after centering; Entropy and True wt are Hopfield retrieval.")
    lines.append("  Spearman(Shift, Δ) across corruptions = "
                 f"{'—' if correlation is None else format(correlation, '+.3f')} (descriptive only)")

    checks = [
        ("uncorrupted set is pixel-identical to the MNIST test set", sanity["uncorrupted_images_match_mnist"], ""),
        ("labels match the MNIST test labels", sanity["labels_match_mnist"], ""),
        ("uncorrupted accuracy equals clean accuracy", sanity["uncorrupted_accuracy_gap"] == 0,
         f"max gap {sanity['uncorrupted_accuracy_gap']:.6f}"),
        ("β=s1_beta readout equals the nearest prototype", sanity["hard_readout_agreement"] >= 0.999,
         f"min agreement {sanity['hard_readout_agreement']:.4f}"),
        ("Baseline and Hopfield share identical weights", sanity["shared_weights_identical"], ""),
    ]
    lines += ["", " Sanity checks"]
    for name, passed, detail in checks:
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    betas = raw["beta_values"]
    if beta_star == betas[-1]:
        lines.append(f"  NOTE  β* = {beta_star:g} is the top of the selection grid; the best value may be higher.")

    main_result = summary["overall"]["hopfield_vs_baseline"]
    wins = {name: sum(c["winner"] == name for c in summary["per_corruption"])
            for name in ("Hopfield", "Baseline", "tie", "mixed")}
    if main_result["winner"] is None:
        outcome = "not tested (needs at least 6 seeds)"
    else:
        outcome = {"first": "Hopfield wins", "second": "Baseline wins", "tie": "tie"}[main_result["winner"]]
    lines += ["", f" VERDICT: {outcome} on mean accuracy over 15 corruptions"
                  f" (Δ {_signed(main_result['delta'] * 100, 2)} pp, Wilcoxon p {_p(main_result['wilcoxon_p'])});"
                  f" per corruption: Hopfield {wins['Hopfield']}, Baseline {wins['Baseline']},"
                  f" tie {wins['tie']}" + (f", mixed {wins['mixed']}" if wins["mixed"] else "") + "."]
    return "\n".join(lines)


def plot_mnist_c_gains(raw: dict[str, Any], path: Path) -> bool:
    """mnist_c_plot.png: Hopfield − Baseline accuracy per corruption, with seed std."""
    plt = _import_pyplot()
    if plt is None:
        return False
    per_corruption = sorted(raw["summary"]["per_corruption"], key=lambda c: c["delta"])
    gains = [c["delta"] * 100 for c in per_corruption]
    errors = [c["delta_std"] * 100 for c in per_corruption]
    colors = [HOPFIELD_COLOR if gain >= 0 else BASELINE_COLOR for gain in gains]
    positions = range(len(per_corruption))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(positions, gains, height=0.55, color=colors,
            xerr=errors, error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"{c['corruption']} ({c['group']})" for c in per_corruption], fontsize=9)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=HOPFIELD_COLOR, label="Hopfield better"),
                 plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR, label="Baseline better")],
        frameon=False, loc="lower right",
    )
    _style(ax, f"MNIST-C: Hopfield − Baseline accuracy, mean ± std over {len(raw['config']['seeds'])} seed(s)",
           "accuracy difference (percentage points)", "")
    ax.grid(False, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return True


def save_mnist_c_reports(raw: dict[str, Any], text: str, output_dir: Path) -> None:
    """Write the mnist_c_* report files; files of the main experiment are not touched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnist_c_summary.md").write_text(
        f"# MNIST-C: Baseline vs Hopfield\n\n```\n{text}\n```\n", encoding="utf-8"
    )
    with open(output_dir / "mnist_c_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw["rows"][0]))
        writer.writeheader()
        writer.writerows(raw["rows"])
    with open(output_dir / "mnist_c_results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, default=str)
    plot_mnist_c_gains(raw, output_dir / "mnist_c_plot.png")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("MNIST-C: frozen Baseline vs Hopfield (L8) on unseen corruptions.").parse_args(argv)
    cfg = config_from_args(args)
    print(f"Running MNIST-C from {args.config}{' (quick)' if cfg['quick'] else ''}")
    raw = run_mnist_c_experiment(cfg, retrain=args.retrain)
    text = format_mnist_c_report(raw)
    print()
    print(text)
    save_mnist_c_reports(raw, text, Path(cfg["reports_dir"]))
    print(f"\nReports saved to: {cfg['reports_dir']}")


if __name__ == "__main__":
    main()

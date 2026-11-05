"""MNIST-C: test-batch recentering at L8, a layer change of the Hopfield comparison.

The L8 Hopfield layer normally centers each query by the train-mean embedding.
Here it can instead center each test batch on its own mean
(``query_centering="test_batch_mean"``), and the Baseline gets a matching
``BatchRecenteringLayer`` at L8 (h = z - batch mean + train mean). There are no
new parameters and nothing is retrained; beta* is the stored MNIST-C value
from ``benchmarks/mnist/tuned_params.json``.

Run from the repository root (tune once first):

    python -m benchmarks.mnist.tuning
    python -m benchmarks.mnist.hopfield_recentering            # 10 seeds
    python -m benchmarks.mnist.hopfield_recentering --quick    # smoke run
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmarks.mnist.baseline import (
    MNIST_C_CORRUPTION_GROUPS,
    MNIST_C_UNCORRUPTED_NAME,
    EmbeddingClassifier,
    build_arg_parser,
    config_from_args,
    encode,
    environment_info,
    forward_embeddings,
    load_mnist_c_test_set,
    mean_std,
)
from benchmarks.mnist.hopfield import (
    BASELINE_COLOR,
    HOPFIELD_COLOR,
    HopfieldRetrievalLayer,
    _import_pyplot,
    _p,
    _signed,
    _style,
    decide_winner,
    holm_adjusted_p_values,
    mcnemar_test,
    shared_weights_equal,
    table_header,
)
from benchmarks.mnist.hopfield_mnist_c import (
    _paired_summary,
    prepare_mnist_c_nets,
)

MIXED_CORRUPTIONS_NAME = "mixed_corruptions"
RECENTERING_DETAIL_BATCH_SIZE = 256


class BatchRecenteringLayer(nn.Module):
    """L8 for the recentered baseline: move the batch mean onto the train mean, no retrieval.

    z [B, D] -> h = z - mean(z over the batch) + train_mean [B, D].
    """

    def __init__(self, train_mean: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("train_mean", train_mean.detach().clone().float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z - z.mean(dim=0, keepdim=True) + self.train_mean


def mix_corruptions(
    embeddings_by_corruption: Sequence[torch.Tensor],
    *,
    seed: int,
) -> torch.Tensor:
    """Give each test image [N, D] the embedding of one randomly chosen corruption.

    Image order is kept, so the result stays paired with the test labels and a
    batch holds different corruptions, which is the hard case for recentering.
    """
    generator = torch.Generator().manual_seed(seed)
    num_images = embeddings_by_corruption[0].shape[0]
    chosen = torch.randint(0, len(embeddings_by_corruption), (num_images,), generator=generator)
    return torch.stack(list(embeddings_by_corruption))[chosen, torch.arange(num_images)]


def _recentering_rows(
    readout: str,
    batch_size: int | None,
    predictions: torch.Tensor,
    baseline_predictions: torch.Tensor,
    labels: torch.Tensor,
    **identity: Any,
) -> dict[str, Any]:
    """One flat result row: accuracy and McNemar against the train-mean baseline."""
    test = mcnemar_test(baseline_predictions == labels, predictions == labels)
    return {**identity, "readout": readout, "batch_size": batch_size,
            "head_acc": (predictions == labels).float().mean().item(),
            "mcnemar_b": test["b"], "mcnemar_c": test["c"], "mcnemar_p": test["p"]}


def run_mnist_c_recentering_experiment(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """Compare train-mean and test-batch centering for both variants on MNIST-C.

    Every test set runs at every configured batch size, because the batch mean
    gets noisier as batches shrink. Train-mean readouts do not depend on batch
    size and are evaluated once. beta* is selected exactly as for ``--mnist-c``.
    """
    device, runs, hopfield_nets, beta_star, _ = prepare_mnist_c_nets(cfg, retrain=retrain)
    corruption_names = [n for names in MNIST_C_CORRUPTION_GROUPS.values() for n in names]
    rows: list[dict[str, Any]] = []
    shared_weights_ok: list[bool] = []
    test_subset = cfg["test_subset"]

    for run, hopfield_net in zip(runs, hopfield_nets):
        print(f"  seed {run.seed}: evaluating recentering")
        bridge = hopfield_net.bridge
        assert isinstance(bridge, HopfieldRetrievalLayer) and bridge.center is not None
        bridge.set_retrieval(beta=beta_star, num_updates=1)
        shared_weights_ok.append(shared_weights_equal(run.net, hopfield_net))
        recentered_baseline = EmbeddingClassifier(
            run.net.encoder, BatchRecenteringLayer(bridge.center), run.net.head
        ).eval()
        labels = run.splits.test_labels

        embeddings: dict[str, torch.Tensor] = {}
        for name in [MNIST_C_UNCORRUPTED_NAME, *corruption_names]:
            images, set_labels = load_mnist_c_test_set(cfg["data_dir"], name, test_subset=test_subset)
            if not torch.equal(set_labels, labels):
                raise ValueError(f"MNIST-C {name} labels do not match the MNIST test labels.")
            embeddings[name] = encode(run.net.encoder, images, batch_size=run.batch_size, device=device)
        embeddings[MIXED_CORRUPTIONS_NAME] = mix_corruptions(
            [embeddings[name] for name in corruption_names], seed=run.seed
        )

        def predict(net: EmbeddingClassifier, z: torch.Tensor, batch_size: int) -> torch.Tensor:
            return forward_embeddings(net, z, batch_size=batch_size, device=device)["head_pred"]

        for name, query_embeddings in embeddings.items():
            identity = {"seed": run.seed, "test_set": name}
            bridge.set_retrieval(query_centering="train_mean")
            baseline = predict(run.net, query_embeddings, run.batch_size)
            hopfield = predict(hopfield_net, query_embeddings, run.batch_size)
            rows.append(_recentering_rows("baseline", None, baseline, baseline, labels, **identity))
            rows.append(_recentering_rows("hopfield", None, hopfield, baseline, labels, **identity))
            bridge.set_retrieval(query_centering="test_batch_mean")
            for batch_size in cfg["recentering_batch_sizes"]:
                rows.append(_recentering_rows(
                    "baseline_recentered", batch_size,
                    predict(recentered_baseline, query_embeddings, batch_size), baseline, labels, **identity,
                ))
                rows.append(_recentering_rows(
                    "hopfield_recentered", batch_size,
                    predict(hopfield_net, query_embeddings, batch_size), baseline, labels, **identity,
                ))
        bridge.set_retrieval(query_centering="train_mean")

    return {
        "config": cfg,
        "environment": environment_info(device),
        "beta_star": beta_star,
        "rows": rows,
        "n_test": int(runs[0].splits.test_labels.shape[0]),
        "num_cached_backbones": sum(run.from_cache for run in runs),
        "shared_weights_identical": all(shared_weights_ok),
        "summary": summarize_recentering(rows, corruption_names, cfg["recentering_batch_sizes"]),
    }


def summarize_recentering(
    rows: Sequence[dict[str, Any]],
    corruption_names: Sequence[str],
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    """Recentered Hopfield vs each reference, per batch size and per corruption.

    Each table is one family: Wilcoxon p-values are Holm-adjusted within it,
    per comparison, and the winner is decided on the adjusted p-value.
    """
    seeds = sorted({r["seed"] for r in rows})
    by_key = {(r["seed"], r["test_set"], r["readout"], r["batch_size"]): r["head_acc"] for r in rows}

    def accuracy_by_seed(readout: str, names: Sequence[str], batch_size: int) -> list[float]:
        key_batch = batch_size if readout.endswith("_recentered") else None
        return [sum(by_key[(s, n, readout, key_batch)] for n in names) / len(names) for s in seeds]

    def comparison_row(names: Sequence[str], batch_size: int) -> dict[str, Any]:
        recentered = accuracy_by_seed("hopfield_recentered", names, batch_size)
        return {
            "accuracy": {r: mean_std(accuracy_by_seed(r, names, batch_size)) for r in
                         ("baseline", "hopfield", "baseline_recentered", "hopfield_recentered")},
            "vs_baseline": _paired_summary(recentered, accuracy_by_seed("baseline", names, batch_size)),
            "vs_hopfield": _paired_summary(recentered, accuracy_by_seed("hopfield", names, batch_size)),
            "vs_baseline_recentered": _paired_summary(
                recentered, accuracy_by_seed("baseline_recentered", names, batch_size)
            ),
        }

    def adjust(family: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for key in ("vs_baseline", "vs_hopfield", "vs_baseline_recentered"):
            have_wilcoxon = all(item[key]["wilcoxon_p"] is not None for item in family)
            adjusted = (holm_adjusted_p_values([item[key]["wilcoxon_p"] for item in family])
                        if have_wilcoxon else [None] * len(family))
            for item, holm_p in zip(family, adjusted):
                item[key]["holm_p"] = holm_p
                item[key]["winner"] = (None if holm_p is None
                                       else decide_winner(item[key]["delta"], holm_p, (), holm_p))
        return family

    def by_batch(names: Sequence[str]) -> list[dict[str, Any]]:
        return adjust([{"batch_size": b, **comparison_row(names, b)} for b in batch_sizes])

    detail_batch = (RECENTERING_DETAIL_BATCH_SIZE if RECENTERING_DETAIL_BATCH_SIZE in batch_sizes
                    else batch_sizes[-1])
    return {
        "mean_over_corruptions": by_batch(corruption_names),
        "mixed_corruptions": by_batch([MIXED_CORRUPTIONS_NAME]),
        "uncorrupted": by_batch([MNIST_C_UNCORRUPTED_NAME]),
        "detail_batch_size": detail_batch,
        "per_corruption": adjust([{"corruption": n, **comparison_row([n], detail_batch)}
                                  for n in corruption_names]),
    }


def format_recentering_report(
    raw: dict[str, Any],
    *,
    dataset_name: str = "MNIST-C",
    metric: str = "head_acc",
) -> str:
    """Render the recentering tables, the sanity report and the verdict."""
    summary = raw["summary"]
    n_seeds = len(raw["config"]["seeds"])
    context = (f"seeds={n_seeds}  n_test={raw['n_test']}  β*={raw['beta_star']:g}  backbones reused"
               f" from cache: {raw['num_cached_backbones']}/{n_seeds}")
    held = "encoder and head weights, clean training, β*, 8 prototypes per class"

    def comparison_cells(item: dict[str, Any]) -> str:
        base, ref = item["vs_baseline"], item["vs_baseline_recentered"]
        acc = item["accuracy"]
        return (
            f"{acc['baseline'][0]:>8.4f} │ {acc['hopfield'][0]:>8.4f} │ {acc['baseline_recentered'][0]:>8.4f} │"
            f" {acc['hopfield_recentered'][0]:>8.4f} │ {_signed(base['delta']):>8} │"
            f" {base['n_pos']}/{base['n_neg']:<3} │ {_p(base['holm_p']):>7} │ {base['winner'] or '—':<8} │"
            f" {_signed(ref['delta']):>8} │ {ref['n_pos']}/{ref['n_neg']:<3} │ {_p(ref['holm_p']):>7}"
        )

    def table_columns(first: str, width: int) -> list[str]:
        return [
            (f"  {first:>{width}} │ {'Baseline':>8} │ {'Hopfield':>8} │ {'Base rec':>8} │ {'Hop rec':>8} │"
             f" {'HR−B':>8} │ {'+/−':<5} │ {'Holm p':>7} │ {'Winner':<8} │ {'HR−BR':>8} │ {'+/−':<5} │ {'Holm p':>7}"),
        ]

    lines: list[str] = []
    tables = (
        ("mean_over_corruptions", f"Mean {metric} over the 15 corruptions (each batch holds one corruption)"),
        ("mixed_corruptions", "Mixed corruptions (each image has a random corruption)"),
        ("uncorrupted", "Uncorrupted MNIST test set (the price of recentering)"),
    )
    for key, title in tables:
        lines += [*table_header(dataset_name, title, "batch size", held, context), *table_columns("batch", 6)]
        lines += [f"  {item['batch_size']:>6} │ {comparison_cells(item)}" for item in summary[key]]
        lines.append("")

    lines += table_header(
        dataset_name, f"Per corruption at batch size {summary['detail_batch_size']}", "corruption", held,
        "HR = Hopfield recentered, B = Baseline, BR = Baseline recentered; Holm across 15 corruptions",
    )
    lines += table_columns("Corruption", 14)
    lines += [f"  {item['corruption']:>14} │ {comparison_cells(item)}" for item in summary["per_corruption"]]

    lines += ["", " Sanity checks",
              f"  {'PASS' if raw['shared_weights_identical'] else 'FAIL'}  Baseline and Hopfield share identical weights"]
    lines += ["", " VERDICT (Hopfield recentered vs Baseline, mean over 15 corruptions):"]
    for item in summary["mean_over_corruptions"]:
        base = item["vs_baseline"]
        mixed = next(m for m in summary["mixed_corruptions"] if m["batch_size"] == item["batch_size"])
        clean = next(u for u in summary["uncorrupted"] if u["batch_size"] == item["batch_size"])
        lines.append(
            f"  batch {item['batch_size']:>4}: {base['winner'] or 'not tested'}"
            f" (Δ {_signed(base['delta'] * 100, 2)} pp, Holm p {_p(base['holm_p'])});"
            f" mixed Δ {_signed(mixed['vs_baseline']['delta'] * 100, 2)} pp;"
            f" uncorrupted Δ {_signed(clean['vs_baseline']['delta'] * 100, 2)} pp"
        )
    return "\n".join(lines)


def plot_recentering(raw: dict[str, Any], path: Path, *, metric: str = "head_acc") -> bool:
    """Accuracy vs batch size for the three test conditions (mixed and uncorrupted stay head_acc)."""
    plt = _import_pyplot()
    if plt is None:
        return False
    panels = (("mean_over_corruptions", "Mean over 15 corruptions"),
              ("mixed_corruptions", "Mixed corruptions"),
              ("uncorrupted", "Uncorrupted"))
    series = (("baseline", "Baseline", BASELINE_COLOR, "--"),
              ("hopfield", "Hopfield", HOPFIELD_COLOR, "--"),
              ("baseline_recentered", "Baseline recentered", BASELINE_COLOR, "-"),
              ("hopfield_recentered", "Hopfield recentered", HOPFIELD_COLOR, "-"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, title) in zip(axes, panels):
        items = raw["summary"][key]
        xs = [item["batch_size"] for item in items]
        for readout, label, color, style in series:
            ax.plot(xs, [item["accuracy"][readout][0] for item in items], color=color, linestyle=style,
                    linewidth=2, marker="o", markersize=5, label=label)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        _style(ax, title, "batch size (log scale)", metric if key == "mean_over_corruptions" else "head_acc")
    axes[0].legend(frameon=False, fontsize=9, handlelength=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return True


def save_recentering_reports(raw: dict[str, Any], text: str, output_dir: Path) -> None:
    """Write the mnist_c_recentering_* files; no other report is touched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnist_c_recentering_summary.md").write_text(
        f"# MNIST-C: test-batch recentering\n\n```\n{text}\n```\n", encoding="utf-8"
    )
    with open(output_dir / "mnist_c_recentering_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw["rows"][0]))
        writer.writeheader()
        writer.writerows(raw["rows"])
    with open(output_dir / "mnist_c_recentering_results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, default=str)
    plot_recentering(raw, output_dir / "mnist_c_recentering_plot.png")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("MNIST-C: train-mean vs test-batch centering at L8.").parse_args(argv)
    cfg = config_from_args(args)
    print(f"Running MNIST-C recentering from {args.config}{' (quick)' if cfg['quick'] else ''}")
    raw = run_mnist_c_recentering_experiment(cfg, retrain=args.retrain)
    text = format_recentering_report(raw)
    print()
    print(text)
    save_recentering_reports(raw, text, Path(cfg["reports_dir"]))
    print(f"\nReports saved to: {cfg['reports_dir']}")


if __name__ == "__main__":
    main()

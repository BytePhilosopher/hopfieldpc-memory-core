"""CIFAR-10-C: independent test of test-batch recentering, Baseline vs Hopfield (Colab, GPU).

Same controlled design as ``benchmarks.mnist.hopfield``: one backbone,
and only layer L8 differs. The method was fixed before any CIFAR-10-C result
existed, so this run can confirm or reject what was found on MNIST-C:

    Main result: Hopfield recentered vs Baseline at batch 256, mean accuracy AUC
    over severities 0-5 (0 = clean) across the 15 standard corruptions, tested
    with Wilcoxon across seeds.

Everything is selected on validation data by ``benchmarks.cifar10.tuning``,
which trains the backbones, applies the quality stop and stores beta* in
``benchmarks/cifar10/tuned_params.json``. This module only reads beta*;
corrupted test data is downloaded only after tuning is complete.

Colab setup (Runtime > Change runtime type > T4 GPU), one cell per block:

    # 1. Put the project on Colab: upload hopfieldpc_code.zip, or clone your branch.
    !unzip -q -o /content/hopfieldpc_code.zip -d /content/hopfieldpc-memory-core
    %cd /content/hopfieldpc-memory-core
    !pip install -q --no-deps -e .

    # 2. Keep checkpoints on Drive so a disconnect loses at most one seed.
    from google.colab import drive
    drive.mount("/content/drive")

    # 3. Short trial first, then the full run. Tune once, then evaluate; rerun the
    #    same command to resume. Keep the tuned-parameter file on Drive too, and
    #    copy it back to benchmarks/cifar10/tuned_params.json to commit it.
    !python -m benchmarks.cifar10.tuning --quick --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval_quick \
        --tuned-params /content/drive/MyDrive/cifar10_tuned_params.json
    !python -m benchmarks.cifar10.hopfield --quick --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval_quick \
        --tuned-params /content/drive/MyDrive/cifar10_tuned_params.json
    !python -m benchmarks.cifar10.tuning --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval \
        --tuned-params /content/drive/MyDrive/cifar10_tuned_params.json
    !python -m benchmarks.cifar10.hopfield --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval \
        --tuned-params /content/drive/MyDrive/cifar10_tuned_params.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from benchmarks.cifar10.baseline import (
    CORRUPTION_NAMES,
    DETAIL_BATCH_SIZE,
    HOPFIELD_SECTION,
    IMAGES_PER_SEVERITY,
    SEVERITIES,
    Cifar10CConfig,
    _load_matching_json,
    _save_json,
    add_common_arguments,
    config_from_args,
    ensure_cifar10_c_downloaded,
    load_cifar10,
    load_cifar10_c_test_set,
    prepare_seed_run,
)
from benchmarks.common import tuned_params
from benchmarks.mnist.baseline import (
    MNIST_C_UNCORRUPTED_NAME,
    EmbeddingClassifier,
    accuracy_auc_over_severity,
    encode,
    environment_info,
    forward_embeddings,
    mean_std,
    summarize_split_accuracy,
)
from benchmarks.mnist.hopfield import (
    HopfieldRetrievalLayer,
    build_embedding_net,
    shared_weights_equal,
)
from benchmarks.mnist.hopfield_mnist_c import _paired_summary
from benchmarks.mnist.hopfield_recentering import (
    MIXED_CORRUPTIONS_NAME,
    BatchRecenteringLayer,
    _recentering_rows,
    format_recentering_report,
    mix_corruptions,
    plot_recentering,
    readout_split_accuracy,
    summarize_recentering,
)



def evaluate_seed(
    cfg: Cifar10CConfig,
    data: Sequence[torch.Tensor],
    seed: int,
    beta_star: float,
    device: torch.device,
) -> dict[str, Any]:
    """Per-severity accuracy rows of every readout for one seed; saved so a rerun can resume."""
    path = Path(cfg.output_dir) / f"results_seed{seed}.json"
    fingerprint = {**cfg.evaluation_fingerprint(seed), "beta_star": beta_star}
    saved = _load_matching_json(path, fingerprint)
    if saved is not None and "split_accuracy" in saved:
        print(f"  seed {seed}: loaded saved results")
        return saved

    run = prepare_seed_run(cfg, data, seed, device)
    hopfield_net = build_embedding_net(run, cfg.prototypes_per_class, cfg.center_prototypes, beta=beta_star)
    bridge = hopfield_net.bridge
    assert isinstance(bridge, HopfieldRetrievalLayer) and bridge.center is not None
    recentered_baseline = EmbeddingClassifier(
        run.net.encoder, BatchRecenteringLayer(bridge.center), run.net.head  # type: ignore[arg-type]
    ).eval()
    split_acc = readout_split_accuracy(run, hopfield_net, recentered_baseline)
    print(f"  seed {seed}: train/test accuracy " + "  ".join(
        f"{m} {a['train']:.4f}/{a['test']:.4f}" for m, a in split_acc.items()), flush=True)
    if saved is not None:
        # Results saved before train/test accuracy was logged: add it, keep everything else.
        saved["split_accuracy"] = split_acc
        _save_json(path, saved)
        return saved

    print(f"  seed {seed}: evaluating {len(CORRUPTION_NAMES)} corruptions at {len(SEVERITIES)} severities", flush=True)

    def predict(net: EmbeddingClassifier, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        return forward_embeddings(net, z, batch_size=batch_size, device=device)["head_pred"]

    clean_labels = run.splits.test_labels
    # (corruption, severity) -> (embeddings, labels); images themselves are not kept.
    test_sets: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {
        (MNIST_C_UNCORRUPTED_NAME, 0): (
            encode(run.net.encoder, run.splits.test_images, batch_size=run.batch_size, device=device),
            clean_labels,
        ),
    }
    labels_match = True
    for name in CORRUPTION_NAMES:
        for severity in SEVERITIES:
            images, labels = load_cifar10_c_test_set(cfg.data_dir, name, severity, test_subset=cfg.test_subset)
            labels_match = labels_match and torch.equal(labels, clean_labels)
            test_sets[(name, severity)] = (
                encode(run.net.encoder, images, batch_size=run.batch_size, device=device), labels,
            )
    corrupted = [test_sets[(n, s)][0] for n in CORRUPTION_NAMES for s in SEVERITIES]
    test_sets[(MIXED_CORRUPTIONS_NAME, 0)] = (mix_corruptions(corrupted, seed=seed), test_sets[(CORRUPTION_NAMES[0], 1)][1])

    rows: list[dict[str, Any]] = []
    for (name, severity), (query_embeddings, labels) in test_sets.items():
        identity = {"seed": seed, "test_set": name, "severity": severity}
        bridge.set_retrieval(beta=beta_star, query_centering="train_mean")
        baseline = predict(run.net, query_embeddings, run.batch_size)
        rows.append(_recentering_rows("baseline", None, baseline, baseline, labels, **identity))
        rows.append(_recentering_rows(
            "hopfield", None, predict(hopfield_net, query_embeddings, run.batch_size), baseline, labels, **identity,
        ))
        bridge.set_retrieval(query_centering="test_batch_mean")
        for batch_size in cfg.recentering_batch_sizes:
            rows.append(_recentering_rows(
                "baseline_recentered", batch_size, predict(recentered_baseline, query_embeddings, batch_size),
                baseline, labels, **identity,
            ))
            rows.append(_recentering_rows(
                "hopfield_recentered", batch_size, predict(hopfield_net, query_embeddings, batch_size),
                baseline, labels, **identity,
            ))
        # Soft vs hard retrieval: the same recentered layer at a very large beta.
        bridge.set_retrieval(beta=cfg.hard_readout_beta)
        rows.append(_recentering_rows(
            "nearest_recentered", DETAIL_BATCH_SIZE, predict(hopfield_net, query_embeddings, DETAIL_BATCH_SIZE),
            baseline, labels, **identity,
        ))

    result = {
        "fingerprint": fingerprint,
        "rows": rows,
        "labels_match_cifar10": labels_match,
        "shared_weights_identical": shared_weights_equal(run.net, hopfield_net),
        "train_time_s": run.train_time_s,
        "split_accuracy": split_acc,
    }
    _save_json(path, result)
    return result


def collapse_to_severity_auc(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (seed, test set, readout, batch size), head_acc = AUC over severities 0-5.

    The clean accuracy of the same readout is severity 0. Clean and mixed test
    sets have no severity, so they keep their accuracy. The result has the
    shape ``summarize_recentering`` expects.
    """
    accuracy = {(r["seed"], r["test_set"], r["severity"], r["readout"], r["batch_size"]): r["head_acc"]
                for r in rows}
    collapsed = []
    for row in rows:
        if row["test_set"] in (MNIST_C_UNCORRUPTED_NAME, MIXED_CORRUPTIONS_NAME):
            collapsed.append(row)
        elif row["severity"] == SEVERITIES[0]:
            key = (row["seed"], row["test_set"])
            suffix = (row["readout"], row["batch_size"])
            curve = [accuracy[(row["seed"], MNIST_C_UNCORRUPTED_NAME, 0, *suffix)]]
            curve += [accuracy[(*key, s, *suffix)] for s in SEVERITIES]
            collapsed.append({**row, "head_acc": accuracy_auc_over_severity([0, *SEVERITIES], curve)})
    return collapsed


def format_severity_table(rows: Sequence[dict[str, Any]], batch_size: int) -> list[str]:
    """Mean accuracy over the 15 corruptions at each severity, for all four readouts."""
    readouts = (("baseline", None), ("hopfield", None),
                ("baseline_recentered", batch_size), ("hopfield_recentered", batch_size))
    lines = [f" Mean head_acc per severity (recentered readouts at batch {batch_size})",
             f"  {'severity':>8} │ {'Baseline':>8} │ {'Hopfield':>8} │ {'Base rec':>8} │ {'Hop rec':>8}"]
    for severity in SEVERITIES:
        cells = []
        for readout, batch in readouts:
            values = [r["head_acc"] for r in rows
                      if r["severity"] == severity and r["readout"] == readout and r["batch_size"] == batch
                      and r["test_set"] in CORRUPTION_NAMES]
            cells.append(sum(values) / len(values))
        lines.append(f"  {severity:>8} │ " + " │ ".join(f"{c:>8.4f}" for c in cells))
    return lines


def format_soft_vs_hard_line(collapsed: Sequence[dict[str, Any]]) -> str:
    """Does soft retrieval add anything over a hard nearest-prototype readout after recentering?"""
    seeds = sorted({r["seed"] for r in collapsed})

    def auc_by_seed(readout: str) -> list[float]:
        values = []
        for seed in seeds:
            aucs = [r["head_acc"] for r in collapsed
                    if r["seed"] == seed and r["readout"] == readout and r["test_set"] in CORRUPTION_NAMES
                    and r["batch_size"] == DETAIL_BATCH_SIZE]
            values.append(sum(aucs) / len(aucs))
        return values

    comparison = _paired_summary(auc_by_seed("hopfield_recentered"), auc_by_seed("nearest_recentered"))
    p_value = comparison["wilcoxon_p"]
    return (f" Soft vs hard retrieval, recentered, batch {DETAIL_BATCH_SIZE}: Δ {comparison['delta'] * 100:+.2f} pp,"
            f" seeds {comparison['n_pos']}/{comparison['n_neg']},"
            f" Wilcoxon p {'not tested' if p_value is None else format(p_value, '.3f')}")


def hopfield_params(cfg: Cifar10CConfig) -> dict[str, Any]:
    """Stored beta* and validation summary; raises when tuning has not been run for these settings."""
    return tuned_params.require(
        cfg.tuned_params, tuned_params.profile_name(cfg.quick), HOPFIELD_SECTION, cfg.tuning_fingerprint(),
        tuning_command=cfg.tuning_command(),
    )


def run_experiment(cfg: Cifar10CConfig) -> tuple[dict[str, Any], str]:
    """Evaluate every seed at the stored beta*, and write the reports; nothing is tuned here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  torch {torch.__version__}  output: {output_dir}")
    data = load_cifar10(cfg.data_dir)

    tuned = hopfield_params(cfg)
    beta_star: float = tuned["beta_star"]
    num_loaded: int = tuned["num_cached_backbones"]
    clean_val_accuracy: float = tuned["clean_val_accuracy"]
    print(f"beta* = {beta_star:g} (stored; selected on validation data by benchmarks.cifar10.tuning)")

    print("Stage 2: corrupted test sets")
    ensure_cifar10_c_downloaded(cfg.data_dir)
    results = [evaluate_seed(cfg, data, seed, beta_star, device) for seed in cfg.seeds]
    rows = [row for result in results for row in result["rows"]]
    collapsed = collapse_to_severity_auc(rows)
    train_times = [result["train_time_s"] for result in results]

    raw: dict[str, Any] = {
        "config": asdict(cfg),
        "environment": environment_info(device),
        "beta_star": beta_star,
        "clean_val_accuracy": clean_val_accuracy,
        "train_time_s": mean_std(train_times),
        "n_test": cfg.test_subset or IMAGES_PER_SEVERITY,
        "num_cached_backbones": num_loaded,
        "shared_weights_identical": all(r["shared_weights_identical"] for r in results),
        "labels_match_cifar10": all(r["labels_match_cifar10"] for r in results),
        "split_accuracy": {"per_seed": [r["split_accuracy"] for r in results],
                           "summary": summarize_split_accuracy([r["split_accuracy"] for r in results])},
        "summary": summarize_recentering(collapsed, CORRUPTION_NAMES, cfg.recentering_batch_sizes),
    }
    report = format_recentering_report(raw, dataset_name="CIFAR-10-C", metric="accuracy AUC over severity")
    report_parts = [
        report, "",
        f" Mean clean validation accuracy: {clean_val_accuracy:.4f}",
        f" Labels match the CIFAR-10 test labels: {'PASS' if raw['labels_match_cifar10'] else 'FAIL'}",
        format_soft_vs_hard_line(collapsed), "",
        *format_severity_table(rows, DETAIL_BATCH_SIZE),
    ]
    text = "\n".join(report_parts)
    print(text)

    (output_dir / "cifar10_c_summary.md").write_text(
        f"# CIFAR-10-C: Baseline vs Hopfield with test-batch recentering\n\n```\n{text}\n```\n", encoding="utf-8"
    )
    fieldnames = list(rows[0])
    with open(output_dir / "cifar10_c_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "cifar10_c_results.json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    plot_recentering(raw, output_dir / "cifar10_c_plot.png", metric="accuracy AUC")
    print(f"\nReports saved to: {output_dir}")
    return raw, text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10-C: Baseline vs Hopfield with test-batch recentering.")
    add_common_arguments(parser)
    run_experiment(config_from_args(parser.parse_args(argv)))


if __name__ == "__main__":
    main()

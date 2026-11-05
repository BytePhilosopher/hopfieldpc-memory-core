"""CIFAR-10 tuning: train the baseline backbones and select the Hopfield beta*, once.

This is the only CIFAR-10 code that tunes anything. The backbone recipe is
fixed, so the tuned value is beta*: argmax over ``beta_values`` of the
seed-mean validation accuracy over ``beta_selection_sets`` (clean + noisy
validation images, never test data). Every seed must also pass the clean
validation quality stop before beta* is stored.

The result goes to ``benchmarks/cifar10/tuned_params.json`` and
``benchmarks.cifar10.hopfield`` only reads it. Running this again does nothing
unless the settings changed or ``--force`` is passed. Per-seed validation
curves are kept in the output directory, so an interrupted run resumes.

    python -m benchmarks.cifar10.tuning [--quick] [--force] [--data-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from benchmarks.cifar10.baseline import (
    HOPFIELD_SECTION,
    Cifar10CConfig,
    _load_matching_json,
    _save_json,
    add_common_arguments,
    config_from_args,
    load_cifar10,
    prepare_seed_run,
)
from benchmarks.common import tuned_params
from benchmarks.mnist.baseline import forward_embeddings
from benchmarks.mnist.hopfield import build_embedding_net, select_beta, validation_accuracy_by_beta


def select_validation_beta(
    cfg: Cifar10CConfig,
    data: Sequence[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Train every seed, enforce the quality stop and select beta*; no corrupted test data is used."""
    curves: list[list[float]] = []
    accuracies: list[float] = []
    num_loaded = 0
    for seed in cfg.seeds:
        path = Path(cfg.output_dir) / f"validation_seed{seed}.json"
        fingerprint = cfg.evaluation_fingerprint(seed)
        saved = _load_matching_json(path, fingerprint)
        if saved is None:
            run = prepare_seed_run(cfg, data, seed, device)
            clean_out = forward_embeddings(run.net, run.z_val, batch_size=run.batch_size, device=device)
            accuracy = (clean_out["head_pred"] == run.splits.val_labels).float().mean().item()
            hopfield_net = build_embedding_net(
                run, cfg.prototypes_per_class, cfg.center_prototypes, beta=cfg.beta_values[0]
            )
            curve = validation_accuracy_by_beta(hopfield_net, run, cfg.beta_values, cfg.beta_selection_sets)
            saved = {"fingerprint": fingerprint, "clean_val_accuracy": accuracy, "curve": curve,
                     "from_cache": run.from_cache}
            _save_json(path, saved)
        if saved["clean_val_accuracy"] < cfg.min_clean_val_accuracy:
            raise SystemExit(
                f"Seed {seed}: clean validation accuracy {saved['clean_val_accuracy']:.4f} is below"
                f" {cfg.min_clean_val_accuracy}. Stopping before any corrupted test data is used."
            )
        print(f"  seed {seed}: clean validation accuracy {saved['clean_val_accuracy']:.4f}", flush=True)
        curves.append(saved["curve"])
        accuracies.append(saved["clean_val_accuracy"])
        num_loaded += bool(saved["from_cache"])
    beta_star, mean_curve = select_beta(cfg.beta_values, curves)
    return {
        "beta_star": beta_star,
        "mean_val_curve": mean_curve,
        "val_curves": curves,
        "clean_val_accuracy": sum(accuracies) / len(accuracies),
        "num_cached_backbones": num_loaded,
        "seeds": list(cfg.seeds),
    }


def tune(cfg: Cifar10CConfig, *, force: bool = False) -> dict[str, Any]:
    """Stored beta* for ``cfg``; tunes (and stores) only when missing, stale or forced."""
    profile = tuned_params.profile_name(cfg.quick)
    fingerprint = cfg.tuning_fingerprint()
    stored = None if force else tuned_params.lookup(cfg.tuned_params, profile, HOPFIELD_SECTION, fingerprint)
    if stored is not None:
        print(f"hopfield: already tuned (beta* = {stored['beta_star']:g}) in {cfg.tuned_params}; skipping")
        return stored
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  torch {torch.__version__}  output: {cfg.output_dir}")
    print("Backbones and beta selection (validation data only)")
    values = select_validation_beta(cfg, load_cifar10(cfg.data_dir), device)
    tuned_params.save_entry(cfg.tuned_params, profile, HOPFIELD_SECTION, fingerprint, values)
    print(f"beta* = {values['beta_star']:g}; saved to {cfg.tuned_params}")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10: train backbones and select beta* once.")
    add_common_arguments(parser)
    parser.add_argument("--force", action="store_true", help="Tune again even if stored values match.")
    args = parser.parse_args(argv)
    tune(config_from_args(args), force=args.force)


if __name__ == "__main__":
    main()

"""MNIST tuning for the trainable Hopfield memory (a separate layer change), computed once.

Tunes the ``hopfield_trainable`` section of ``benchmarks/mnist/tuned_params.json``:
the memory learning rate and initial beta per placement (L5 and L8), on one
seed's validation data only. It needs the ``baseline`` and ``hopfield_l5``
sections first, because the memory starts from the stored fixed betas:

    python -m benchmarks.mnist.tuning                 # baseline, L8, MNIST-C, L5
    python -m benchmarks.mnist.tuning_trainable       # this section (skipped when already tuned)
    python -m benchmarks.mnist.tuning_trainable --force
"""

from __future__ import annotations

import argparse
import copy
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from benchmarks.mnist.baseline import (
    apply_baseline_params,
    configure_threads,
    load_mnist,
    prepare_baseline_run,
    resolve_device,
    train_size,
)
from benchmarks.mnist.hopfield_l5 import encode_features
from benchmarks.mnist.hopfield_trainable import DEFAULT_CONFIG_PATH as TRAINABLE_CONFIG_PATH
from benchmarks.mnist.hopfield_trainable import (
    TRAINABLE_SECTION,
    apply_fixed_betas,
    load_trainable_config,
    make_memory_net,
    train_stage,
    trainable_fingerprint,
    validation_sets,
)
from benchmarks.mnist.tuning import _tuned_or_run


def tune_hopfield_trainable(
    cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device, *, force: bool = False
) -> dict[str, Any]:
    """Memory lr and initial beta per placement for the trainable memory, on one seed's validation data."""
    apply_fixed_betas(cfg, data)
    tuning = cfg["tuning"]

    def search() -> dict[str, Any]:
        run = prepare_baseline_run(cfg, data, tuning["seed"], device)
        val = validation_sets(cfg, run)
        train_features = encode_features(
            run.net.encoder, run.splits.train_images, batch_size=run.batch_size, device=device
        )
        limit = tuning["train_images"]
        result: dict[str, Any] = {"grid": {}, "best": {}}
        for placement, inputs, beta in (("l5", train_features, cfg["beta_l5"]), ("l8", run.z_train, cfg["beta_l8"])):
            start = make_memory_net(run, placement, inputs, cfg, beta=beta)
            grid = []
            for lr, factor in itertools.product(tuning["memory_lr_values"], tuning["beta_factors"]):
                net = copy.deepcopy(start)
                net.bridge.log_beta.data = torch.tensor(float(beta * factor)).log().to(device)
                stage = train_stage(net, run.splits.train_images[:limit], run.splits.train_labels[:limit], val, cfg,
                                    epochs=tuning["epochs"], memory_lr=lr, finetune=False, seed=tuning["seed"],
                                    device=device)
                grid.append({"memory_lr": lr, "beta_factor": factor, "initial_beta": beta * factor,
                             "val": stage["best_val"]})
                print(f"  tuning {placement}: lr {lr:g}, beta {beta * factor:g}: val {stage['best_val']:.4f}",
                      flush=True)
            result["grid"][placement] = grid
            result["best"][placement] = max(grid, key=lambda g: g["val"])
        return result

    return _tuned_or_run(
        cfg, TRAINABLE_SECTION, trainable_fingerprint(cfg, train_size(cfg, data)), search, force=force,
        describe=lambda v: "; ".join(
            f"{p}: memory lr {b['memory_lr']:g}, initial beta {b['initial_beta']:g}" for p, b in v["best"].items()
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MNIST: tune the trainable Hopfield memory once.")
    parser.add_argument("--force", action="store_true", help="Tune again even if stored values match.")
    parser.add_argument("--quick", action="store_true", help="Smoke settings, stored under the 'quick' profile.")
    parser.add_argument("--config", type=Path, default=TRAINABLE_CONFIG_PATH)
    parser.add_argument("--data-dir", type=str, default=None, help="MNIST download/cache dir.")
    parser.add_argument("--reports-dir", type=str, default=None, help="Where backbones are cached.")
    parser.add_argument("--tuned-params", type=str, default=None, help="Tuned-parameter file to read and write.")
    args = parser.parse_args(argv)

    cfg, _ = load_trainable_config(args.config, quick=args.quick, tuned_params_path=args.tuned_params)
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.reports_dir is not None:
        cfg["reports_dir"] = args.reports_dir
    configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    apply_baseline_params(cfg, data)
    tune_hopfield_trainable(cfg, data, device, force=args.force)


if __name__ == "__main__":
    main()

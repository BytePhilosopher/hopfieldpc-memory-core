"""MNIST tuning: every tuned value of every MNIST experiment, computed once.

This is the only MNIST code that tunes anything. Each section is scored on
validation data only (never test data) and written to
``benchmarks/mnist/tuned_params.json``, which the experiments only read:

    baseline            lr, batch size, epochs           -> benchmarks.mnist.baseline / every experiment
    hopfield_l8         beta*, pixel beta, variant betas -> benchmarks.mnist.hopfield
    hopfield_mnist_c    beta* on the wider grid          -> benchmarks.mnist.hopfield_mnist_c / _recentering
    hopfield_l5         beta of both placements          -> benchmarks.mnist.hopfield_l5

A section whose stored fingerprint matches the current settings is skipped, so
running this again does nothing unless a relevant setting changed (or
``--force`` is passed). Backbones needed for beta selection are trained once
and cached in the reports directory, exactly as the experiments do.

    python -m benchmarks.mnist.tuning                       # all sections
    python -m benchmarks.mnist.tuning --quick               # smoke settings (separate "quick" profile)
    python -m benchmarks.mnist.tuning --only hopfield_l5    # one section
    python -m benchmarks.mnist.tuning --force               # retune even if stored values match
"""

from __future__ import annotations

import argparse
import gc
import itertools
import statistics
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from benchmarks.common import tuned_params
from benchmarks.mnist.baseline import (
    BASELINE_SECTION,
    DEFAULT_CONFIG_PATH,
    BaselineRun,
    EmbeddingClassifier,
    apply_baseline_params,
    baseline_tuning_fingerprint,
    build_baseline_net,
    configure_threads,
    forward_images,
    format_tuning_table,
    load_config,
    load_mnist,
    make_corrupted_test_set,
    make_splits,
    prepare_baseline_run,
    resolve_device,
    train_baseline,
    train_size,
)
from benchmarks.mnist.hopfield import (
    L8_SECTION,
    build_embedding_net,
    build_seed_nets,
    l8_fingerprint,
    memory_variants,
    select_beta,
    selection_label,
    validation_accuracy_by_beta,
)
from benchmarks.mnist.hopfield_l5 import DEFAULT_CONFIG_PATH as L5_CONFIG_PATH
from benchmarks.mnist.hopfield_l5 import (
    L5_SECTION,
    build_l5_variant_nets,
    encode_features,
    l5_fingerprint,
    load_l5_config,
    validation_curve,
)
from benchmarks.mnist.hopfield_mnist_c import MNIST_C_SECTION, mnist_c_fingerprint

# In dependency order: beta selection needs the tuned backbone. The trainable
# memory (a separate layer change) is tuned by benchmarks.mnist.tuning_trainable.
SECTIONS = (BASELINE_SECTION, L8_SECTION, MNIST_C_SECTION, L5_SECTION)


def _tuned_or_run(
    cfg: dict[str, Any],
    section: str,
    fingerprint: dict[str, Any],
    tune: Callable[[], dict[str, Any]],
    *,
    force: bool,
    describe: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """Stored values when they match ``fingerprint`` (unless forced); otherwise tune and store."""
    path, profile = cfg["tuned_params"], tuned_params.profile_name(bool(cfg.get("quick")))
    stored = None if force else tuned_params.lookup(path, profile, section, fingerprint)
    if stored is not None:
        print(f"{section}: already tuned ({describe(stored)}); skipping", flush=True)
        return stored
    print(f"{section}: tuning ...", flush=True)
    values = tune()
    tuned_params.save_entry(path, profile, section, fingerprint, values)
    print(f"{section}: {describe(values)}; saved to {path}", flush=True)
    return values


def _seed_runs(cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device) -> Iterator[BaselineRun]:
    """Each configured seed's frozen backbone (trained once and cached if missing)."""
    for seed in cfg["seeds"]:
        yield prepare_baseline_run(cfg, data, seed, device)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def tune_baseline(
    cfg: dict[str, Any],
    data: Sequence[torch.Tensor],
    device: torch.device,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """Grid-search the baseline's lr, batch size and epochs on validation data.

    Each (lr, batch_size) is trained once for ``max_epochs`` epochs and scored
    after every epoch, so every epoch count is a candidate. The score is the
    same one that selects beta for the Hopfield variant: mean head_acc over
    ``beta_selection_sets`` (clean + noisy validation, noise stream 1), so
    both variants are tuned on the same criterion and the test set is never
    used. Ties -> fewest epochs, then smallest lr, then smallest batch size.

    The winning values are written into ``cfg`` (lr, batch_size, epochs).
    Returns None when ``baseline_tuning`` is null.
    """
    tuning = cfg["baseline_tuning"]
    if tuning is None:
        print(f"{BASELINE_SECTION}: baseline_tuning is null; using the config's lr / batch / epochs", flush=True)
        return None
    sets = cfg["beta_selection_sets"]

    def run_grid() -> dict[str, Any]:
        scores: dict[tuple[float, int, int], list[dict[str, float]]] = {}
        for seed in tuning["seeds"]:
            splits = make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"])
            # Clean validation is always evaluated first (reported as val_clean);
            # the score averages only the configured selection sets.
            val_sets = [splits.val_images] + [
                make_corrupted_test_set(splits.val_images, seed=seed, kind=kind, level=level, stream=1)
                for kind, level in sets if kind != "clean"
            ]
            scored = val_sets if any(kind == "clean" for kind, _ in sets) else val_sets[1:]
            for lr, bs in itertools.product(tuning["lr_values"], tuning["batch_size_values"]):
                print(f"  tuning seed {seed}: lr={lr:g} batch={bs}", flush=True)
                net = build_baseline_net(seed)

                def score(epoch: int, *, net: EmbeddingClassifier = net, lr: float = lr, bs: int = bs,
                          val_sets: list[torch.Tensor] = val_sets,
                          val_labels: torch.Tensor = splits.val_labels,
                          num_scored: int = len(scored)) -> None:
                    accs = [
                        (forward_images(net, images, batch_size=cfg["eval_batch_size"],
                                        device=device)["head_pred"] == val_labels)
                        .float().mean().item()
                        for images in val_sets
                    ]
                    scored_accs = accs[len(accs) - num_scored:]
                    scores.setdefault((lr, bs, epoch), []).append(
                        {"val_clean": accs[0], "val_score": sum(scored_accs) / num_scored}
                    )

                train_baseline(net, splits.train_images, splits.train_labels,
                               epochs=tuning["max_epochs"], batch_size=bs, lr=lr,
                               seed=seed, device=device, on_epoch_end=score)
        grid = [
            {
                "lr": lr, "batch_size": bs, "epochs": ep,
                "val_clean": statistics.fmean(s["val_clean"] for s in per_seed),
                "val_score": statistics.fmean(s["val_score"] for s in per_seed),
            }
            for (lr, bs, ep), per_seed in scores.items()
        ]
        best = min(grid, key=lambda r: (-r["val_score"], r["epochs"], r["lr"], r["batch_size"]))
        untuned = next(
            (r for r in grid if (r["lr"], r["batch_size"], r["epochs"])
             == (cfg["lr"], cfg["batch_size"], cfg["epochs"])),
            None,
        )
        return {
            "selection_sets": [list(s) for s in sets],
            "seeds": tuning["seeds"],
            "grid": grid,
            "best": {k: best[k] for k in ("lr", "batch_size", "epochs")},
            "best_scores": {k: best[k] for k in ("val_clean", "val_score")},
            "untuned": untuned,
        }

    result = _tuned_or_run(
        cfg, BASELINE_SECTION, baseline_tuning_fingerprint(cfg, train_size(cfg, data)), run_grid, force=force,
        describe=lambda v: ", ".join(f"{k}={v['best'][k]:g}" for k in ("lr", "batch_size", "epochs")),
    )
    cfg.update(result["best"])
    return result


def tune_hopfield_l8(
    cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device, *, force: bool = False
) -> dict[str, Any]:
    """beta* of the configured L8 memory, of each memory-design variant (E9) and of the pixel placement."""
    betas, sets = cfg["beta_values"], cfg["beta_selection_sets"]

    def select() -> dict[str, Any]:
        val_curves: dict[str, list[list[float]]] = {"embedding": [], "pixel": []}
        variant_curves: dict[str, list[list[float]]] = {}
        for run in _seed_runs(cfg, data, device):
            nets = build_seed_nets(cfg, run, beta=betas[0])
            val_curves["embedding"].append(validation_accuracy_by_beta(nets.hop, run, betas, sets))
            for name, net in nets.variants.items():
                variant_curves.setdefault(name, []).append(
                    val_curves["embedding"][-1] if net is nets.hop
                    else validation_accuracy_by_beta(net, run, betas, sets)
                )
            if nets.pixel is not None:
                val_curves["pixel"].append(validation_accuracy_by_beta(nets.pixel, run, betas, sets, pixel=True))
        beta_star, mean_val_curve = select_beta(betas, val_curves["embedding"])
        return {
            "beta_star": beta_star,
            "beta_pixel": select_beta(betas, val_curves["pixel"])[0] if val_curves["pixel"] else None,
            "variant_betas": {name: select_beta(betas, c)[0] for name, c in variant_curves.items()},
            "val_curves": val_curves,
            "mean_val_curve": mean_val_curve,
            "seeds": cfg["seeds"],
        }

    return _tuned_or_run(
        cfg, L8_SECTION, l8_fingerprint(cfg, train_size(cfg, data)), select, force=force,
        describe=lambda v: f"beta* = {v['beta_star']:g} on {selection_label(sets)}",
    )


def tune_hopfield_mnist_c(
    cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device, *, force: bool = False
) -> dict[str, Any]:
    """beta* of the configured L8 memory on the wider MNIST-C grid (validation data only)."""
    betas, sets = cfg["mnist_c_beta_values"], cfg["beta_selection_sets"]
    per_class, center = cfg["prototypes_per_class"], cfg["center_prototypes"]

    def select() -> dict[str, Any]:
        curves = [
            validation_accuracy_by_beta(build_embedding_net(run, per_class, center, beta=betas[0]), run, betas, sets)
            for run in _seed_runs(cfg, data, device)
        ]
        beta_star, mean_val_curve = select_beta(betas, curves)
        return {"beta_star": beta_star, "mean_val_curve": mean_val_curve, "val_curves": curves,
                "seeds": cfg["seeds"]}

    return _tuned_or_run(
        cfg, MNIST_C_SECTION, mnist_c_fingerprint(cfg, train_size(cfg, data)), select, force=force,
        describe=lambda v: f"beta* = {v['beta_star']:g} on {selection_label(sets)}",
    )


def tune_hopfield_l5(
    cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device, *, force: bool = False
) -> dict[str, Any]:
    """beta of every Hopfield@L5 memory variant and of the Hopfield@L8 reference, on one shared grid."""
    betas, sets = cfg["l5_betas"], cfg["beta_selection_sets"]
    configured = (cfg["prototypes_per_class"], cfg["center_prototypes"])
    configured_name = next(n for n, k, c in memory_variants(configured) if (k, c) == configured)

    def select() -> dict[str, Any]:
        curves_l5: dict[str, list[list[float]]] = {}
        curves_l8: list[list[float]] = []
        for run in _seed_runs(cfg, data, device):
            kw = {"batch_size": run.batch_size, "device": device}
            labels = run.splits.val_labels
            # L5 features are large, so they are freed after each seed.
            val_features = [encode_features(run.net.encoder, run.val_images(k, lvl), **kw) for k, lvl in sets]
            for name, net in build_l5_variant_nets(cfg, run, betas[0]).items():
                curves_l5.setdefault(name, []).append(validation_curve(net, val_features, labels, betas, **kw))
            hop_l8 = build_embedding_net(run, *configured, beta=betas[0])
            val_z = [run.val_embeddings(k, lvl) for k, lvl in sets]
            curves_l8.append(validation_curve(hop_l8, val_z, labels, betas, **kw))
            print(f"  seed {run.seed}: validation curves done", flush=True)
            del run, val_features, val_z
            gc.collect()
        variant_betas = {name: select_beta(betas, c)[0] for name, c in curves_l5.items()}
        return {
            "beta_l5": variant_betas[configured_name],
            "beta_l8": select_beta(betas, curves_l8)[0],
            "variant_betas": variant_betas,
            "val_curve_l5": select_beta(betas, curves_l5[configured_name])[1],
            "val_curve_l8": select_beta(betas, curves_l8)[1],
            "seeds": cfg["seeds"],
        }

    return _tuned_or_run(
        cfg, L5_SECTION, l5_fingerprint(cfg, train_size(cfg, data)), select, force=force,
        describe=lambda v: f"beta L5 = {v['beta_l5']:g}, L8 = {v['beta_l8']:g}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_tuning(
    sections: Sequence[str],
    *,
    config: Path = DEFAULT_CONFIG_PATH,
    l5_config: Path = L5_CONFIG_PATH,
    quick: bool = False,
    force: bool = False,
    data_dir: str | None = None,
    reports_dir: str | None = None,
    tuned_params_path: str | None = None,
    data: Sequence[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Tune the requested sections in dependency order; returns {section: stored values}."""
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown sections {unknown}; choose from {SECTIONS}.")
    cfg = load_config(config, quick=quick, data_dir=data_dir, reports_dir=reports_dir,
                      tuned_params_path=tuned_params_path)
    configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    print(f"Device: {device}  tuned parameters: {cfg['tuned_params']} ({tuned_params.profile_name(quick)} profile)")
    data = load_mnist(cfg["data_dir"]) if data is None else data

    def derived(loaded: dict[str, Any]) -> dict[str, Any]:
        # Same data, backbones and tuned-parameter file as the main config.
        loaded.update(data_dir=cfg["data_dir"], reports_dir=cfg["reports_dir"], tuned_params=cfg["tuned_params"])
        return loaded

    results: dict[str, Any] = {}
    if BASELINE_SECTION in sections:
        results[BASELINE_SECTION] = tune_baseline(cfg, data, device, force=force)
        if results[BASELINE_SECTION] is not None:
            print("\n".join(format_tuning_table(results[BASELINE_SECTION])))
    else:
        apply_baseline_params(cfg, data)
    if L8_SECTION in sections:
        results[L8_SECTION] = tune_hopfield_l8(cfg, data, device, force=force)
    if MNIST_C_SECTION in sections:
        results[MNIST_C_SECTION] = tune_hopfield_mnist_c(cfg, data, device, force=force)
    if L5_SECTION in sections:
        l5_cfg = derived(load_l5_config(l5_config, quick=quick)[0])
        apply_baseline_params(l5_cfg, data)
        results[L5_SECTION] = tune_hopfield_l5(l5_cfg, data, device, force=force)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MNIST: tune every experiment once and store the results.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                        help="Sections to tune (default: all, in dependency order).")
    parser.add_argument("--force", action="store_true", help="Tune again even if stored values match.")
    parser.add_argument("--quick", action="store_true", help="Smoke settings, stored under the 'quick' profile.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--l5-config", type=Path, default=L5_CONFIG_PATH)
    parser.add_argument("--data-dir", type=str, default=None, help="MNIST download/cache dir.")
    parser.add_argument("--reports-dir", type=str, default=None, help="Where backbones are cached.")
    parser.add_argument("--tuned-params", type=str, default=None, help="Tuned-parameter file to read and write.")
    args = parser.parse_args(argv)
    run_tuning(
        [s for s in SECTIONS if s in args.only], config=args.config, l5_config=args.l5_config, quick=args.quick, force=args.force, data_dir=args.data_dir,
        reports_dir=args.reports_dir, tuned_params_path=args.tuned_params,
    )


if __name__ == "__main__":
    main()

"""MNIST: Hopfield layer after L5 (Flatten) vs Baseline, with the L8 placement as a reference.

All three models share the same frozen weights per seed:

    Baseline:     L1-L5 -> L6 -> L7 -> L8 Identity -> L9
    Hopfield@L5:  L1-L5 -> Hopfield(3136) -> L6 -> L7 -> L8 Identity -> L9
    Hopfield@L8:  L1-L5 -> L6 -> L7 -> Hopfield(64) -> L9

Memory for both placements is 8 k-means prototypes per class, centered by the
train mean. beta for both placements was selected on validation data only by
``benchmarks.mnist.tuning`` and is read from ``benchmarks/mnist/tuned_params.json``.
The backbones are loaded from the main experiment's reports directory and
never trained here.
The main test, fixed before running, is Hopfield@L5 vs Baseline on Gaussian and
salt-and-pepper robustness AUC (Wilcoxon across seeds, Holm-corrected over the two).

Run from the repository root (tune once first):

    python -m benchmarks.mnist.tuning
    python -m benchmarks.mnist.hopfield_l5            # 10 seeds
    python -m benchmarks.mnist.hopfield_l5 --quick    # smoke run
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from benchmarks.mnist.baseline import (
    MNIST_MEAN,
    MNIST_STD,
    NOISE_TYPES,
    BaselineRun,
    ClassifierHead,
    EmbeddingClassifier,
    MnistEncoder,
    accuracy_auc_over_severity,
    analytical_macs,
    apply_baseline_params,
    checkpoint_fingerprint,
    clean_split_inputs,
    configure_threads,
    copy_net,
    count_buffer_values,
    count_trainable_params,
    environment_info,
    format_split_accuracy,
    forward_embeddings,
    interleaved_cost,
    load_config,
    load_mnist,
    make_corrupted_test_set,
    make_splits,
    mean_std,
    model_size_bytes,
    nearest_prototype,
    noise_grids,
    prepare_baseline_run,
    require_tuned,
    resolve_device,
    split_accuracy,
    summarize_outputs,
    summarize_split_accuracy,
    train_size,
)
from benchmarks.mnist.hopfield import (
    ALPHA,
    MIN_SEEDS_FOR_WILCOXON,
    WINNER_DELTA,
    HopfieldRetrievalLayer,
    _import_pyplot,
    _style,
    build_embedding_net,
    build_prototype_memory,
    holm_adjusted_p_values,
    mcnemar_test,
    memory_fingerprint,
    memory_variants,
    selection_label,
    shared_weights_equal,
    wilcoxon_signed_rank_p,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mnist" / "hopfield_l5.yaml"

MODELS = ("baseline", "hopfield_l5", "hopfield_l8")
MODEL_LABELS = {"baseline": "Baseline", "hopfield_l5": "Hopfield@L5", "hopfield_l8": "Hopfield@L8"}
# (model, reference): the Δ is model − reference.
COMPARISONS = (("hopfield_l5", "baseline"), ("hopfield_l8", "baseline"), ("hopfield_l5", "hopfield_l8"))
MODEL_COLORS = {"baseline": "#2a78d6", "hopfield_l5": "#eb6834", "hopfield_l8": "#1baf7a"}
MODEL_MARKERS = {"baseline": "o", "hopfield_l5": "s", "hopfield_l8": "^"}
METRICS = ("head_acc", "proto_acc", "dist_to_true_proto", "entropy_norm", "top1_weight", "true_proto_weight")
PRIMARY_KINDS = ("gaussian", "salt_pepper")
MAX_LATENCY_OVERHEAD_PCT = 10.0
NOISE_LABELS = {"gaussian": "σ", "salt_pepper": "p", "occlusion": "side"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def flatten_features(encoder: MnistEncoder, images: torch.Tensor) -> torch.Tensor:
    """L1-L5: images [B, 1, 28, 28] -> features [B, 3136]."""
    x = (images - MNIST_MEAN) / MNIST_STD
    x = encoder.pool1(encoder.relu1(encoder.conv1(x)))
    x = encoder.pool2(encoder.relu2(encoder.conv2(x)))
    return encoder.flatten(x)


def embed_features(encoder: MnistEncoder, features: torch.Tensor) -> torch.Tensor:
    """L6-L7: features [B, 3136] -> embedding z [B, 64]."""
    return encoder.fc2(encoder.relu3(encoder.fc1(features)))


class FlattenHopfieldNet(nn.Module):
    """L1-L5 -> Hopfield retrieval on the flatten features -> L6-L7 -> L9.

    Returns ``(logits, z, h, aux)`` with h = z, because L8 is the identity here.
    """

    def __init__(self, encoder: MnistEncoder, bridge: HopfieldRetrievalLayer, head: ClassifierHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.head = head

    def forward_from_embedding(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run from precomputed L5 features [B, 3136]; named to work with ``forward_embeddings``."""
        retrieved, aux = self.bridge(features)
        z = embed_features(self.encoder, retrieved)
        return self.head(z), z, z, aux

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.forward_from_embedding(flatten_features(self.encoder, images))

    def macs_per_image(self, baseline: EmbeddingClassifier) -> int:
        return sum(m for _, m in analytical_macs(baseline)) + self.bridge.macs_per_image()


@torch.inference_mode()
def encode_features(
    encoder: MnistEncoder, images: torch.Tensor, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    """L5 features [N, 3136] on CPU."""
    encoder.eval()
    return torch.cat([
        flatten_features(encoder, images[i : i + batch_size].to(device)).cpu()
        for i in range(0, images.shape[0], batch_size)
    ])


def build_l5_net(
    baseline: EmbeddingClassifier,
    train_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    center: bool,
    beta: float,
    seed: int,
) -> FlattenHopfieldNet:
    """Hopfield@L5 with copies of the baseline's layers; adds no trainable parameters."""
    memory, memory_labels, offset = build_prototype_memory(
        train_features, labels, per_class=per_class, center=center, seed=seed
    )
    device = next(baseline.parameters()).device
    bridge = HopfieldRetrievalLayer(memory, memory_labels, beta=beta, center=offset).to(device)
    return FlattenHopfieldNet(copy_net(baseline.encoder), bridge, copy_net(baseline.head)).eval()


# ---------------------------------------------------------------------------
# Configuration: reuse the main experiment's cached work, or stop
# ---------------------------------------------------------------------------


def build_l5_variant_nets(
    cfg: dict[str, Any],
    run: BaselineRun,
    betas: dict[str, float] | float,
    *,
    train_features: torch.Tensor | None = None,
) -> dict[str, FlattenHopfieldNet]:
    """Hopfield@L5 for every memory-design variant of one seed (``betas``: one value or per variant)."""
    if train_features is None:
        train_features = encode_features(
            run.net.encoder, run.splits.train_images, batch_size=run.batch_size, device=run.device
        )
    nets = {}
    for name, per_class, center in memory_variants((cfg["prototypes_per_class"], cfg["center_prototypes"])):
        nets[name] = build_l5_net(
            run.net, train_features, run.splits.train_labels, per_class=per_class, center=center,
            beta=betas if isinstance(betas, float | int) else betas[name], seed=run.seed,
        )
    return nets


def load_l5_config(
    path: Path,
    *,
    quick: bool = False,
    seed: int | None = None,
    data_dir: str | None = None,
    reports_dir: str | None = None,
    tuned_params_path: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Return (source config with the beta grid to use, output directory)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = load_config(Path(raw["source_config"]), quick=quick, seed=seed, data_dir=data_dir,
                      tuned_params_path=tuned_params_path)
    cfg["l5_betas"] = list(cfg[raw["beta_grid_key"]])
    output = reports_dir or raw["quick_reports_dir" if quick else "reports_dir"]
    return cfg, Path(output)


L5_SECTION = "hopfield_l5"


def l5_fingerprint(cfg: dict[str, Any], num_train: int) -> dict[str, Any]:
    return memory_fingerprint(cfg, num_train, cfg["l5_betas"])


def hopfield_l5_params(cfg: dict[str, Any], data: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Stored betas of both placements (and every L5 memory variant) with their validation curves."""
    return require_tuned(cfg, L5_SECTION, l5_fingerprint(cfg, train_size(cfg, data)))


def require_cached_backbone(cfg: dict[str, Any], data: Sequence[torch.Tensor], seed: int) -> None:
    """Stop unless the seed's backbone exists and matches, so these weights are never retrained."""
    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run benchmarks.mnist.tuning (which trains it) first.")
    splits = make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"])
    expected = checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0])
    found = torch.load(path, map_location="cpu", weights_only=True).get("fingerprint")
    if found != expected:
        raise ValueError(f"{path} was trained with {found}, expected {expected}.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_images_for(run: BaselineRun, kind: str, level: float) -> torch.Tensor:
    """The same seeded test images as ``BaselineRun.test_embeddings``."""
    if level == 0:
        return run.splits.test_images
    return make_corrupted_test_set(run.splits.test_images, seed=run.seed, kind=kind, level=level)


def validation_curve(
    net: nn.Module,
    inputs: Sequence[torch.Tensor],
    labels: torch.Tensor,
    betas: Sequence[float],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Validation score per beta: mean head_acc over the selection sets."""
    curve = []
    for beta in betas:
        net.bridge.set_retrieval(beta=beta)
        accs = [
            (forward_embeddings(net, x, batch_size=batch_size, device=device)["head_pred"] == labels)
            .float().mean().item()
            for x in inputs
        ]
        curve.append(sum(accs) / len(accs))
    return curve


def evaluate_models(
    run: BaselineRun,
    nets: dict[str, nn.Module],
    kind: str,
    level: float,
    features: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Metrics per model and paired McNemar tests per comparison on one test set."""
    labels = run.splits.test_labels
    z = run.test_embeddings(kind, level)
    if "hopfield_l5" in nets and features is None:
        features = encode_features(
            run.net.encoder, test_images_for(run, kind, level), batch_size=run.batch_size, device=run.device
        )
    metrics: dict[str, dict[str, Any]] = {}
    correct: dict[str, dict[str, torch.Tensor]] = {}
    for name, net in nets.items():
        inputs = features if name == "hopfield_l5" else z
        out = forward_embeddings(net, inputs, batch_size=run.batch_size, device=run.device)
        metrics[name], correct[name], _ = summarize_outputs(out, labels, run.prototypes, run.memory_labels)
    mcnemar = {
        f"{a}|{b}": mcnemar_test(correct[b]["head"], correct[a]["head"])
        for a, b in COMPARISONS
        if a in nets and b in nets
    }
    return {"metrics": metrics, "mcnemar": mcnemar}


def _add(results: dict[str, Any], exp: str, value: Any, seed: int, evaluation: dict[str, Any], **extra: Any) -> None:
    results.setdefault(exp, []).append({"value": value, "seed": seed, **evaluation, **extra})


def sanity_for_seed(
    run: BaselineRun,
    hop_l5: FlattenHopfieldNet,
    beta_star: float,
    s1_beta: float,
    clean_features: torch.Tensor,
) -> dict[str, Any]:
    """Per-seed checks on one Hopfield@L5 network; leaves beta at ``beta_star``."""
    encoder = run.net.encoder
    bridge = hop_l5.bridge
    with torch.inference_mode():
        x = run.splits.test_images[:256].to(run.device)
        same_split = torch.equal(embed_features(encoder, flatten_features(encoder, x)), encoder(x))
        # Features of the corrupted images must reproduce the z the baseline is scored on.
        noisy = make_corrupted_test_set(run.splits.test_images, seed=run.seed, kind="gaussian", level=0.5)
        same_images = torch.equal(
            embed_features(encoder, flatten_features(encoder, noisy[:256].to(run.device))).cpu(),
            run.test_embeddings("gaussian", 0.5)[:256],
        )
        logits = run.net.head(embed_features(encoder, bridge.prototypes))
        own_class = (logits.argmax(dim=-1) == bridge.memory_labels).float().mean().item()

    bridge.set_retrieval(beta=s1_beta)
    hard = forward_embeddings(hop_l5, clean_features, batch_size=run.batch_size, device=run.device)
    bridge.set_retrieval(beta=beta_star)
    memory, offset = bridge.prototypes.cpu(), bridge.center.cpu()
    nearest = bridge.memory_labels.cpu()[nearest_prototype(clean_features - offset, memory - offset)]
    return {
        "split_matches_encoder": same_split,
        "same_corrupted_images": same_images,
        "shared_weights_equal": shared_weights_equal(run.net, hop_l5),
        "prototype_own_class": own_class,
        "hard_readout_agreement": (hard["proto_pred"] == nearest).float().mean().item(),
    }


def run_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every paired test-set experiment at the stored betas; nothing is tuned here."""
    threads = configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    env = environment_info(device)
    print(f"Device: {device}  threads: {threads}  torch {torch.__version__}", flush=True)
    data = load_mnist(cfg["data_dir"])
    tuning = apply_baseline_params(cfg, data)
    tuned = hopfield_l5_params(cfg, data)
    for seed in cfg["seeds"]:
        require_cached_backbone(cfg, data, seed)

    betas = cfg["l5_betas"]
    sel_sets = cfg["beta_selection_sets"]
    configured = (cfg["prototypes_per_class"], cfg["center_prototypes"])
    variants = memory_variants(configured)
    configured_name = next(n for n, k, c in variants if (k, c) == configured)
    variant_betas: dict[str, float] = tuned["variant_betas"]
    beta_l5: float = tuned["beta_l5"]
    beta_l8: float = tuned["beta_l8"]
    print(f"beta*: L5 = {beta_l5:g}, L8 = {beta_l8:g} (stored; {selection_label(sel_sets)},"
          f" tuned on seeds {tuned['seeds']})", flush=True)

    # Stage 2: paired test-set experiments.
    results: dict[str, Any] = {}
    grids = noise_grids(cfg)
    sigma = cfg["noisy_sigma"]
    sanity: list[dict[str, Any]] = []
    split_acc: list[dict[str, dict[str, float]]] = []
    cost: dict[str, Any] | None = None
    for seed in cfg["seeds"]:
        start = time.perf_counter()
        run = prepare_baseline_run(cfg, data, seed, device)
        if not run.from_cache:
            raise RuntimeError(f"seed {seed}: backbone was not loaded from cache.")
        train_features = encode_features(
            run.net.encoder, run.splits.train_images, batch_size=run.batch_size, device=device
        )
        l5_nets = build_l5_variant_nets(cfg, run, variant_betas, train_features=train_features)
        hop_l5 = l5_nets[configured_name]
        hop_l8 = build_embedding_net(run, *configured, beta=beta_l8)
        hop_l5.bridge.set_retrieval(beta=beta_l5, num_updates=1)
        hop_l8.bridge.set_retrieval(beta=beta_l8, num_updates=1)
        nets = {"baseline": run.net, "hopfield_l5": hop_l5, "hopfield_l8": hop_l8}

        def features_of(kind: str, level: float, net_run: BaselineRun = run) -> torch.Tensor:
            return encode_features(
                net_run.net.encoder, test_images_for(net_run, kind, level),
                batch_size=net_run.batch_size, device=device,
            )

        clean = features_of("gaussian", 0.0)
        noisy = features_of("gaussian", sigma)

        # Clean train / validation / test accuracy of every model (L5 runs from L5 features).
        inputs, split_labels = clean_split_inputs(run)
        l5_inputs = {"train": train_features, "test": clean, "val": encode_features(
            run.net.encoder, run.splits.val_images, batch_size=run.batch_size, device=device)}
        split_acc.append({name: split_accuracy(net, l5_inputs if name == "hopfield_l5" else inputs,
                                               split_labels, batch_size=run.batch_size, device=device)
                          for name, net in nets.items()})
        print(f"  seed {seed}: train/test accuracy " + "  ".join(
            f"{m} {a['train']:.4f}/{a['test']:.4f}" for m, a in split_acc[-1].items()), flush=True)
        del train_features, l5_inputs
        _add(results, "clean", "clean", seed, evaluate_models(run, nets, "gaussian", 0.0, clean))
        for kind in NOISE_TYPES:
            for level in grids[kind]:
                features = clean if level == 0 else (noisy if (kind, level) == ("gaussian", sigma) else None)
                _add(results, f"noise_{kind}", level, seed, evaluate_models(run, nets, kind, level, features))

        pair = {"baseline": run.net, "hopfield_l5": hop_l5}
        for beta in betas:
            hop_l5.bridge.set_retrieval(beta=beta)
            _add(results, "beta_clean", beta, seed, evaluate_models(run, pair, "gaussian", 0.0, clean))
            _add(results, "beta_noisy", beta, seed, evaluate_models(run, pair, "gaussian", sigma, noisy))
        hop_l5.bridge.set_retrieval(beta=beta_l5)
        for steps in cfg["num_updates_values"]:
            hop_l5.bridge.set_retrieval(num_updates=steps)
            _add(results, "steps_noisy", steps, seed, evaluate_models(run, pair, "gaussian", sigma, noisy))
        hop_l5.bridge.set_retrieval(num_updates=1)

        for name, _, _ in variants:
            variant = l5_nets[name]
            variant.bridge.set_retrieval(beta=variant_betas[name], num_updates=1)
            vpair = {"baseline": run.net, "hopfield_l5": variant}
            _add(results, "memory_clean", name, seed, evaluate_models(run, vpair, "gaussian", 0.0, clean),
                 beta=variant_betas[name])
            _add(results, "memory_noisy", name, seed, evaluate_models(run, vpair, "gaussian", sigma, noisy),
                 beta=variant_betas[name])

        sanity.append(sanity_for_seed(run, hop_l5, beta_l5, cfg["s1_beta"], clean))
        if cost is None:
            cost = measure_costs(cfg, run, hop_l5, hop_l8, device)
        print(f"  seed {seed}: evaluated in {time.perf_counter() - start:.0f}s", flush=True)
        del run, clean, noisy, nets, pair, l5_nets
        gc.collect()

    return {
        "config": cfg,
        "environment": env,
        "baseline_tuning_best": tuning["best"] if tuning else None,
        "beta_l5": beta_l5,
        "beta_l8": beta_l8,
        "variant_betas": variant_betas,
        "betas": betas,
        "beta_selection": selection_label(sel_sets),
        "val_curve_l5": tuned["val_curve_l5"],
        "val_curve_l8": tuned["val_curve_l8"],
        "n_test": int(data[2].shape[0]) if cfg["test_subset"] is None else cfg["test_subset"],
        "results": results,
        "sanity": sanity,
        "split_accuracy": {"per_seed": split_acc, "summary": summarize_split_accuracy(split_acc)},
        "cost": cost,
    }


def measure_costs(
    cfg: dict[str, Any],
    run: BaselineRun,
    hop_l5: FlattenHopfieldNet,
    hop_l8: EmbeddingClassifier,
    device: torch.device,
) -> dict[str, Any]:
    """Size and MACs per model, and latency at each batch size with interleaved repeats."""
    nets: dict[str, nn.Module] = {"baseline": run.net, "hopfield_l5": hop_l5, "hopfield_l8": hop_l8}
    macs = {
        "baseline": sum(m for _, m in analytical_macs(run.net)),
        "hopfield_l5": hop_l5.macs_per_image(run.net),
        "hopfield_l8": sum(m for _, m in analytical_macs(hop_l8)),
    }
    c = cfg["cost"]
    latency = {
        str(batch): interleaved_cost(
            nets, batch, device, repeats=c["repeats"], warmup=c["warmup"], iters=c["iters"]
        )
        for batch in c["batch_sizes"]
    }
    return {
        "trainable_params": {k: count_trainable_params(n) for k, n in nets.items()},
        "buffer_values": {k: count_buffer_values(n) for k, n in nets.items()},
        "model_size_bytes": {k: model_size_bytes(n) for k, n in nets.items()},
        "macs": macs,
        "latency": latency,
        "reference_batch": str(c["batch_size"]),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def values_in_order(rows: list[dict[str, Any]]) -> list[Any]:
    seen: list[Any] = []
    for row in rows:
        if row["value"] not in seen:
            seen.append(row["value"])
    return seen


def compare(deltas_by_seed: Sequence[float]) -> dict[str, Any]:
    """Across-seed summary of one comparison; 'tie' under the main experiment's rule."""
    delta = sum(deltas_by_seed) / len(deltas_by_seed)
    wilcoxon = wilcoxon_signed_rank_p(deltas_by_seed) if len(deltas_by_seed) >= MIN_SEEDS_FOR_WILCOXON else None
    return {
        "delta": delta,
        "plus": sum(d > 0 for d in deltas_by_seed),
        "minus": sum(d < 0 for d in deltas_by_seed),
        "wilcoxon_p": wilcoxon,
        "significant": wilcoxon is not None and abs(delta) >= WINNER_DELTA and wilcoxon < ALPHA,
    }


def winner(stats: dict[str, Any], model: str, reference: str) -> str:
    if not stats["significant"]:
        return "tie"
    return MODEL_LABELS[model] if stats["delta"] > 0 else MODEL_LABELS[reference]


def point_stats(rows: list[dict[str, Any]], value: Any, model: str, metric: str) -> tuple[float, float] | None:
    vals = [r["metrics"][model][metric] for r in rows if r["value"] == value and model in r["metrics"]]
    if not vals or vals[0] is None:
        return None
    return mean_std(vals)


def auc_by_seed(rows: list[dict[str, Any]], model: str, metric: str = "head_acc") -> list[float]:
    levels = values_in_order(rows)
    seeds = sorted({r["seed"] for r in rows})
    return [
        accuracy_auc_over_severity(
            levels,
            [next(r for r in rows if r["seed"] == s and r["value"] == v)["metrics"][model][metric] for v in levels],
        )
        for s in seeds
    ]


def primary_results(raw: dict[str, Any]) -> dict[str, Any]:
    """AUC per noise type for each comparison; Holm covers only the two main tests."""
    out: dict[str, Any] = {}
    for kind in NOISE_TYPES:
        rows = raw["results"][f"noise_{kind}"]
        aucs = {m: auc_by_seed(rows, m) for m in MODELS}
        out[kind] = {
            "auc": {m: mean_std(a) for m, a in aucs.items()},
            "comparisons": {
                f"{a}|{b}": compare([x - y for x, y in zip(aucs[a], aucs[b])]) for a, b in COMPARISONS
            },
        }
    p_values = [out[k]["comparisons"]["hopfield_l5|baseline"]["wilcoxon_p"] for k in PRIMARY_KINDS]
    if all(p is not None for p in p_values):
        for kind, p in zip(PRIMARY_KINDS, holm_adjusted_p_values(p_values)):
            out[kind]["comparisons"]["hopfield_l5|baseline"]["holm_p"] = p
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(stat: tuple[float, float] | None) -> str:
    return "—" if stat is None else f"{stat[0] * 100:.2f} ± {stat[1] * 100:.2f}"


def _pp(delta: float) -> str:
    return f"{delta * 100:+.2f}"


def _p(p: float | None) -> str:
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _table(header: Sequence[str], rows: list[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
        "",
    ]


def _seed_compare(rows: list[dict[str, Any]], value: Any, a: str, b: str, metric: str = "head_acc") -> dict[str, Any]:
    point = sorted((r for r in rows if r["value"] == value), key=lambda r: r["seed"])
    return compare([r["metrics"][a][metric] - r["metrics"][b][metric] for r in point])


def _max_mcnemar(rows: list[dict[str, Any]], value: Any, key: str) -> float:
    return max(r["mcnemar"][key]["p"] for r in rows if r["value"] == value)


def sanity_checks(raw: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    """(check, expected, measured, passed)."""
    s = raw["sanity"]
    betas = raw["betas"]
    cost = raw["cost"]
    reference = cost["latency"][cost["reference_batch"]]
    base_ms = reference["baseline"]["latency_ms_p50"]["median"]
    l5_ms = reference["hopfield_l5"]["latency_ms_p50"]["median"]
    overhead = 100.0 * (l5_ms - base_ms) / base_ms
    return [
        ("L1-L5 + L6-L7 equals the encoder's own forward", "all seeds", f"{sum(x['split_matches_encoder'] for x in s)}/{len(s)}",
         all(x["split_matches_encoder"] for x in s)),
        ("All models see the same corrupted test images", "all seeds", f"{sum(x['same_corrupted_images'] for x in s)}/{len(s)}",
         all(x["same_corrupted_images"] for x in s)),
        ("Shared weights are identical to the baseline", "all seeds", f"{sum(x['shared_weights_equal'] for x in s)}/{len(s)}",
         all(x["shared_weights_equal"] for x in s)),
        ("Hopfield@L5 adds no trainable parameters", "equal",
         f"{cost['trainable_params']['hopfield_l5']:,} vs {cost['trainable_params']['baseline']:,}",
         cost["trainable_params"]["hopfield_l5"] == cost["trainable_params"]["baseline"]),
        ("Very high β equals the nearest centered prototype", "≥ 99 %",
         f"{100 * min(x['hard_readout_agreement'] for x in s):.2f} %", min(x["hard_readout_agreement"] for x in s) >= 0.99),
        ("L5 prototypes through L6-L9 predict their own class", "≥ 95 %",
         f"{100 * min(x['prototype_own_class'] for x in s):.1f} %", min(x["prototype_own_class"] for x in s) >= 0.95),
        ("β* for L5 is not on the edge of the grid", f"inside {betas[0]:g}–{betas[-1]:g}", f"{raw['beta_l5']:g}",
         betas[0] < raw["beta_l5"] < betas[-1]),
        ("β* for L8 is not on the edge of the grid", f"inside {betas[0]:g}–{betas[-1]:g}", f"{raw['beta_l8']:g}",
         betas[0] < raw["beta_l8"] < betas[-1]),
        (f"Hopfield@L5 latency overhead at batch {cost['reference_batch']}", f"< {MAX_LATENCY_OVERHEAD_PCT:g} %",
         f"{overhead:+.1f} %", overhead < MAX_LATENCY_OVERHEAD_PCT),
    ]


def build_verdict(raw: dict[str, Any], primary: dict[str, Any]) -> str:
    rows = raw["results"]["clean"]
    clean = _seed_compare(rows, "clean", "hopfield_l5", "baseline")
    parts = [f"Clean accuracy Δ {_pp(clean['delta'])} pp ({winner(clean, 'hopfield_l5', 'baseline')})"]
    for kind in NOISE_TYPES:
        c = primary[kind]["comparisons"]["hopfield_l5|baseline"]
        p = c.get("holm_p", c["wilcoxon_p"])
        result = "no significant effect"
        if c["significant"] and (p is not None and p < ALPHA):
            result = "helps" if c["delta"] > 0 else "hurts"
        parts.append(f"{kind} AUC Δ {_pp(c['delta'])} pp ({result}, {c['plus']}/{c['minus']} seeds, p={_p(p)})")
    vs_l8 = primary["gaussian"]["comparisons"]["hopfield_l5|hopfield_l8"]
    parts.append(f"L5 vs L8 on gaussian AUC {_pp(vs_l8['delta'])} pp")
    return "; ".join(parts) + "."


def build_report(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = raw["results"]
    primary = primary_results(raw)
    seeds = raw["config"]["seeds"]
    cols = [MODEL_LABELS[m] for m in MODELS]
    verdict = build_verdict(raw, primary)
    lines = [
        "# MNIST: Hopfield after L5 (Flatten) vs Baseline",
        "",
        f"**Verdict:** {verdict}",
        "",
        (f"{len(seeds)} seeds, {raw['n_test']} test images, same frozen weights per seed. "
        f"Memory: 8 k-means prototypes per class, centered. β selected on {raw['beta_selection']}: "
        f"**L5 β\\* = {raw['beta_l5']:g}, L8 β\\* = {raw['beta_l8']:g}**. Device {raw['environment'].get('device', '')}."),
        "",
        ("Winner is `tie` when |Δ| < 0.5 pp or Wilcoxon p ≥ 0.05. Accuracy values are % mean ± std across seeds; "
        "Δ is in percentage points."),
        "",
        "## 1. Main result: robustness AUC",
        "",
        "Holm p corrects the two main tests (Gaussian, salt-and-pepper, L5 vs Baseline). Occlusion is secondary.",
        "",
    ]
    rows = []
    for kind in NOISE_TYPES:
        c = primary[kind]["comparisons"]["hopfield_l5|baseline"]
        auc = primary[kind]["auc"]
        rows.append([kind, *[_pct(auc[m]) for m in MODELS], _pp(c["delta"]), f"{c['plus']}/{c['minus']}",
                     _p(c["wilcoxon_p"]), _p(c.get("holm_p")), winner(c, "hopfield_l5", "baseline")])
    lines += _table(["Noise", *cols, "Δ L5−Base", "Seeds +/−", "Wilcoxon p", "Holm p", "Winner"], rows)

    lines += ["## 2. Clean accuracy", ""]
    rows = []
    clean = results["clean"]
    for metric, label in (("head_acc", "Head accuracy"), ("proto_acc", "Prototype accuracy")):
        c = _seed_compare(clean, "clean", "hopfield_l5", "baseline", metric)
        rows.append([label, *[_pct(point_stats(clean, "clean", m, metric)) for m in MODELS],
                     _pp(c["delta"]), f"{c['plus']}/{c['minus']}", _p(c["wilcoxon_p"]),
                     winner(c, "hopfield_l5", "baseline")])
    lines += _table(["Metric", *cols, "Δ L5−Base", "Seeds +/−", "Wilcoxon p", "Winner"], rows)
    if raw.get("split_accuracy"):
        lines += ["**Train vs test accuracy**", "",
                  *format_split_accuracy(raw["split_accuracy"]["summary"], MODEL_LABELS, markdown=True)]

    lines += ["## 3. Accuracy at each noise level", ""]
    for kind in NOISE_TYPES:
        pts = results[f"noise_{kind}"]
        lines += [f"**{kind}** ({NOISE_LABELS[kind]})", ""]
        rows = []
        for level in values_in_order(pts):
            c = _seed_compare(pts, level, "hopfield_l5", "baseline")
            d8 = _seed_compare(pts, level, "hopfield_l5", "hopfield_l8")
            rows.append([f"{level:g}", *[_pct(point_stats(pts, level, m, "head_acc")) for m in MODELS],
                         _pp(c["delta"]), f"{c['plus']}/{c['minus']}", _p(c["wilcoxon_p"]),
                         _p(_max_mcnemar(pts, level, "hopfield_l5|baseline")),
                         winner(c, "hopfield_l5", "baseline"), _pp(d8["delta"])])
        lines += _table([NOISE_LABELS[kind], *cols, "Δ L5−Base", "Seeds +/−", "Wilcoxon p", "McNemar p (max)",
                         "Winner", "Δ L5−L8"], rows)

    lines += ["## 4. Retrieval quality", "",
              "Averaged across seeds. Entropy is normalized to 0–1; lower means a more decisive retrieval.", ""]
    rows = []
    gaussian = results["noise_gaussian"]
    for level in values_in_order(gaussian):
        for model in ("hopfield_l5", "hopfield_l8"):
            rows.append([f"{level:g}", MODEL_LABELS[model],
                         *[f"{point_stats(gaussian, level, model, k)[0]:.3f}"
                           for k in ("entropy_norm", "top1_weight", "true_proto_weight")]])
    lines += _table(["Gaussian σ", "Model", "Entropy", "Top-1 weight", "Weight on true class"], rows)

    def sweep(exp: str, title: str, varied: str, fmt: Any) -> list[str]:
        pts = results[exp]
        out = [f"**{title}**", ""]
        table = []
        for value in values_in_order(pts):
            c = _seed_compare(pts, value, "hopfield_l5", "baseline")
            beta = next((r["beta"] for r in pts if r["value"] == value and "beta" in r), None)
            prefix = [fmt(value)] + ([f"{beta:g}"] if beta is not None else [])
            table.append([*prefix, _pct(point_stats(pts, value, "baseline", "head_acc")),
                          _pct(point_stats(pts, value, "hopfield_l5", "head_acc")),
                          _pp(c["delta"]), f"{c['plus']}/{c['minus']}", _p(c["wilcoxon_p"]),
                          winner(c, "hopfield_l5", "baseline")])
        has_beta = any("beta" in r for r in pts)
        return out + _table([varied, *(["β*"] if has_beta else []), "Baseline", "Hopfield@L5", "Δ", "Seeds +/−",
                             "Wilcoxon p", "Winner"], table)

    fmt_num = lambda v: f"{v:g}"
    lines += ["## 5. Sensitivity of Hopfield@L5", ""]
    lines += sweep("beta_clean", "β on clean images", "β", fmt_num)
    lines += sweep("beta_noisy", f"β on Gaussian σ = {raw['config']['noisy_sigma']:g}", "β", fmt_num)
    lines += sweep("steps_noisy", f"Retrieval steps T on Gaussian σ = {raw['config']['noisy_sigma']:g}", "T", fmt_num)
    lines += ["## 6. Memory design at L5", "", "Each design uses its own validation-selected β.", ""]
    lines += sweep("memory_clean", "Clean", "Design", str)
    lines += sweep("memory_noisy", f"Gaussian σ = {raw['config']['noisy_sigma']:g}", "Design", str)

    cost = raw["cost"]
    lines += ["## 7. Cost", ""]
    base_macs = cost["macs"]["baseline"]
    rows = [
        ["Trainable parameters", *[f"{cost['trainable_params'][m]:,}" for m in MODELS]],
        ["Frozen buffer values", *[f"{cost['buffer_values'][m]:,}" for m in MODELS]],
        ["Saved model size (bytes)", *[f"{cost['model_size_bytes'][m]:,}" for m in MODELS]],
        ["MACs per image", *[f"{cost['macs'][m]:,}" for m in MODELS]],
        ["MACs vs Baseline", *[f"{100 * (cost['macs'][m] - base_macs) / base_macs:+.2f} %" for m in MODELS]],
    ]
    for batch, stats in cost["latency"].items():
        rows.append([f"Latency, batch {batch} (ms, p50)",
                     *[f"{stats[m]['latency_ms_p50']['median']:.2f}" for m in MODELS]])
        rows.append([f"Throughput, batch {batch} (images/s)",
                     *[f"{stats[m]['throughput_img_s']['median']:,.0f}" for m in MODELS]])
    lines += _table(["Measure", *cols], rows)
    lines += ["Timings come from one seed on this machine and only compare within this run.", ""]

    checks = sanity_checks(raw)
    lines += ["## 8. Sanity checks", ""]
    lines += _table(["Check", "Expected", "Measured", "Result"],
                    [[c, e, m, "PASS" if ok else "FAIL"] for c, e, m, ok in checks])
    summary = {"verdict": verdict, "primary": primary,
               "sanity_checks": [dict(zip(("check", "expected", "measured", "passed"), c)) for c in checks]}
    return "\n".join(lines), summary


def flatten_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (experiment, value, seed, model)."""
    rows = []
    for exp, points in raw["results"].items():
        for point in points:
            for model, metrics in point["metrics"].items():
                rows.append({"experiment": exp, "value": point["value"], "seed": point["seed"], "model": model,
                             "beta": point.get("beta", ""), **{m: metrics[m] for m in METRICS}})
    return rows


def save_reports(raw: dict[str, Any], text: str, summary: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(output / "results.json", "w", encoding="utf-8") as f:
        json.dump({**raw, "summary": summary}, f, indent=2, default=str)
    rows = flatten_rows(raw)
    with open(output / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _model_line(ax: Any, xs: list[float], rows: list[dict[str, Any]], values: list[Any], model: str) -> None:
    stats = [point_stats(rows, v, model, "head_acc") for v in values]
    mean = [100 * s[0] for s in stats]
    std = [100 * s[1] for s in stats]
    color = MODEL_COLORS[model]
    ax.plot(xs, mean, color=color, linewidth=2, marker=MODEL_MARKERS[model], markersize=6,
            markeredgecolor="white", markeredgewidth=1, label=MODEL_LABELS[model])
    ax.fill_between(xs, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                    color=color, alpha=0.15, linewidth=0)


def plot_comparison(raw: dict[str, Any], path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    titles = {"gaussian": "Gaussian noise", "salt_pepper": "Salt-and-pepper noise", "occlusion": "Occlusion"}
    xlabels = {"gaussian": "Noise σ", "salt_pepper": "Corrupted pixel share p", "occlusion": "Square side (pixels)"}
    for ax, kind in zip(axes.flat, NOISE_TYPES):
        rows = raw["results"][f"noise_{kind}"]
        values = values_in_order(rows)
        for model in MODELS:
            _model_line(ax, values, rows, values, model)
        _style(ax, titles[kind], xlabels[kind], "Test accuracy (%)")
        ax.set_ylim(0, 100)
    axes[0, 0].legend(frameon=False, loc="lower left")

    ax = axes[1, 1]
    for key, rows_key, style in (("clean", "beta_clean", "-"), (f"σ = {raw['config']['noisy_sigma']:g}", "beta_noisy", "--")):
        rows = raw["results"][rows_key]
        values = values_in_order(rows)
        mean = [100 * point_stats(rows, v, "hopfield_l5", "head_acc")[0] for v in values]
        ax.plot(values, mean, color=MODEL_COLORS["hopfield_l5"], linewidth=2, linestyle=style, marker="s",
                markersize=6, markeredgecolor="white", label=f"Hopfield@L5, {key}")
        base = 100 * point_stats(rows, values[0], "baseline", "head_acc")[0]
        ax.axhline(base, color=MODEL_COLORS["baseline"], linewidth=1.5, linestyle=style, label=f"Baseline, {key}")
    ax.axvline(raw["beta_l5"], color="#8a8a85", linewidth=1, linestyle=":")
    ax.annotate(f"β* = {raw['beta_l5']:g}", (raw["beta_l5"], 4), xytext=(4, 0), textcoords="offset points",
                fontsize=9, color="#55554f")
    ax.set_xscale("log")
    _style(ax, "Hopfield@L5 accuracy across β", "β (log scale)", "Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_cost(raw: dict[str, Any], path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    cost = raw["cost"]
    batches = list(cost["latency"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    xs = [int(b) for b in batches]
    for model in MODELS:
        common = {"color": MODEL_COLORS[model], "linewidth": 2, "marker": MODEL_MARKERS[model], "markersize": 6,
                  "markeredgecolor": "white", "label": MODEL_LABELS[model]}
        axes[0].plot(xs, [cost["latency"][b][model]["latency_per_image_us"]["median"] for b in batches], **common)
        axes[1].plot(xs, [cost["latency"][b][model]["throughput_img_s"]["median"] for b in batches], **common)
    _style(axes[0], "Latency per image", "Batch size (log scale)", "Microseconds per image (log scale)")
    _style(axes[1], "Throughput", "Batch size (log scale)", "Images per second (log scale)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true", help="1 seed and reduced grids; separate output folder.")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--tuned-params", type=str, default=None,
                        help="Tuned-parameter file written by benchmarks.mnist.tuning.")
    args = parser.parse_args(argv)

    cfg, output = load_l5_config(
        args.config, quick=args.quick, seed=args.seed, data_dir=args.data_dir, reports_dir=args.reports_dir,
        tuned_params_path=args.tuned_params,
    )
    start = time.perf_counter()
    raw = run_experiment(cfg)
    raw["wall_time_s"] = time.perf_counter() - start
    text, summary = build_report(raw)
    save_reports(raw, text, summary, output)
    plot_comparison(raw, output / "comparison.png")
    plot_cost(raw, output / "cost.png")
    print(f"\n{summary['verdict']}\nWrote {output}/ in {raw['wall_time_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()

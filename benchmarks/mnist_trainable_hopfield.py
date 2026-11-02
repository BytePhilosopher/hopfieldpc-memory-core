"""MNIST: trainable Hopfield memory after L5, with a trainable L8 control.

Every model starts from the same frozen clean backbone per seed:

    baseline             clean backbone, not trained further
    baseline_aug         L6-L9 fine-tuned on noisy batches
    baseline_aug_full    L1-L9 fine-tuned on noisy batches (a stronger reference)
    fixed_l5             k-means memory after L5, not trained
    trainable_l5_a       Stage A: memory and beta trained, every other layer frozen
    trainable_l5         Stage B: then memory, beta and L6-L9 trained together
    trainable_l5_snapped each trained prototype replaced by its nearest real training feature
    trainable_l8         the same two stages with the memory at L8

The best epoch of each stage, including "no training", is chosen on validation
data only. Success test, fixed before running: trainable_l5 vs baseline_aug must
be within 0.5 pp on clean accuracy and significantly better on Gaussian and
salt-and-pepper AUC (Wilcoxon across seeds, Holm-corrected over the two).

Run from the repository root. Finished seeds are kept, so a stopped run resumes:

    python -m benchmarks.mnist_trainable_hopfield            # 10 seeds
    python -m benchmarks.mnist_trainable_hopfield --quick    # smoke run
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from benchmarks.mnist_baseline import (
    NOISE_TYPES,
    BaselineRun,
    MnistEncoder,
    accuracy_auc_over_severity,
    analytical_macs,
    configure_threads,
    copy_net,
    corrupt_images,
    environment_info,
    forward_embeddings,
    interleaved_cost,
    load_config,
    load_mnist,
    mean_std,
    model_size_bytes,
    noise_grids,
    prepare_baseline_run,
    resolve_device,
    summarize_outputs,
)
from benchmarks.mnist_l5_hopfield import (
    _p,
    _pct,
    _pp,
    _table,
    apply_cached_tuning,
    compare,
    embed_features,
    encode_features,
    flatten_features,
    require_cached_backbone,
    test_images_for,
    values_in_order,
)
from benchmarks.mnist_prototype_retrieval import (
    ALPHA,
    WINNER_DELTA,
    _import_pyplot,
    _style,
    build_prototype_memory,
    holm_adjusted_p_values,
    mcnemar_test,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mnist_trainable_hopfield.yaml"

MODELS = (
    "baseline",
    "baseline_aug",
    "baseline_aug_full",
    "fixed_l5",
    "trainable_l5_a",
    "trainable_l5",
    "trainable_l5_snapped",
    "trainable_l8",
)
MODEL_LABELS = {
    "baseline": "Baseline (clean)",
    "baseline_aug": "Baseline+noise (L6–L9)",
    "baseline_aug_full": "Baseline+noise (L1–L9)",
    "fixed_l5": "Fixed L5",
    "trainable_l5_a": "Trainable L5, memory only",
    "trainable_l5": "Trainable L5",
    "trainable_l5_snapped": "Trainable L5, snapped",
    "trainable_l8": "Trainable L8",
}
# (model, reference): Δ is model − reference. The first is the success test.
COMPARISONS = (
    ("trainable_l5", "baseline_aug"),
    ("trainable_l5", "baseline_aug_full"),
    ("trainable_l5", "fixed_l5"),
    ("trainable_l5", "baseline"),
    ("trainable_l5", "trainable_l8"),
    ("trainable_l8", "baseline_aug"),
    ("trainable_l5_snapped", "trainable_l5"),
)
PLOT_MODELS = ("baseline", "baseline_aug", "baseline_aug_full", "fixed_l5", "trainable_l5", "trainable_l8")
MODEL_COLORS = dict(zip(PLOT_MODELS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
MODEL_MARKERS = dict(zip(PLOT_MODELS, ("o", "s", "^", "D", "v", "P")))
METRICS = ("head_acc", "proto_acc", "entropy_norm", "top1_weight", "true_proto_weight")
PRIMARY_KINDS = ("gaussian", "salt_pepper")
CLEAN_MARGIN = 0.005
# A finished seed is reused only if these settings match.
RESUME_KEYS = (
    "beta_l5", "beta_l8", "train_gaussian_sigmas", "train_salt_pepper_probs", "memory_epochs", "finetune_epochs",
    "backbone_lr", "beta_lr", "cluster_weight", "separation_weight", "center_ema", "batch_size", "lr", "epochs",
    "gaussian_sigmas", "salt_pepper_probs", "occlusion_sizes", "train_subset", "test_subset", "val_size",
)
NOISE_LABELS = {"gaussian": "σ", "salt_pepper": "p", "occlusion": "side"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TrainableHopfieldLayer(nn.Module):
    """Hopfield retrieval with a learned memory and a learned beta.

    ``HopfieldMemory`` takes beta as a plain float, so it cannot learn it. This
    layer computes the same centered one-step retrieval as
    ``HopfieldRetrievalLayer``:

        p = softmax(beta * cos(z - c, M - c)),  h = p (M - c) + c

    memory [K, D] and log_beta are parameters; memory_labels [K] and center [D]
    are buffers. With ``track_center`` the center follows the batch mean as an
    exponential moving average, in training mode only.
    """

    def __init__(
        self,
        memory: torch.Tensor,
        memory_labels: torch.Tensor,
        center: torch.Tensor,
        *,
        beta: float,
        center_ema: float = 0.99,
    ) -> None:
        super().__init__()
        self.memory = nn.Parameter(memory.detach().clone().float())
        self.log_beta = nn.Parameter(torch.tensor(float(beta)).log())
        self.register_buffer("memory_labels", memory_labels.detach().clone().long())
        self.register_buffer("center", center.detach().clone().float())
        self.num_classes = int(memory_labels.max().item()) + 1
        self.center_ema = center_ema
        self.track_center = False

    @property
    def beta(self) -> float:
        return float(self.log_beta.exp().item())

    def macs_per_image(self) -> int:
        k, d = self.memory.shape
        return 2 * k * d + d

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.training and self.track_center:
            with torch.no_grad():
                self.center.mul_(self.center_ema).add_(z.mean(dim=0) * (1.0 - self.center_ema))
        memory = self.memory - self.center
        cosine = F.normalize(z - self.center, dim=-1) @ F.normalize(memory, dim=-1).T
        weights = torch.softmax(self.log_beta.exp() * cosine, dim=-1)
        h = weights @ memory + self.center
        class_mass = weights.new_zeros(weights.shape[0], self.num_classes)
        class_mass = class_mass.index_add(1, self.memory_labels, weights)
        return h, {
            "cosine": cosine,
            "weights": weights,
            "class_mass": class_mass,
            "proto_pred": class_mass.argmax(dim=-1),
        }


class PlacedNet(nn.Module):
    """L1-L5 -> optional memory -> L6-L7 -> optional memory -> L9; placement is 'none', 'l5' or 'l8'.

    Returns ``(logits, z, h, aux)``. ``forward_from_embedding`` takes L5 features, and
    is named to work with ``forward_embeddings``.
    """

    def __init__(self, encoder: MnistEncoder, head: nn.Module, bridge: TrainableHopfieldLayer | None, placement: str) -> None:
        super().__init__()
        if placement not in ("none", "l5", "l8") or (bridge is None) != (placement == "none"):
            raise ValueError(f"placement {placement!r} does not match bridge {type(bridge).__name__}.")
        self.encoder = encoder
        self.head = head
        self.bridge = bridge
        self.placement = placement

    def forward_from_embedding(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        aux: dict[str, Any] = {}
        if self.placement == "l5":
            features, aux = self.bridge(features)
        z = embed_features(self.encoder, features)
        h = z
        if self.placement == "l8":
            h, aux = self.bridge(z)
        return self.head(h), z, h, aux

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.forward_from_embedding(flatten_features(self.encoder, images))


def set_trainable(net: PlacedNet, *, memory: bool = False, finetune: bool = False, convs: bool = False) -> None:
    """Choose what trains: the memory and beta, L6-L9, and L1-L3's convolutions."""
    enc = net.encoder
    for module, flag in ((enc.conv1, convs), (enc.conv2, convs), (enc.fc1, finetune), (enc.fc2, finetune), (net.head, finetune)):
        module.requires_grad_(flag)
    if net.bridge is not None:
        net.bridge.requires_grad_(memory)


def snap_to_real_features(
    layer: TrainableHopfieldLayer, features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Nearest same-class training feature per prototype, by centered cosine, and the mean cosine."""
    center = layer.center.cpu()
    memory = F.normalize(layer.memory.detach().cpu() - center, dim=-1)
    snapped, best = [], []
    for proto, label in zip(memory, layer.memory_labels.cpu()):
        candidates = features[labels == label]
        cosine = F.normalize(candidates - center, dim=-1) @ proto
        snapped.append(candidates[cosine.argmax()])
        best.append(cosine.max().item())
    return torch.stack(snapped), sum(best) / len(best)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def augment_batch(
    images: torch.Tensor, gaussian: Sequence[float], salt_pepper: Sequence[float], generator: torch.Generator
) -> torch.Tensor:
    """First third clean, second Gaussian, last salt-and-pepper, at one drawn level each."""
    n = images.shape[0]
    a, b = n // 3, 2 * n // 3
    out = images.clone()
    sigma = gaussian[int(torch.randint(len(gaussian), (1,), generator=generator))]
    prob = salt_pepper[int(torch.randint(len(salt_pepper), (1,), generator=generator))]
    out[a:b] = corrupt_images(images[a:b], "gaussian", sigma, generator)
    out[b:] = corrupt_images(images[b:], "salt_pepper", prob, generator)
    return out


def memory_loss(logits: torch.Tensor, aux: dict[str, Any], labels: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Cross-entropy, plus cluster and separation terms on the centered cosines when there is a memory."""
    loss = F.cross_entropy(logits, labels)
    if "cosine" not in aux:
        return loss
    cosine = aux["cosine"]
    own = aux["memory_labels"][None, :] == labels[:, None]
    closest_own = cosine.masked_fill(~own, -2.0).max(dim=1).values
    closest_other = cosine.masked_fill(own, -2.0).max(dim=1).values
    return loss + cfg["cluster_weight"] * (1.0 - closest_own).mean() + cfg["separation_weight"] * closest_other.mean()


def validation_score(net: PlacedNet, val: dict[str, Any], *, shared: bool, batch_size: int, device: torch.device) -> float:
    """Mean head accuracy over the validation selection sets."""
    accs = []
    for images, features in zip(val["images"], val["features"]):
        inputs = features if shared else encode_features(net.encoder, images, batch_size=batch_size, device=device)
        out = forward_embeddings(net, inputs, batch_size=batch_size, device=device)
        accs.append((out["head_pred"] == val["labels"]).float().mean().item())
    return sum(accs) / len(accs)


def train_stage(
    net: PlacedNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    val: dict[str, Any],
    cfg: dict[str, Any],
    *,
    epochs: int,
    memory_lr: float,
    finetune: bool,
    convs: bool = False,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train on noisy batches and restore the best epoch by validation score (0 = no training)."""
    has_memory = net.bridge is not None
    set_trainable(net, memory=has_memory, finetune=finetune, convs=convs)
    groups: list[dict[str, Any]] = []
    if has_memory:
        groups += [{"params": [net.bridge.memory], "lr": memory_lr}, {"params": [net.bridge.log_beta], "lr": cfg["beta_lr"]}]
    backbone = [p for name, p in net.named_parameters() if p.requires_grad and not name.startswith("bridge.")]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["backbone_lr"]})
    optimizer = torch.optim.Adam(groups)
    generator = torch.Generator().manual_seed(seed)
    batch_size, eval_batch = cfg["batch_size"], cfg["eval_batch_size"]

    best_score = validation_score(net, val, shared=not convs, batch_size=eval_batch, device=device)
    best_state, best_epoch = copy.deepcopy(net.state_dict()), 0
    history = [best_score]
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(images.shape[0], generator=generator)
        total_loss, total = 0.0, 0
        for i in range(0, images.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            x = augment_batch(images[idx], cfg["train_gaussian_sigmas"], cfg["train_salt_pepper_probs"], generator).to(device)
            y = labels[idx].to(device)
            with torch.set_grad_enabled(convs):
                features = flatten_features(net.encoder, x)
            logits, _, _, aux = net.forward_from_embedding(features)
            if has_memory:
                aux = {**aux, "memory_labels": net.bridge.memory_labels}
            loss = memory_loss(logits, aux, y, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
            total += idx.numel()
        net.eval()
        score = validation_score(net, val, shared=not convs, batch_size=eval_batch, device=device)
        history.append(score)
        print(f"      epoch {epoch}/{epochs}  loss {total_loss / max(total, 1):.4f}  val {score:.4f}", flush=True)
        if score > best_score:
            best_score, best_state, best_epoch = score, copy.deepcopy(net.state_dict()), epoch
    net.load_state_dict(best_state)
    net.eval()
    set_trainable(net)
    return {"best_epoch": best_epoch, "best_val": best_score, "val_history": history}


def validation_sets(cfg: dict[str, Any], run: BaselineRun) -> dict[str, Any]:
    images = [run.val_images(kind, level) for kind, level in cfg["beta_selection_sets"]]
    features = [encode_features(run.net.encoder, x, batch_size=run.batch_size, device=run.device) for x in images]
    return {"images": images, "features": features, "labels": run.splits.val_labels}


def make_memory_net(
    run: BaselineRun, placement: str, inputs: torch.Tensor, cfg: dict[str, Any], *, beta: float
) -> PlacedNet:
    memory, labels, center = build_prototype_memory(
        inputs, run.splits.train_labels, per_class=cfg["prototypes_per_class"], center=True, seed=run.seed
    )
    layer = TrainableHopfieldLayer(memory, labels, center, beta=beta, center_ema=cfg["center_ema"]).to(run.device)
    return PlacedNet(copy_net(run.net.encoder), copy_net(run.net.head), layer, placement).eval()


def tune_memory(cfg: dict[str, Any], data: Sequence[torch.Tensor], device: torch.device, output: Path) -> dict[str, Any]:
    """Choose the memory lr and initial beta per placement, on validation data only; cached."""
    path = output / "memory_tuning.json"
    fingerprint = {k: cfg[k] for k in ("tuning", "beta_l5", "beta_l8", "train_gaussian_sigmas", "train_salt_pepper_probs",
                                       "cluster_weight", "separation_weight", "beta_lr")}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached["fingerprint"] == fingerprint:
            print(f"Memory tuning: loaded cached result {path}", flush=True)
            return cached
    tuning = cfg["tuning"]
    run = prepare_baseline_run(cfg, data, tuning["seed"], device)
    val = validation_sets(cfg, run)
    train_features = encode_features(run.net.encoder, run.splits.train_images, batch_size=run.batch_size, device=device)
    limit = tuning["train_images"]
    result: dict[str, Any] = {"fingerprint": fingerprint, "grid": {}, "best": {}}
    for placement, inputs, beta in (("l5", train_features, cfg["beta_l5"]), ("l8", run.z_train, cfg["beta_l8"])):
        start = make_memory_net(run, placement, inputs, cfg, beta=beta)
        grid = []
        for lr, factor in itertools.product(tuning["memory_lr_values"], tuning["beta_factors"]):
            net = copy.deepcopy(start)
            net.bridge.log_beta.data = torch.tensor(float(beta * factor)).log().to(device)
            stage = train_stage(net, run.splits.train_images[:limit], run.splits.train_labels[:limit], val, cfg,
                                epochs=tuning["epochs"], memory_lr=lr, finetune=False, seed=tuning["seed"], device=device)
            grid.append({"memory_lr": lr, "beta_factor": factor, "initial_beta": beta * factor, "val": stage["best_val"]})
            print(f"  tuning {placement}: lr {lr:g}, beta {beta * factor:g}: val {stage['best_val']:.4f}", flush=True)
        result["grid"][placement] = grid
        result["best"][placement] = max(grid, key=lambda g: g["val"])
    output.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------


def evaluate_set(run: BaselineRun, nets: dict[str, PlacedNet], kind: str, level: float, *, full_model: str) -> dict[str, Any]:
    """Metrics per model and paired McNemar tests on one test set."""
    images = test_images_for(run, kind, level)
    kw = {"batch_size": run.batch_size, "device": run.device}
    shared = encode_features(run.net.encoder, images, **kw)
    metrics, correct = {}, {}
    for name, net in nets.items():
        inputs = encode_features(net.encoder, images, **kw) if name == full_model else shared
        out = forward_embeddings(net, inputs, **kw)
        values, flags, _ = summarize_outputs(out, run.splits.test_labels, run.prototypes, run.memory_labels)
        metrics[name] = {m: values[m] for m in METRICS}
        correct[name] = flags["head"]
    mcnemar = {f"{a}|{b}": mcnemar_test(correct[b], correct[a]) for a, b in COMPARISONS}
    return {"value": level, "seed": run.seed, "metrics": metrics, "mcnemar": mcnemar}


def seed_sanity(
    run: BaselineRun, nets: dict[str, PlacedNet], cfg: dict[str, Any], rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    enc = run.net.encoder
    conv_keys = [k for k in enc.state_dict() if k.startswith("conv")]
    frozen = [n for n in nets if n != "baseline_aug_full"]
    frozen_ok = all(
        torch.equal(nets[n].encoder.state_dict()[k], enc.state_dict()[k]) for n in frozen for k in conv_keys
    )
    trained = nets["trainable_l5"]
    with torch.inference_mode():
        logits = trained.head(embed_features(trained.encoder, trained.bridge.memory))
        own_class = (logits.argmax(dim=-1) == trained.bridge.memory_labels).float().mean().item()
        x = run.splits.test_images[:64].to(run.device)
        repeatable = all(torch.equal(n(x)[0], n(x)[0]) for n in nets.values())

    prior = cfg.get("prior_rows", {}).get(run.seed)
    matches: dict[str, float] = {}
    if prior:
        now = {r["value"]: r["metrics"] for r in rows["noise_gaussian"]}
        for level, old in prior.items():
            matches["baseline"] = max(matches.get("baseline", 0.0), abs(now[level]["baseline"]["head_acc"] - old["baseline"]))
            matches["fixed_l5"] = max(matches.get("fixed_l5", 0.0), abs(now[level]["fixed_l5"]["head_acc"] - old["hopfield_l5"]))
    return {"convs_frozen": frozen_ok, "prototype_own_class": own_class, "repeatable": repeatable, "prior_gap": matches}


def run_seed(seed: int, cfg: dict[str, Any], data: Sequence[torch.Tensor], tuning: dict[str, Any], device: torch.device,
             *, measure_cost: bool, output: Path) -> dict[str, Any]:
    start = time.perf_counter()
    run = prepare_baseline_run(cfg, data, seed, device)
    images, labels = run.splits.train_images, run.splits.train_labels
    val = validation_sets(cfg, run)
    train_features = encode_features(run.net.encoder, images, batch_size=run.batch_size, device=device)
    print(f"  seed {seed}: building memory", flush=True)
    kw: dict[str, Any] = {"cfg": cfg, "epochs": cfg["finetune_epochs"], "seed": seed, "device": device}

    def as_placed() -> PlacedNet:
        return PlacedNet(copy_net(run.net.encoder), copy_net(run.net.head), None, "none").eval()

    nets: dict[str, PlacedNet] = {"baseline": as_placed()}
    stages: dict[str, Any] = {}

    print("    baseline+noise (L6-L9)", flush=True)
    nets["baseline_aug"] = as_placed()
    stages["baseline_aug"] = train_stage(nets["baseline_aug"], images, labels, val, memory_lr=0.0, finetune=True, **kw)
    print("    baseline+noise (L1-L9)", flush=True)
    nets["baseline_aug_full"] = as_placed()
    stages["baseline_aug_full"] = train_stage(
        nets["baseline_aug_full"], images, labels, val, memory_lr=0.0, finetune=True, convs=True, **kw
    )

    nets["fixed_l5"] = make_memory_net(run, "l5", train_features, cfg, beta=cfg["beta_l5"])
    initial_memory = nets["fixed_l5"].bridge.memory.detach().clone()

    def two_stages(placement: str, inputs: torch.Tensor) -> tuple[PlacedNet, PlacedNet]:
        best = tuning["best"][placement]
        net = make_memory_net(run, placement, inputs, cfg, beta=best["initial_beta"])
        stages[f"{placement}_a"] = train_stage(
            net, images, labels, val, cfg, epochs=cfg["memory_epochs"], memory_lr=best["memory_lr"],
            finetune=False, seed=seed, device=device,
        )
        after_a = copy.deepcopy(net)
        # Only in Stage B can L6-L7 move z away from the L8 center.
        net.bridge.track_center = placement == "l8"
        stages[f"{placement}_b"] = train_stage(net, images, labels, val, memory_lr=best["memory_lr"], finetune=True, **kw)
        net.bridge.track_center = False
        return after_a, net

    print("    trainable L5", flush=True)
    nets["trainable_l5_a"], nets["trainable_l5"] = two_stages("l5", train_features)
    print("    trainable L8", flush=True)
    _, nets["trainable_l8"] = two_stages("l8", run.z_train)

    snapped_memory, nearest_real = snap_to_real_features(nets["trainable_l5"].bridge, train_features, labels)
    nets["trainable_l5_snapped"] = copy.deepcopy(nets["trainable_l5"])
    nets["trainable_l5_snapped"].bridge.memory.data = snapped_memory.to(device)
    center = nets["trainable_l5"].bridge.center
    drift = F.cosine_similarity(nets["trainable_l5"].bridge.memory.detach() - center, initial_memory - center, dim=-1)

    nets = {name: nets[name] for name in MODELS}
    for net in nets.values():
        net.eval()
    print(f"    evaluating ({time.perf_counter() - start:.0f}s so far)", flush=True)
    rows: dict[str, list[dict[str, Any]]] = {}
    grids = noise_grids(cfg)
    for kind in NOISE_TYPES:
        rows[f"noise_{kind}"] = [
            evaluate_set(run, nets, kind, level, full_model="baseline_aug_full") for level in grids[kind]
        ]

    result: dict[str, Any] = {
        "seed": seed,
        "rows": rows,
        "stages": stages,
        "learned_beta": {
            "fixed_l5": nets["fixed_l5"].bridge.beta,
            "trainable_l5_a": nets["trainable_l5_a"].bridge.beta,
            "trainable_l5": nets["trainable_l5"].bridge.beta,
            "trainable_l8": nets["trainable_l8"].bridge.beta,
        },
        "memory_drift_cosine": drift.mean().item(),
        "nearest_real_cosine": nearest_real,
        "sanity": seed_sanity(run, nets, cfg, rows),
        "wall_time_s": time.perf_counter() - start,
    }
    if measure_cost:
        result["cost"] = measure_costs(cfg, nets, device)
    torch.save(
        {name: nets[name].state_dict() for name in ("baseline_aug", "baseline_aug_full", "trainable_l5", "trainable_l8")},
        output / f"trained_seed{seed}.pt",
    )
    return result


def measure_costs(cfg: dict[str, Any], nets: dict[str, PlacedNet], device: torch.device) -> dict[str, Any]:
    names = ("baseline", "baseline_aug_full", "fixed_l5", "trainable_l5", "trainable_l8")
    chosen = {n: nets[n] for n in names}
    base_macs = sum(m for _, m in analytical_macs(_as_embedding_net(nets["baseline"])))
    c = cfg["cost_settings"]
    latency = interleaved_cost(chosen, c["batch_size"], device, repeats=c["repeats"], warmup=c["warmup"], iters=c["iters"])
    return {
        "batch_size": c["batch_size"],
        "weights": {n: sum(p.numel() for p in net.parameters()) for n, net in chosen.items()},
        "model_size_bytes": {n: model_size_bytes(net) for n, net in chosen.items()},
        "macs": {n: base_macs + (net.bridge.macs_per_image() if net.bridge is not None else 0) for n, net in chosen.items()},
        "latency_ms_p50": {n: latency[n]["latency_ms_p50"]["median"] for n in names},
        "throughput_img_s": {n: latency[n]["throughput_img_s"]["median"] for n in names},
    }


def _as_embedding_net(net: PlacedNet) -> nn.Module:
    """``analytical_macs`` reads encoder, head and bridge; an Identity bridge adds no MACs."""
    holder = nn.Module()
    holder.encoder, holder.head, holder.bridge = net.encoder, net.head, nn.Identity()
    return holder


def run_experiment(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    threads = configure_threads(4)
    device = resolve_device(cfg["device"])
    env = environment_info(device)
    print(f"Device: {device}  threads: {threads}  torch {torch.__version__}", flush=True)
    data = load_mnist(cfg["data_dir"])
    apply_cached_tuning(cfg)
    for seed in cfg["seeds"]:
        require_cached_backbone(cfg, data, seed)
    tuning = tune_memory(cfg, data, device, output)
    print(f"Chosen settings: {tuning['best']}", flush=True)

    seeds_dir = output / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    settings_key = {k: cfg[k] for k in RESUME_KEYS} | {"best": tuning["best"]}
    for i, seed in enumerate(cfg["seeds"]):
        path = seeds_dir / f"seed{seed}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                finished = json.load(f)
            if finished["settings"] == settings_key:
                print(f"  seed {seed}: already finished, reusing {path}", flush=True)
                seed_results.append(finished)
                continue
            print(f"  seed {seed}: saved result has different settings; running again", flush=True)
        result = run_seed(seed, cfg, data, tuning, device, measure_cost=i == 0, output=output)
        result["settings"] = settings_key
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        seed_results.append(result)
        print(f"  seed {seed}: done in {result['wall_time_s'] / 60:.1f} min", flush=True)
        gc.collect()
    settings = {k: v for k, v in cfg.items() if k != "prior_rows"}
    return {"config": settings, "environment": env, "tuning": tuning, "seeds": seed_results}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def rows_of(raw: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [row for result in raw["seeds"] for row in result["rows"][f"noise_{kind}"]]


def seed_values(rows: list[dict[str, Any]], value: Any, model: str, metric: str = "head_acc") -> list[float]:
    point = sorted((r for r in rows if r["value"] == value), key=lambda r: r["seed"])
    return [r["metrics"][model][metric] for r in point]


def auc_values(rows: list[dict[str, Any]], model: str) -> list[float]:
    levels = values_in_order(rows)
    by_level = [seed_values(rows, level, model) for level in levels]
    return [accuracy_auc_over_severity(levels, list(col)) for col in zip(*by_level)]


def clean_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in rows_of(raw, "gaussian") if r["value"] == 0]


def compare_values(a: list[float], b: list[float]) -> dict[str, Any]:
    return compare([x - y for x, y in zip(a, b)])


def winner(stats: dict[str, Any], a: str, b: str) -> str:
    if not stats["significant"]:
        return "tie"
    return MODEL_LABELS[a] if stats["delta"] > 0 else MODEL_LABELS[b]


def primary_results(raw: dict[str, Any]) -> dict[str, Any]:
    """AUC per noise type and model; success test with Holm over the two main corruptions."""
    out: dict[str, Any] = {}
    for kind in NOISE_TYPES:
        rows = rows_of(raw, kind)
        aucs = {m: auc_values(rows, m) for m in MODELS}
        out[kind] = {
            "auc": {m: mean_std(a) for m, a in aucs.items()},
            "comparisons": {f"{a}|{b}": compare_values(aucs[a], aucs[b]) for a, b in COMPARISONS},
        }
    key = "trainable_l5|baseline_aug"
    p_values = [out[k]["comparisons"][key]["wilcoxon_p"] for k in PRIMARY_KINDS]
    if all(p is not None for p in p_values):
        for kind, p in zip(PRIMARY_KINDS, holm_adjusted_p_values(p_values)):
            out[kind]["comparisons"][key]["holm_p"] = p
    return out


def success_test(raw: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    key = "trainable_l5|baseline_aug"
    rows = clean_rows(raw)
    clean = compare_values(seed_values(rows, 0, "trainable_l5"), seed_values(rows, 0, "baseline_aug"))
    results = {"clean": {"delta": clean["delta"], "passed": clean["delta"] >= -CLEAN_MARGIN}}
    for kind in PRIMARY_KINDS:
        c = primary[kind]["comparisons"][key]
        holm = c.get("holm_p")
        results[kind] = {"delta": c["delta"], "holm_p": holm,
                         "passed": holm is not None and holm < ALPHA and c["delta"] >= WINNER_DELTA}
    results["passed"] = all(results[k]["passed"] for k in ("clean", *PRIMARY_KINDS))
    return results


def sanity_checks(raw: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    seeds = raw["seeds"]
    s = [r["sanity"] for r in seeds]
    n = len(s)
    checks = [
        ("L1–L3 unchanged in every model but Baseline+noise (L1–L9)", "all seeds", f"{sum(x['convs_frozen'] for x in s)}/{n}",
         all(x["convs_frozen"] for x in s)),
        ("Trained L5 prototypes through L6–L9 predict their own class", "≥ 95 %",
         f"{100 * min(x['prototype_own_class'] for x in s):.1f} %", min(x["prototype_own_class"] for x in s) >= 0.95),
        ("Models give identical outputs when run twice", "all seeds", f"{sum(x['repeatable'] for x in s)}/{n}",
         all(x["repeatable"] for x in s)),
    ]
    gaps = [x["prior_gap"] for x in s if x["prior_gap"]]
    if gaps:
        for model in ("baseline", "fixed_l5"):
            worst = max(g[model] for g in gaps)
            checks.append((f"{MODEL_LABELS[model]} repeats the L5 experiment's accuracy", "gap ≤ 0.1 pp",
                           f"{100 * worst:.3f} pp", worst <= 0.001))
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    primary = primary_results(raw)
    test = success_test(raw, primary)
    seeds = raw["seeds"]
    grid_models = ("baseline", "baseline_aug", "baseline_aug_full", "fixed_l5", "trainable_l5", "trainable_l8")
    testable = test["gaussian"]["holm_p"] is not None
    outcome = ("PASSED" if test["passed"] else "FAILED") if testable else "NOT TESTABLE (too few seeds)"
    verdict = (
        f"Success test {outcome}: trainable L5 vs Baseline+noise (L6–L9) "
        f"clean {_pp(test['clean']['delta'])} pp, Gaussian AUC {_pp(test['gaussian']['delta'])} pp "
        f"(Holm p={_p(test['gaussian']['holm_p'])}), salt-and-pepper AUC {_pp(test['salt_pepper']['delta'])} pp "
        f"(Holm p={_p(test['salt_pepper']['holm_p'])})."
    )
    lines = [
        "# MNIST: trainable Hopfield memory", "", f"**Verdict:** {verdict}", "",
        (f"{len(seeds)} seeds. Accuracy values are % mean ± std across seeds; Δ is in percentage points. "
        "`tie` means |Δ| < 0.5 pp or Wilcoxon p ≥ 0.05."), "",
        "## 1. Success test", "",
        "Fixed before running: within 0.5 pp on clean accuracy, and significantly better on both main AUCs.", "",
    ]
    lines += _table(["Requirement", "Δ vs Baseline+noise (L6–L9)", "Holm p", "Result"], [
        ["Clean within 0.5 pp", _pp(test["clean"]["delta"]), "—", "PASS" if test["clean"]["passed"] else "FAIL"],
        ["Better Gaussian AUC", _pp(test["gaussian"]["delta"]), _p(test["gaussian"]["holm_p"]), "PASS" if test["gaussian"]["passed"] else "FAIL"],
        ["Better salt-and-pepper AUC", _pp(test["salt_pepper"]["delta"]), _p(test["salt_pepper"]["holm_p"]), "PASS" if test["salt_pepper"]["passed"] else "FAIL"],
    ])

    lines += ["## 2. All models", "", "Occlusion was never trained on, so it is an unseen corruption.", ""]
    clean = clean_rows(raw)
    table = []
    for m in MODELS:
        table.append([MODEL_LABELS[m], _pct(mean_std(seed_values(clean, 0, m))),
                      *[_pct(primary[k]["auc"][m]) for k in NOISE_TYPES]])
    lines += _table(["Model", "Clean", "Gaussian AUC", "Salt-and-pepper AUC", "Occlusion AUC"], table)

    lines += ["## 3. Paired comparisons", ""]
    table = []
    for a, b in COMPARISONS:
        ca = compare_values(seed_values(clean, 0, a), seed_values(clean, 0, b))
        row = [f"{MODEL_LABELS[a]} − {MODEL_LABELS[b]}", _pp(ca["delta"]), f"{ca['plus']}/{ca['minus']}", _p(ca["wilcoxon_p"])]
        for kind in NOISE_TYPES:
            c = primary[kind]["comparisons"][f"{a}|{b}"]
            row += [_pp(c["delta"]), f"{c['plus']}/{c['minus']}", _p(c["wilcoxon_p"])]
        table.append(row)
    lines += _table(["Comparison", "Clean Δ", "+/−", "p", "Gaussian Δ", "+/−", "p", "S&P Δ", "+/−", "p", "Occlusion Δ", "+/−", "p"], table)

    lines += ["## 4. Accuracy at each noise level", ""]
    for kind in NOISE_TYPES:
        rows = rows_of(raw, kind)
        lines += [f"**{kind}** ({NOISE_LABELS[kind]})", ""]
        table = []
        for level in values_in_order(rows):
            c = compare_values(seed_values(rows, level, "trainable_l5"), seed_values(rows, level, "baseline_aug"))
            table.append([f"{level:g}", *[_pct(mean_std(seed_values(rows, level, m))) for m in grid_models],
                          _pp(c["delta"]), winner(c, "trainable_l5", "baseline_aug")])
        lines += _table([NOISE_LABELS[kind], *[MODEL_LABELS[m] for m in grid_models], "Δ Trainable L5 − Baseline+noise (L6–L9)", "Winner"], table)

    lines += ["## 5. What training changed", ""]
    table = []
    for name in ("fixed_l5", "trainable_l5_a", "trainable_l5", "trainable_l8"):
        vals = [r["learned_beta"][name] for r in seeds]
        table.append([MODEL_LABELS[name], f"{sum(vals) / len(vals):.1f}", f"{min(vals):.1f}–{max(vals):.1f}"])
    lines += _table(["Model", "Final β (mean)", "Range across seeds"], table)
    drift = mean_std([r["memory_drift_cosine"] for r in seeds])
    real = mean_std([r["nearest_real_cosine"] for r in seeds])
    lines += [(f"Centered cosine between trained and k-means L5 prototypes: {drift[0]:.3f} ± {drift[1]:.3f}. "
              f"Between trained prototypes and their nearest real training feature: {real[0]:.3f} ± {real[1]:.3f}."), ""]
    table = []
    for stage, label in (("baseline_aug", "Baseline+noise (L6–L9)"), ("baseline_aug_full", "Baseline+noise (L1–L9)"),
                         ("l5_a", "Trainable L5, Stage A"), ("l5_b", "Trainable L5, Stage B"),
                         ("l8_a", "Trainable L8, Stage A"), ("l8_b", "Trainable L8, Stage B")):
        epochs = [r["stages"][stage]["best_epoch"] for r in seeds]
        start = [r["stages"][stage]["val_history"][0] for r in seeds]
        best = [r["stages"][stage]["best_val"] for r in seeds]
        table.append([label, _pct(mean_std(start)), _pct(mean_std(best)), ", ".join(str(e) for e in epochs)])
    lines += _table(["Stage", "Validation before", "Best validation", "Best epoch per seed (0 = no training)"], table)

    lines += ["## 6. Memory tuning", "", f"Seed {raw['config']['tuning']['seed']}, validation data only, memory-only training.", ""]
    table = []
    for placement, grid in raw["tuning"]["grid"].items():
        for g in grid:
            chosen = "yes" if g == raw["tuning"]["best"][placement] else ""
            table.append([placement.upper(), f"{g['memory_lr']:g}", f"{g['initial_beta']:g}", f"{100 * g['val']:.2f}", chosen])
    lines += _table(["Placement", "Memory lr", "Initial β", "Validation %", "Chosen"], table)

    cost = next((r["cost"] for r in seeds if "cost" in r), None)
    if cost:
        names = list(cost["macs"])
        base = cost["macs"]["baseline"]
        lines += ["## 7. Cost", ""]
        lines += _table(["Measure", *[MODEL_LABELS[n] for n in names]], [
            ["Weights and memory values", *[f"{cost['weights'][n]:,}" for n in names]],
            ["Saved model size (bytes)", *[f"{cost['model_size_bytes'][n]:,}" for n in names]],
            ["MACs per image", *[f"{cost['macs'][n]:,}" for n in names]],
            ["MACs vs Baseline", *[f"{100 * (cost['macs'][n] - base) / base:+.2f} %" for n in names]],
            [f"Latency, batch {cost['batch_size']} (ms, p50)", *[f"{cost['latency_ms_p50'][n]:.1f}" for n in names]],
            [f"Throughput, batch {cost['batch_size']} (images/s)", *[f"{cost['throughput_img_s'][n]:,.0f}" for n in names]],
        ])
        lines += ["Timings come from one seed on this machine and only compare within this run.", ""]

    checks = sanity_checks(raw)
    lines += ["## 8. Sanity checks", ""]
    lines += _table(["Check", "Expected", "Measured", "Result"], [[c, e, m, "PASS" if ok else "FAIL"] for c, e, m, ok in checks])
    summary = {"verdict": verdict, "success_test": test, "primary": primary,
               "sanity_checks": [dict(zip(("check", "expected", "measured", "passed"), c)) for c in checks]}
    return "\n".join(lines), summary


def save_reports(raw: dict[str, Any], text: str, summary: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(output / "results.json", "w", encoding="utf-8") as f:
        json.dump({**raw, "summary": summary}, f, indent=2, default=str)
    rows = [
        {"noise": kind, "level": row["value"], "seed": row["seed"], "model": model, **row["metrics"][model]}
        for kind in NOISE_TYPES for row in rows_of(raw, kind) for model in MODELS
    ]
    with open(output / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(raw: dict[str, Any], path: Path) -> bool:
    plt = _import_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    titles = {"gaussian": "Gaussian noise", "salt_pepper": "Salt-and-pepper noise", "occlusion": "Occlusion (never trained on)"}
    xlabels = {"gaussian": "Noise σ", "salt_pepper": "Corrupted pixel share p", "occlusion": "Square side (pixels)"}
    for ax, kind in zip(axes.flat, NOISE_TYPES):
        rows = rows_of(raw, kind)
        levels = values_in_order(rows)
        for model in PLOT_MODELS:
            mean = [100 * mean_std(seed_values(rows, level, model))[0] for level in levels]
            ax.plot(levels, mean, color=MODEL_COLORS[model], linewidth=2, marker=MODEL_MARKERS[model], markersize=7,
                    markeredgecolor="white", markeredgewidth=1, label=MODEL_LABELS[model])
        _style(ax, titles[kind], xlabels[kind], "Test accuracy (%)")
        ax.set_ylim(0, 100)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8)

    ax = axes[1, 1]
    rows = clean_rows(raw)
    means = [100 * mean_std(seed_values(rows, 0, m))[0] for m in PLOT_MODELS]
    positions = list(range(len(PLOT_MODELS)))[::-1]
    ax.barh(positions, means, height=0.6, color=[MODEL_COLORS[m] for m in PLOT_MODELS])
    ax.set_yticks(positions, [MODEL_LABELS[m] for m in PLOT_MODELS])
    for y, value in zip(positions, means):
        ax.text(value + 0.05, y, f"{value:.2f}", va="center", fontsize=9, color="#3a3a36")
    _style(ax, "Clean accuracy (axis starts at 90 %)", "Test accuracy (%)", "")
    ax.set_xlim(90, 100.5)
    ax.grid(False, axis="y")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Configuration and entry point
# ---------------------------------------------------------------------------


def load_prior_rows(path: Path) -> tuple[float, float, dict[int, dict[float, dict[str, float]]]]:
    """Fixed betas and the L5 experiment's Gaussian accuracies per seed, to check they repeat."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run benchmarks.mnist_l5_hopfield first.")
    with open(path, encoding="utf-8") as f:
        prior = json.load(f)
    rows: dict[int, dict[float, dict[str, float]]] = {}
    for row in prior["results"]["noise_gaussian"]:
        rows.setdefault(row["seed"], {})[row["value"]] = {
            "baseline": row["metrics"]["baseline"]["head_acc"],
            "hopfield_l5": row["metrics"]["hopfield_l5"]["head_acc"],
        }
    return prior["beta_l5"], prior["beta_l8"], rows


def load_trainable_config(path: Path, *, quick: bool = False, seed: int | None = None, reports_dir: str | None = None) -> tuple[dict[str, Any], Path]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    overrides = raw.pop("quick", {}) if isinstance(raw.get("quick"), dict) else {}
    if quick:
        raw.update(overrides)
    cfg = load_config(Path(raw["source_config"]), quick=quick, seed=seed)
    output = Path(reports_dir or raw["reports_dir"])
    cfg["cost_settings"] = raw["cost"]
    for key in ("train_gaussian_sigmas", "train_salt_pepper_probs", "memory_epochs", "finetune_epochs", "backbone_lr",
                "beta_lr", "cluster_weight", "separation_weight", "center_ema", "tuning"):
        cfg[key] = raw[key]
    cfg["beta_l5"], cfg["beta_l8"], cfg["prior_rows"] = load_prior_rows(Path(raw["fixed_results"]))
    return cfg, output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true", help="1 seed, 1 epoch per stage; separate output folder.")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed.")
    parser.add_argument("--reports-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args(argv)

    cfg, output = load_trainable_config(args.config, quick=args.quick, seed=args.seed, reports_dir=args.reports_dir)
    start = time.perf_counter()
    raw = run_experiment(cfg, output)
    text, summary = build_report(raw)
    save_reports(raw, text, summary, output)
    plot_comparison(raw, output / "comparison.png")
    print(f"\n{summary['verdict']}\nWrote {output}/ in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

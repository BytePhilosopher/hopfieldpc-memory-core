"""CIFAR-10-C: independent test of test-batch recentering, Baseline vs Hopfield (Colab, GPU).

Same controlled design as ``benchmarks.mnist_prototype_retrieval``: one backbone,
and only layer L8 differs. The method was fixed before any CIFAR-10-C result
existed, so this run can confirm or reject what was found on MNIST-C:

    Main result: Hopfield recentered vs Baseline at batch 256, mean accuracy AUC
    over severities 0-5 (0 = clean) across the 15 standard corruptions, tested
    with Wilcoxon across seeds.

Everything is selected on validation data; corrupted test data is downloaded
only after training and beta selection are complete.

Colab setup (Runtime > Change runtime type > T4 GPU), one cell per block:

    # 1. Put the project on Colab: upload hopfieldpc_code.zip, or clone your branch.
    !unzip -q -o /content/hopfieldpc_code.zip -d /content/hopfieldpc-memory-core
    %cd /content/hopfieldpc-memory-core
    !pip install -q --no-deps -e .

    # 2. Keep checkpoints on Drive so a disconnect loses at most one seed.
    from google.colab import drive
    drive.mount("/content/drive")

    # 3. Short trial first, then the full run. Rerun the same command to resume.
    !python -m benchmarks.cifar10_c_retrieval --quick --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval_quick
    !python -m benchmarks.cifar10_c_retrieval --data-dir /content/data \
        --output-dir /content/drive/MyDrive/cifar10_c_retrieval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.mnist_baseline import (
    MNIST_C_UNCORRUPTED_NAME,
    BaselineRun,
    ClassifierHead,
    EmbeddingClassifier,
    accuracy_auc_over_severity,
    compute_prototypes,
    encode,
    environment_info,
    forward_embeddings,
    make_splits,
    mean_std,
)
from benchmarks.mnist_prototype_retrieval import (
    MIXED_CORRUPTIONS_NAME,
    BatchRecenteringLayer,
    HopfieldRetrievalLayer,
    _paired_summary,
    _recentering_rows,
    build_embedding_net,
    format_recentering_report,
    mix_corruptions,
    plot_recentering,
    select_beta,
    shared_weights_equal,
    summarize_recentering,
    validation_accuracy_by_beta,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
CIFAR10_C_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
IMAGES_PER_SEVERITY = 10_000
SEVERITIES = (1, 2, 3, 4, 5)
CIFAR10_C_CORRUPTION_GROUPS = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
CORRUPTION_NAMES = [name for names in CIFAR10_C_CORRUPTION_GROUPS.values() for name in names]
DETAIL_BATCH_SIZE = 256


@dataclass
class Cifar10CConfig:
    """Every setting of the run; a change to a training field invalidates saved backbones."""

    data_dir: str = "data"
    output_dir: str = "reports/cifar10_c_retrieval"
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    # Backbone (fixed recipe, no tuning grid: the comparison shares these weights).
    base_width: int = 64
    embed_dim: int = 64
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    val_size: int = 5000
    train_subset: int | None = None
    test_subset: int | None = None
    min_clean_val_accuracy: float = 0.90  # stop before any corrupted data is used
    # Memory and retrieval (same protocol as MNIST).
    prototypes_per_class: int = 8
    center_prototypes: bool = True
    beta_values: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    )
    hard_readout_beta: float = 1e4
    # Smaller than MNIST's levels because CIFAR images are already low-contrast colour images.
    beta_selection_sets: list[tuple[str, float]] = field(
        default_factory=lambda: [("clean", 0.0), ("gaussian", 0.06), ("salt_pepper", 0.02)]
    )
    recentering_batch_sizes: list[int] = field(default_factory=lambda: [16, 64, 256, 1000])
    eval_batch_size: int = 1000

    def training_fingerprint(self, seed: int) -> dict[str, Any]:
        """Settings a saved backbone must match to be reused."""
        return {
            "seed": seed, "base_width": self.base_width, "embed_dim": self.embed_dim,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "momentum": self.momentum, "weight_decay": self.weight_decay,
            "val_size": self.val_size, "train_subset": self.train_subset,
        }

    def evaluation_fingerprint(self, seed: int) -> dict[str, Any]:
        """Settings saved validation curves and per-seed results must match to be reused."""
        return {
            **self.training_fingerprint(seed),
            "test_subset": self.test_subset,
            "prototypes_per_class": self.prototypes_per_class,
            "center_prototypes": self.center_prototypes,
            "beta_values": self.beta_values,
            "hard_readout_beta": self.hard_readout_beta,
            "beta_selection_sets": [list(s) for s in self.beta_selection_sets],
            "recentering_batch_sizes": self.recentering_batch_sizes,
        }


def quick_config(cfg: Cifar10CConfig) -> Cifar10CConfig:
    """Short trial of the whole pipeline; its accuracy means nothing, so the quality stop is off."""
    return replace(
        cfg, seeds=[0], epochs=1, train_subset=5000, val_size=1000, test_subset=1000,
        min_clean_val_accuracy=0.0,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _pixels_to_tensor(pixels: np.ndarray) -> torch.Tensor:
    """uint8 images [N, H, W, C] -> float tensor [N, C, H, W] in [0, 1]."""
    return torch.from_numpy(np.array(pixels)).permute(0, 3, 1, 2).float().div(255.0)


def load_cifar10(data_dir: str | Path) -> tuple[torch.Tensor, ...]:
    """Load (train_images, train_labels, test_images, test_labels) as tensors."""
    import torchvision

    tensors: list[torch.Tensor] = []
    for train in (True, False):
        dataset = torchvision.datasets.CIFAR10(root=str(data_dir), train=train, download=True)
        tensors += [_pixels_to_tensor(dataset.data), torch.tensor(dataset.targets, dtype=torch.long)]
    return tuple(tensors)


def ensure_cifar10_c_downloaded(data_dir: str | Path) -> None:
    """Download and extract CIFAR-10-C (Hendrycks & Dietterich 2019) unless already present."""
    root = Path(data_dir) / "CIFAR-10-C"
    expected = [*CORRUPTION_NAMES, "labels"]
    if all((root / f"{name}.npy").exists() for name in expected):
        return
    from torchvision.datasets.utils import download_and_extract_archive

    download_and_extract_archive(
        CIFAR10_C_URL, download_root=str(data_dir), md5=CIFAR10_C_MD5, remove_finished=True
    )


def load_cifar10_c_test_set(
    data_dir: str | Path,
    corruption_name: str,
    severity: int,
    *,
    test_subset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One corruption at one severity: images [N, 3, 32, 32] in [0, 1] and labels [N].

    Each .npy file holds 10,000 images per severity, severity 1 first. The file
    is memory-mapped, so only the requested block is read into RAM.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}.")
    root = Path(data_dir) / "CIFAR-10-C"
    start = (severity - 1) * IMAGES_PER_SEVERITY
    stop = start + (IMAGES_PER_SEVERITY if test_subset is None else test_subset)
    images = np.load(root / f"{corruption_name}.npy", mmap_mode="r")[start:stop]
    labels = np.load(root / "labels.npy", mmap_mode="r")[start:stop]
    return _pixels_to_tensor(images), torch.from_numpy(np.array(labels)).long()


def random_crop_and_flip(images: torch.Tensor, *, padding: int = 4) -> torch.Tensor:
    """Standard CIFAR augmentation for a batch [B, C, H, W], one random crop and flip per image."""
    b, _, height, width = images.shape
    padded = F.pad(images, (padding,) * 4, mode="reflect")
    tops = torch.randint(0, 2 * padding + 1, (b,), device=images.device)
    lefts = torch.randint(0, 2 * padding + 1, (b,), device=images.device)
    rows = tops[:, None] + torch.arange(height, device=images.device)  # [B, H]
    cols = lefts[:, None] + torch.arange(width, device=images.device)  # [B, W]
    batch = torch.arange(b, device=images.device)[:, None, None]
    # Indices around the channel slice move it last: [B, H, W, C].
    cropped = padded[batch, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + self.shortcut(x))


class CifarResNetEncoder(nn.Module):
    """ResNet-18 (CIFAR variant, 3x3 stem, no max-pool): image [B, 3, 32, 32] in [0, 1] -> z [B, D].

    Inputs are normalized inside the forward pass, so corruptions work in raw
    pixel space, as with ``MnistEncoder``. Tests shrink ``base_width``.
    """

    def __init__(self, base_width: int = 64, embed_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("channel_means", torch.tensor(CIFAR10_MEAN)[None, :, None, None])
        self.register_buffer("channel_stds", torch.tensor(CIFAR10_STD)[None, :, None, None])
        self.embed_dim = embed_dim
        self.stem = nn.Conv2d(3, base_width, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(base_width)
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        stages: list[nn.Module] = []
        in_channels = base_width
        for stage_index, width in enumerate(widths):
            stride = 1 if stage_index == 0 else 2
            stages += [_ResidualBlock(in_channels, width, stride), _ResidualBlock(width, width, 1)]
            in_channels = width
        self.stages = nn.Sequential(*stages)
        self.embedding = nn.Linear(widths[-1], embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.channel_means) / self.channel_stds
        x = self.stages(F.relu(self.stem_norm(self.stem(x))))
        return self.embedding(F.adaptive_avg_pool2d(x, 1).flatten(1))


def train_backbone(
    net: EmbeddingClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: Cifar10CConfig,
    device: torch.device,
) -> float:
    """SGD with a per-step cosine schedule on clean augmented images; returns wall seconds."""
    start = time.perf_counter()
    use_amp = device.type == "cuda"
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(1, images.shape[0] // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs * steps_per_epoch)
    # torch.amp.GradScaler appeared in torch 2.3; older versions only have the cuda one.
    scaler_class = getattr(torch.amp, "GradScaler", None)
    scaler = scaler_class("cuda", enabled=use_amp) if scaler_class else torch.cuda.amp.GradScaler(enabled=use_amp)
    net.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(images.shape[0], device=device)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            index = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = net(random_crop_and_flip(images[index]))[0]
                loss = F.cross_entropy(logits, labels[index])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        print(f"    epoch {epoch + 1}/{cfg.epochs}  loss {total_loss / steps_per_epoch:.3f}"
              f"  {time.perf_counter() - start:.0f}s", flush=True)
    return time.perf_counter() - start


def _atomic_write(path: Path, write: Any) -> None:
    """Write through a temporary file, so a disconnect never leaves a half-written result."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def _save_json(path: Path, payload: Any) -> None:
    _atomic_write(path, lambda p: p.write_text(json.dumps(payload, default=str), encoding="utf-8"))


def _load_matching_json(path: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """Saved payload if it was produced with these settings, else None."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("fingerprint") == fingerprint else None


def prepare_seed_run(
    cfg: Cifar10CConfig,
    data: Sequence[torch.Tensor],
    seed: int,
    device: torch.device,
) -> BaselineRun:
    """Train or load one seed's backbone and embed its clean splits."""
    splits = make_splits(data, seed=seed, val_size=cfg.val_size, train_subset=cfg.train_subset,
                         test_subset=cfg.test_subset)
    encoder = CifarResNetEncoder(cfg.base_width, cfg.embed_dim)
    net = EmbeddingClassifier(encoder, nn.Identity(), ClassifierHead(cfg.embed_dim)).to(device)  # type: ignore[arg-type]
    path = Path(cfg.output_dir) / f"backbone_seed{seed}.pt"
    fingerprint = cfg.training_fingerprint(seed)
    saved = torch.load(path, map_location=device) if path.exists() else None
    from_cache = saved is not None and saved["fingerprint"] == fingerprint
    train_time_s = 0.0
    if from_cache:
        assert saved is not None
        net.load_state_dict(saved["state_dict"])
        train_time_s = saved["train_time_s"]
        print(f"  seed {seed}: loaded saved backbone")
    else:
        print(f"  seed {seed}: training for {cfg.epochs} epochs")
        torch.manual_seed(seed)
        train_time_s = train_backbone(net, splits.train_images, splits.train_labels, cfg, device)
        checkpoint = {"fingerprint": fingerprint, "state_dict": net.state_dict(), "train_time_s": train_time_s}
        _atomic_write(path, lambda p: torch.save(checkpoint, p))
    net.eval()

    start = time.perf_counter()
    z_train = encode(net.encoder, splits.train_images, batch_size=cfg.eval_batch_size, device=device)
    prototypes, memory_labels = compute_prototypes(z_train, splits.train_labels)
    prototype_time_s = time.perf_counter() - start
    z_val = encode(net.encoder, splits.val_images, batch_size=cfg.eval_batch_size, device=device)
    return BaselineRun(
        seed=seed, splits=splits, net=net, train_time_s=train_time_s, from_cache=from_cache,
        prototypes=prototypes, memory_labels=memory_labels, prototype_time_s=prototype_time_s,
        z_train=z_train, z_val=z_val, batch_size=cfg.eval_batch_size, device=device,
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def select_validation_beta(
    cfg: Cifar10CConfig,
    data: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[float, int, float]:
    """Train every seed, enforce the quality stop and select beta*; no corrupted test data is used.

    Returns (beta*, number of backbones loaded rather than trained, mean clean validation accuracy).
    """
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
    beta_star, _ = select_beta(cfg.beta_values, curves)
    return beta_star, num_loaded, sum(accuracies) / len(accuracies)


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
    if saved is not None:
        print(f"  seed {seed}: loaded saved results")
        return saved

    print(f"  seed {seed}: evaluating {len(CORRUPTION_NAMES)} corruptions at {len(SEVERITIES)} severities", flush=True)
    run = prepare_seed_run(cfg, data, seed, device)
    hopfield_net = build_embedding_net(run, cfg.prototypes_per_class, cfg.center_prototypes, beta=beta_star)
    bridge = hopfield_net.bridge
    assert isinstance(bridge, HopfieldRetrievalLayer) and bridge.center is not None
    recentered_baseline = EmbeddingClassifier(
        run.net.encoder, BatchRecenteringLayer(bridge.center), run.net.head  # type: ignore[arg-type]
    ).eval()

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


def run_experiment(cfg: Cifar10CConfig) -> tuple[dict[str, Any], str]:
    """Train, select beta on validation data, evaluate every seed, and write the reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  torch {torch.__version__}  output: {output_dir}")
    data = load_cifar10(cfg.data_dir)

    print("Stage 1: backbones and beta selection (validation data only)")
    beta_star, num_loaded, clean_val_accuracy = select_validation_beta(cfg, data, device)
    print(f"beta* = {beta_star:g}")

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
    parser.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 and CIFAR-10-C download dir.")
    parser.add_argument("--output-dir", type=str, default="reports/cifar10_c_retrieval",
                        help="Backbones, per-seed results and reports; rerun with the same dir to resume.")
    parser.add_argument("--quick", action="store_true", help="1 seed, 1 epoch, small subsets: a pipeline trial.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to run (default 0-9).")
    parser.add_argument("--allow-cpu", action="store_true", help="Run without a GPU (very slow for the full run).")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No GPU found. In Colab use Runtime > Change runtime type > T4 GPU, or pass --allow-cpu.")
    cfg = Cifar10CConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.quick:
        cfg = quick_config(cfg)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    run_experiment(cfg)


if __name__ == "__main__":
    main()

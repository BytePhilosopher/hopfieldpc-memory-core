"""CIFAR-10 baseline: ResNet-18 backbone, data loading and the shared run settings.

The baseline is the ResNet with an Identity bridge at L8. It is trained once
per seed on clean CIFAR-10 and saved as ``backbone_seed{seed}.pt`` in the
output directory; ``benchmarks.cifar10.tuning`` and ``benchmarks.cifar10.hopfield``
both load it from there. The training recipe is fixed (no tuning grid), so the
only tuned value on CIFAR-10 is the Hopfield beta*.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.mnist.baseline import (
    BaselineRun,
    ClassifierHead,
    EmbeddingClassifier,
    compute_prototypes,
    encode,
    make_splits,
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

TUNED_PARAMS_PATH = Path(__file__).resolve().with_name("tuned_params.json")
HOPFIELD_SECTION = "hopfield"


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
    # Written only by benchmarks.cifar10.tuning; the experiment reads it.
    tuned_params: str = str(TUNED_PARAMS_PATH)
    quick: bool = False

    def training_fingerprint(self, seed: int) -> dict[str, Any]:
        """Settings a saved backbone must match to be reused."""
        return {
            "seed": seed, "base_width": self.base_width, "embed_dim": self.embed_dim,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "momentum": self.momentum, "weight_decay": self.weight_decay,
            "val_size": self.val_size, "train_subset": self.train_subset,
        }

    def tuning_fingerprint(self) -> dict[str, Any]:
        """Settings the stored beta* depends on (every seed's backbone and the selection protocol)."""
        training = self.training_fingerprint(0)
        del training["seed"]
        return {
            **training,
            "seeds": self.seeds,
            "prototypes_per_class": self.prototypes_per_class,
            "center_prototypes": self.center_prototypes,
            "beta_values": self.beta_values,
            "beta_selection_sets": [list(s) for s in self.beta_selection_sets],
            "min_clean_val_accuracy": self.min_clean_val_accuracy,
        }

    def tuning_command(self) -> str:
        parts = ["python -m benchmarks.cifar10.tuning", f"--data-dir {self.data_dir}", f"--output-dir {self.output_dir}"]
        if self.quick:
            parts.append("--quick")
        if self.seeds != list(range(10)):
            parts.append("--seeds " + " ".join(str(s) for s in self.seeds))
        if Path(self.tuned_params).resolve() != TUNED_PARAMS_PATH.resolve():
            parts.append(f"--tuned-params {self.tuned_params}")
        return " ".join(parts)

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
        min_clean_val_accuracy=0.0, quick=True,
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


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI shared by ``benchmarks.cifar10.tuning`` and ``benchmarks.cifar10.hopfield``."""
    parser.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 and CIFAR-10-C download dir.")
    parser.add_argument("--output-dir", type=str, default="reports/cifar10_c_retrieval",
                        help="Backbones, per-seed results and reports; rerun with the same dir to resume.")
    parser.add_argument("--quick", action="store_true", help="1 seed, 1 epoch, small subsets: a pipeline trial.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to run (default 0-9).")
    parser.add_argument("--allow-cpu", action="store_true", help="Run without a GPU (very slow for the full run).")
    parser.add_argument("--tuned-params", type=str, default=None,
                        help="Tuned-parameter file (on Colab, point it at Drive so it survives the session).")


def config_from_args(args: argparse.Namespace) -> Cifar10CConfig:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No GPU found. In Colab use Runtime > Change runtime type > T4 GPU, or pass --allow-cpu.")
    cfg = Cifar10CConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.quick:
        cfg = quick_config(cfg)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.tuned_params is not None:
        cfg.tuned_params = args.tuned_params
    return cfg

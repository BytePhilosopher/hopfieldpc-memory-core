"""Task 13 non-Hopfield baseline: MNIST CNN with an Identity bridge at L8.

This module holds the baseline network and every piece shared with the
Hopfield comparison in ``benchmarks.mnist_prototype_retrieval``: config
validation, fast tensor data loading, seeded corruptions, the CNN layers,
training with a checkpoint cache, prototype construction, nearest-prototype
readout, metrics, and the cost/timing harness. It never imports the Hopfield
module (the dependency only goes the other way).

Architecture (L8 is the only slot that differs between the two variants):

    L1 Conv2d(1,32,3,pad=1)+ReLU -> L2 MaxPool2 -> L3 Conv2d(32,64,3,pad=1)+ReLU
    -> L4 MaxPool2 -> L5 Flatten(3136) -> L6 Linear(3136,128)+ReLU
    -> L7 Linear(128,64) = z -> L8 bridge (Identity here) = h -> L9 Linear(64,10)

Run the baseline alone (E1 and E3 baseline columns) from the repository root:

    python -m benchmarks.mnist_baseline --config configs/mnist_prototype_retrieval.yaml
"""

from __future__ import annotations

import argparse
import copy
import io
import itertools
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from benchmarks.validation import validate_beta_values
from hopfieldpc.api import validate_beta
from hopfieldpc.diagnostics import attention_entropy, retrieval_distance, topk_mass

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
NUM_CLASSES = 10
IMAGE_SIZE = 28

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "mnist_prototype_retrieval.yaml"
)
DEFAULT_REPORTS_DIR = "reports/mnist_prototype_retrieval"

NOISE_TYPES = ("gaussian", "salt_pepper", "occlusion")
NOISE_GRID_KEYS = {
    "gaussian": "gaussian_sigmas",
    "salt_pepper": "salt_pepper_probs",
    "occlusion": "occlusion_sizes",
}
PLACEMENTS = ("pixel", "embedding")
DEVICES = ("auto", "cpu", "cuda")

COST_KEYS = ("batch_size", "batch_sizes", "warmup", "iters", "repeats", "num_threads")
CONFIG_KEYS = (
    "seeds",
    "data_dir",
    "reports_dir",
    "device",
    "val_size",
    "train_subset",
    "test_subset",
    "epochs",
    "batch_size",
    "lr",
    "eval_batch_size",
    "beta_values",
    "s1_beta",
    "num_updates_values",
    "gaussian_sigmas",
    "salt_pepper_probs",
    "occlusion_sizes",
    "noisy_sigma",
    "placements",
    "center_prototypes",
    "prototypes_per_class",
    "beta_selection_sets",
    "baseline_tuning",
    "cost",
    "quick",
)
TUNING_KEYS = ("seeds", "lr_values", "batch_size_values", "max_epochs")
SELECTION_KINDS = ("clean", *NOISE_TYPES)
# Default memory protocol (Ramsauer et al. 2020): centered prototypes, 8 per class,
# beta chosen on clean + noisy validation sets whose levels are not in the test grids.
DEFAULT_PROTOTYPES_PER_CLASS = 8
DEFAULT_SELECTION_SETS = [["clean", 0], ["gaussian", 0.4], ["salt_pepper", 0.15]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    """Return ``value`` if it is a genuine integer >= minimum, else raise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _require_float(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
    strict: bool = False,
) -> float:
    """Return ``value`` as float if finite and inside the allowed range, else raise."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (strict and value == minimum)
        or (maximum is not None and value > maximum)
    ):
        bound = f"> {minimum}" if strict else f">= {minimum}"
        if maximum is not None:
            bound += f" and <= {maximum}"
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}.")
    return float(value)


def _require_ascending(name: str, values: list[Any]) -> None:
    """Reject grids that are not strictly increasing (which also rejects duplicates)."""
    for prev, cur in itertools.pairwise(values):
        if cur <= prev:
            raise ValueError(
                f"{name} must be strictly increasing without duplicates, got {values!r}."
            )


def _require_noise_grid(
    name: str,
    values: Any,
    *,
    integer: bool,
    maximum: float,
) -> list[Any]:
    """Validate a noise grid: non-empty, starts at the clean level 0, ascending."""
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{name} must be a list with at least two levels, got {values!r}.")
    if integer:
        resolved: list[Any] = [_require_int(name, v, minimum=0) for v in values]
        if resolved[-1] > maximum:
            raise ValueError(f"{name} values must be <= {maximum}, got {values!r}.")
    else:
        resolved = [_require_float(name, v, minimum=0.0, maximum=maximum) for v in values]
    if resolved[0] != 0:
        raise ValueError(f"{name} must start with the clean level 0, got {values!r}.")
    _require_ascending(name, resolved)
    return resolved


def _require_seeds(seeds: Any) -> list[int]:
    """Validate a non-empty list of unique non-negative integer seeds."""
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("seeds must be a non-empty list of non-negative integers.")
    resolved: list[int] = []
    for s in seeds:
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ValueError(f"All seeds must be non-negative integers, got {s!r}.")
        if s in resolved:
            raise ValueError(
                "seeds must contain unique values; "
                f"duplicate seed {s!r} would repeat the same experiment."
            )
        resolved.append(s)
    return resolved


def _validate_cost_config(cost: Any) -> dict[str, Any]:
    """Validate the nested ``cost`` section used by E4 latency, E7 and E8."""
    if not isinstance(cost, dict):
        raise TypeError(f"cost must be a mapping, got {type(cost).__name__}.")
    unknown = sorted(set(cost) - set(COST_KEYS))
    if unknown:
        raise ValueError(f"Unknown cost keys: {unknown}.")

    batch_sizes = cost.get("batch_sizes", [1, 32, 256, 1024])
    if not isinstance(batch_sizes, list) or len(batch_sizes) == 0:
        raise ValueError("cost.batch_sizes must be a non-empty list of positive integers.")
    batch_sizes = [_require_int("cost.batch_sizes", b, minimum=1) for b in batch_sizes]
    _require_ascending("cost.batch_sizes", batch_sizes)

    num_threads = cost.get("num_threads", None)
    if num_threads is not None:
        num_threads = _require_int("cost.num_threads", num_threads, minimum=1)

    return {
        "batch_size": _require_int("cost.batch_size", cost.get("batch_size", 256), minimum=1),
        "batch_sizes": batch_sizes,
        "warmup": _require_int("cost.warmup", cost.get("warmup", 10), minimum=0),
        "iters": _require_int("cost.iters", cost.get("iters", 100), minimum=1),
        "repeats": _require_int("cost.repeats", cost.get("repeats", 5), minimum=1),
        "num_threads": num_threads,
    }


def _validate_selection_sets(sets: Any) -> list[tuple[str, float]]:
    """Validate the validation sets whose mean head_acc selects beta.

    Each entry is ``[kind, level]`` with kind in ``SELECTION_KINDS``;
    ``[["clean", 0]]`` selects on clean validation only.
    """
    if not isinstance(sets, list) or len(sets) == 0:
        raise ValueError("beta_selection_sets must be a non-empty list of [kind, level] pairs.")
    resolved: list[tuple[str, float]] = []
    for entry in sets:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"beta_selection_sets entries must be [kind, level], got {entry!r}.")
        kind, level = entry
        if kind not in SELECTION_KINDS:
            raise ValueError(f"beta_selection_sets kind must be one of {SELECTION_KINDS}, got {kind!r}.")
        maximum = 1.0 if kind == "salt_pepper" else float(IMAGE_SIZE) if kind == "occlusion" else math.inf
        level = _require_float("beta_selection_sets level", level, minimum=0.0, maximum=maximum)
        if (kind == "clean") != (level == 0):
            raise ValueError(f"Only 'clean' may (and must) use level 0, got {entry!r}.")
        if (kind, level) in resolved:
            raise ValueError(f"Duplicate beta_selection_sets entry {entry!r}.")
        resolved.append((kind, level))
    return resolved


def _validate_tuning_config(tuning: Any) -> dict[str, Any] | None:
    """Validate the baseline grid search (``null`` disables it).

    The grid is lr x batch_size; every epoch up to ``max_epochs`` is a
    candidate, scored from one training run per (lr, batch_size).
    """
    if tuning is None:
        return None
    if not isinstance(tuning, dict):
        raise TypeError(f"baseline_tuning must be a mapping or null, got {type(tuning).__name__}.")
    unknown = sorted(set(tuning) - set(TUNING_KEYS))
    if unknown:
        raise ValueError(f"Unknown baseline_tuning keys: {unknown}.")

    lr_values = tuning.get("lr_values", [5e-4, 1e-3, 2e-3])
    if not isinstance(lr_values, list) or len(lr_values) == 0:
        raise ValueError("baseline_tuning.lr_values must be a non-empty list of positive floats.")
    lr_values = [
        _require_float("baseline_tuning.lr_values", v, minimum=0.0, strict=True) for v in lr_values
    ]
    _require_ascending("baseline_tuning.lr_values", lr_values)

    batch_sizes = tuning.get("batch_size_values", [64, 128])
    if not isinstance(batch_sizes, list) or len(batch_sizes) == 0:
        raise ValueError("baseline_tuning.batch_size_values must be a non-empty list of positive integers.")
    batch_sizes = [_require_int("baseline_tuning.batch_size_values", b, minimum=1) for b in batch_sizes]
    _require_ascending("baseline_tuning.batch_size_values", batch_sizes)

    return {
        "seeds": _require_seeds(tuning.get("seeds", [0])),
        "lr_values": lr_values,
        "batch_size_values": batch_sizes,
        "max_epochs": _require_int("baseline_tuning.max_epochs", tuning.get("max_epochs", 6), minimum=1),
    }


def apply_quick_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Merge the ``quick`` section over the top-level config (``cost`` merges per key)."""
    quick = config.get("quick") or {}
    if not isinstance(quick, dict):
        raise TypeError(f"quick must be a mapping, got {type(quick).__name__}.")
    unknown = sorted(set(quick) - set(CONFIG_KEYS) - {"quick"})
    if unknown or "quick" in quick:
        raise ValueError(f"Unknown quick override keys: {unknown or ['quick']}.")

    merged = {k: v for k, v in config.items() if k != "quick"}
    for key, value in quick.items():
        if key == "cost" and isinstance(value, dict):
            merged["cost"] = {**(merged.get("cost") or {}), **value}
        else:
            merged[key] = value
    return merged


def validate_mnist_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the Task 13 MNIST configuration.

    Unknown keys are rejected. Every noise grid must start at 0 (the clean
    level) and increase strictly, so accuracy drops and robust AUC always
    include the clean point. ``beta_values`` must increase strictly so the
    entropy monotonicity check (S6) is well defined.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration must be a dictionary/mapping, got {type(config).__name__}."
        )
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}.")

    seeds = _require_seeds(config.get("seeds", [0, 1, 2]))

    for key in ("data_dir", "reports_dir"):
        value = config.get(key, "data" if key == "data_dir" else DEFAULT_REPORTS_DIR)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string, got {value!r}.")

    device = config.get("device", "auto")
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}.")

    subsets: dict[str, int | None] = {}
    for key in ("train_subset", "test_subset"):
        value = config.get(key, None)
        subsets[key] = None if value is None else _require_int(key, value, minimum=10)

    beta_values = validate_beta_values(config.get("beta_values", [0.5, 1, 2, 5, 10, 20, 50, 100]))
    _require_ascending("beta_values", beta_values)

    num_updates_values = config.get("num_updates_values", [1, 2, 3, 5])
    if not isinstance(num_updates_values, list) or len(num_updates_values) == 0:
        raise ValueError("num_updates_values must be a non-empty list of positive integers.")
    num_updates_values = [
        _require_int("num_updates_values", t, minimum=1) for t in num_updates_values
    ]
    _require_ascending("num_updates_values", num_updates_values)

    placements = config.get("placements", list(PLACEMENTS))
    if (
        not isinstance(placements, list)
        or len(placements) == 0
        or any(p not in PLACEMENTS for p in placements)
        or len(set(placements)) != len(placements)
    ):
        raise ValueError(
            f"placements must be a non-empty list of unique values from {PLACEMENTS}, "
            f"got {placements!r}."
        )

    lr = _require_float("lr", config.get("lr", 1e-3), minimum=0.0, strict=True)
    center_prototypes = config.get("center_prototypes", True)
    if not isinstance(center_prototypes, bool):
        raise TypeError(
            f"center_prototypes must be a genuine boolean (True/False), got {center_prototypes!r}."
        )
    noisy_sigma = _require_float(
        "noisy_sigma", config.get("noisy_sigma", 0.5), minimum=0.0, strict=True
    )

    return {
        "seeds": seeds,
        "data_dir": config.get("data_dir", "data"),
        "reports_dir": config.get("reports_dir", DEFAULT_REPORTS_DIR),
        "device": device,
        "val_size": _require_int("val_size", config.get("val_size", 5000), minimum=10),
        "train_subset": subsets["train_subset"],
        "test_subset": subsets["test_subset"],
        "epochs": _require_int("epochs", config.get("epochs", 3), minimum=1),
        "batch_size": _require_int("batch_size", config.get("batch_size", 128), minimum=1),
        "lr": lr,
        "eval_batch_size": _require_int(
            "eval_batch_size", config.get("eval_batch_size", 1000), minimum=1
        ),
        "beta_values": beta_values,
        "s1_beta": validate_beta(config.get("s1_beta", 1e4)),
        "num_updates_values": num_updates_values,
        "gaussian_sigmas": _require_noise_grid(
            "gaussian_sigmas",
            config.get("gaussian_sigmas", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
            integer=False,
            maximum=math.inf,
        ),
        "salt_pepper_probs": _require_noise_grid(
            "salt_pepper_probs",
            config.get("salt_pepper_probs", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
            integer=False,
            maximum=1.0,
        ),
        "occlusion_sizes": _require_noise_grid(
            "occlusion_sizes",
            config.get("occlusion_sizes", [0, 7, 10, 14]),
            integer=True,
            maximum=IMAGE_SIZE,
        ),
        "noisy_sigma": noisy_sigma,
        "placements": placements,
        "center_prototypes": center_prototypes,
        "prototypes_per_class": _require_int(
            "prototypes_per_class", config.get("prototypes_per_class", DEFAULT_PROTOTYPES_PER_CLASS), minimum=1
        ),
        "beta_selection_sets": _validate_selection_sets(
            config.get("beta_selection_sets", DEFAULT_SELECTION_SETS)
        ),
        "baseline_tuning": _validate_tuning_config(config.get("baseline_tuning", None)),
        "cost": _validate_cost_config(config.get("cost", {})),
    }


def load_config(
    path: Path,
    *,
    quick: bool = False,
    seed: int | None = None,
    data_dir: str | None = None,
    reports_dir: str | None = None,
) -> dict[str, Any]:
    """Read YAML, apply ``--quick`` then CLI overrides, and validate."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(
            f"YAML configuration at {path} must be a mapping, got {type(raw).__name__}."
        )
    config: dict[str, Any] = apply_quick_overrides(raw) if quick else {
        k: v for k, v in raw.items() if k != "quick"
    }
    if seed is not None:
        config["seeds"] = [seed]
    if data_dir is not None:
        config["data_dir"] = data_dir
    if reports_dir is not None:
        config["reports_dir"] = reports_dir
    cfg = validate_mnist_config(config)
    cfg["quick"] = quick
    return cfg


def resolve_device(device: str) -> torch.device:
    """Map the configured device name to a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is not available.")
    return torch.device(device)


def configure_threads(num_threads: int | None) -> int:
    """Fix the intra-op thread count (if configured) and return the value in use."""
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def environment_info(device: torch.device) -> dict[str, Any]:
    """Collect device, CPU/GPU model, thread count and library versions."""
    cpu_model = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu_model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip() or cpu_model
        except (OSError, subprocess.SubprocessError):
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_model": cpu_model,
        "num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mnist_tensors(
    data_dir: str | Path,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one MNIST split as tensors without a per-item Python loop.

    Returns:
        images: float tensor [N, 1, 28, 28] in [0, 1] (not mean/std normalized;
            normalization happens inside ``MnistEncoder``).
        labels: long tensor [N].
    """
    import torchvision

    dataset = torchvision.datasets.MNIST(root=str(data_dir), train=train, download=True)
    images = dataset.data.float().div(255.0).unsqueeze(1)
    labels = dataset.targets.long()
    return images, labels


@dataclass
class MnistSplits:
    """Train/validation/test tensors for one seed. Images are [N, 1, 28, 28]."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def load_mnist(data_dir: str | Path) -> tuple[torch.Tensor, ...]:
    """Load (train_images, train_labels, test_images, test_labels)."""
    train_images, train_labels = load_mnist_tensors(data_dir, train=True)
    test_images, test_labels = load_mnist_tensors(data_dir, train=False)
    return train_images, train_labels, test_images, test_labels


def make_splits(
    data: Sequence[torch.Tensor],
    *,
    seed: int,
    val_size: int,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> MnistSplits:
    """Split the official train set into train/validation with a seeded permutation.

    The validation split (``val_size`` images) is only used to select beta.
    ``train_subset`` caps the remaining training images; ``test_subset`` keeps
    the first images of the official test set (identical across seeds).
    """
    train_images, train_labels, test_images, test_labels = data
    num_train = train_images.shape[0]
    if val_size >= num_train:
        raise ValueError(f"val_size ({val_size}) must be smaller than the train set ({num_train}).")

    perm = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    num_test = test_images.shape[0] if test_subset is None else min(test_subset, test_images.shape[0])
    return MnistSplits(
        train_images=train_images[train_idx],
        train_labels=train_labels[train_idx],
        val_images=train_images[val_idx],
        val_labels=train_labels[val_idx],
        test_images=test_images[:num_test],
        test_labels=test_labels[:num_test],
    )


# ---------------------------------------------------------------------------
# Corruptions (pixel space [0, 1], then clamped)
# ---------------------------------------------------------------------------


def corruption_seed(seed: int, kind: str, level: float, *, stream: int = 0) -> int:
    """Deterministic generator seed for one (seed, noise type, level) set.

    ``stream`` separates independent noise draws (0 = test sets, 1 = validation
    sets used for beta selection), so selection never reuses test noise.
    """
    if kind not in NOISE_TYPES:
        raise ValueError(f"Unknown corruption {kind!r}; expected one of {NOISE_TYPES}.")
    return (
        stream * 1_000_000_000_007
        + seed * 1_000_003
        + NOISE_TYPES.index(kind) * 10_007
        + round(float(level) * 1000)
    )


def corrupt_images(
    images: torch.Tensor,
    kind: str,
    level: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply one corruption to images [N, 1, H, W] in [0, 1]; returns a new tensor.

    gaussian:    x + N(0, level^2), clamped to [0, 1]
    salt_pepper: each pixel independently replaced with probability ``level``,
                 by 1 (salt) or 0 (pepper) with equal probability
    occlusion:   a ``level`` x ``level`` square of zeros at a uniformly random
                 position fully inside the image

    Level 0 returns an unchanged copy.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [N, C, H, W], got {tuple(images.shape)}.")
    if level == 0:
        return images.clone()

    n, _, height, width = images.shape
    if kind == "gaussian":
        noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
        return (images + noise * level).clamp(0.0, 1.0)
    if kind == "salt_pepper":
        hit = torch.rand(images.shape, generator=generator) < level
        salt = torch.rand(images.shape, generator=generator) < 0.5
        out = images.clone()
        out[hit & salt] = 1.0
        out[hit & ~salt] = 0.0
        return out
    if kind == "occlusion":
        side = int(level)
        if side > min(height, width):
            raise ValueError(f"occlusion side {side} exceeds image size {height}x{width}.")
        top = torch.randint(0, height - side + 1, (n,), generator=generator)
        left = torch.randint(0, width - side + 1, (n,), generator=generator)
        rows = torch.arange(height)
        cols = torch.arange(width)
        row_mask = (rows >= top[:, None]) & (rows < (top + side)[:, None])  # [N, H]
        col_mask = (cols >= left[:, None]) & (cols < (left + side)[:, None])  # [N, W]
        square = row_mask[:, :, None] & col_mask[:, None, :]  # [N, H, W]
        return images.masked_fill(square[:, None], 0.0).clamp(0.0, 1.0)
    raise ValueError(f"Unknown corruption {kind!r}; expected one of {NOISE_TYPES}.")


def make_corrupted_test_set(
    images: torch.Tensor,
    *,
    seed: int,
    kind: str,
    level: float,
    stream: int = 0,
) -> torch.Tensor:
    """Corrupt a test set with a generator seeded by (seed, kind, level, stream)."""
    generator = torch.Generator().manual_seed(corruption_seed(seed, kind, level, stream=stream))
    return corrupt_images(images, kind, level, generator)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MnistEncoder(nn.Module):
    """Layers L1-L7: image [B, 1, 28, 28] in [0, 1] -> embedding z [B, embed_dim].

    Inputs are normalized with the MNIST mean/std inside the forward pass, so
    every caller (and every corruption) works in raw [0, 1] pixel space. The
    widths default to the Task 13 architecture; tests shrink them.
    """

    def __init__(
        self,
        c1: int = 32,
        c2: int = 64,
        hidden: int = 128,
        embed_dim: int = 64,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {image_size}.")
        self.c1, self.c2, self.hidden, self.embed_dim = c1, c2, hidden, embed_dim
        self.image_size = image_size
        flat = c2 * (image_size // 4) ** 2
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)  # L1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # L2
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)  # L3
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # L4
        self.flatten = nn.Flatten()  # L5
        self.fc1 = nn.Linear(flat, hidden)  # L6
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, embed_dim)  # L7, no activation: signed z

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - MNIST_MEAN) / MNIST_STD
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.fc1(self.flatten(x)))
        return self.fc2(x)


class ClassifierHead(nn.Module):
    """Layer L9: h [B, embed_dim] -> logits [B, num_classes]."""

    def __init__(self, embed_dim: int = 64, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class MnistNet(nn.Module):
    """encoder (L1-L7) -> bridge (L8) -> head (L9).

    The bridge maps z [B, D] to either h [B, D] (e.g. ``nn.Identity``) or a
    tuple ``(h, aux)`` where aux is a diagnostics dict. ``forward`` always
    returns ``(logits [B, C], z [B, D], h [B, D], aux)``; aux is ``{}`` for a
    bridge that returns a bare tensor.
    """

    def __init__(self, encoder: MnistEncoder, bridge: nn.Module, head: ClassifierHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.head = head

    def forward_from_embedding(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run L8 and L9 on precomputed embeddings z [B, D]."""
        out = self.bridge(z)
        h, aux = out if isinstance(out, tuple) else (out, {})
        return self.head(h), z, h, aux

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.forward_from_embedding(self.encoder(images))


def build_baseline_net(
    seed: int,
    *,
    c1: int = 32,
    c2: int = 64,
    hidden: int = 128,
    embed_dim: int = 64,
) -> MnistNet:
    """Build the baseline network (L8 = Identity) with seeded initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = MnistEncoder(c1=c1, c2=c2, hidden=hidden, embed_dim=embed_dim)
        head = ClassifierHead(embed_dim=embed_dim)
    return MnistNet(encoder, nn.Identity(), head)


def train_baseline(
    net: MnistNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    on_epoch_end: Callable[[int], None] | None = None,
) -> float:
    """Train the baseline on clean images [N, 1, 28, 28] with Adam; return wall seconds.

    ``on_epoch_end(epoch)`` (1-based) runs after each epoch with the net in
    eval mode; its time is included in the returned wall time.
    """
    net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(images.shape[0], generator=generator)
        total_loss, total = 0.0, 0
        for i in range(0, images.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            x, y = images[idx].to(device), labels[idx].to(device)
            logits, _, _, _ = net(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
            total += idx.numel()
        print(f"    epoch {epoch + 1}/{epochs}  train loss {total_loss / max(total, 1):.4f}")
        if on_epoch_end is not None:
            net.eval()
            on_epoch_end(epoch + 1)
            net.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    net.eval()
    return time.perf_counter() - start


def checkpoint_fingerprint(cfg: dict[str, Any], seed: int, num_train: int) -> dict[str, Any]:
    """Settings a cached backbone must match to be reused."""
    return {
        "seed": seed,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "val_size": cfg["val_size"],
        "num_train": num_train,
        "architecture": [32, 64, 128, 64],
    }


def get_or_train_baseline(
    cfg: dict[str, Any],
    splits: MnistSplits,
    seed: int,
    device: torch.device,
    *,
    retrain: bool = False,
) -> tuple[MnistNet, float, bool]:
    """Load ``backbone_seed{seed}.pt`` if it matches the config, else train and cache it.

    Returns ``(net, train_time_s, loaded_from_cache)``. A cached checkpoint
    reports the wall time of the run that produced it. The returned net is in
    eval mode and is never optimized again ("frozen").
    """
    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt"
    fingerprint = checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0])
    net = build_baseline_net(seed)

    if path.exists() and not retrain:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") == fingerprint:
            net.load_state_dict(payload["state_dict"])
            print(f"  seed {seed}: loaded cached backbone {path}")
            return net.to(device).eval(), float(payload["train_time_s"]), True
        print(f"  seed {seed}: cached backbone {path} has different settings; retraining")

    print(f"  seed {seed}: training baseline on {splits.train_images.shape[0]} clean images")
    train_time_s = train_baseline(
        net,
        splits.train_images,
        splits.train_labels,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        seed=seed,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "fingerprint": fingerprint,
            "train_time_s": train_time_s,
        },
        path,
    )
    return net.eval(), train_time_s, False


# ---------------------------------------------------------------------------
# Data-split table and baseline hyperparameter tuning
# ---------------------------------------------------------------------------


def split_table(cfg: dict[str, Any], data: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Sizes and class balance of the train / validation / test splits.

    Split sizes do not depend on the seed (only which images land in train or
    validation does), so the first configured seed is representative.
    """
    splits = make_splits(
        data,
        seed=cfg["seeds"][0],
        val_size=cfg["val_size"],
        train_subset=cfg["train_subset"],
        test_subset=cfg["test_subset"],
    )
    rows = []
    for name, labels, source, use in (
        ("train", splits.train_labels, "official train, seeded split",
         "fit CNN weights; build prototype memory M"),
        ("validation", splits.val_labels, "official train, seeded split",
         "tune baseline (lr, batch, epochs) and β"),
        ("test", splits.test_labels, "official test",
         "final paired Baseline vs Hopfield comparison"),
    ):
        counts = torch.bincount(labels, minlength=NUM_CLASSES)
        rows.append({
            "split": name,
            "images": int(labels.shape[0]),
            "per_class_min": int(counts.min()),
            "per_class_max": int(counts.max()),
            "source": source,
            "used_for": use,
        })
    return {"rows": rows, "seed": cfg["seeds"][0]}


def format_split_table(table: dict[str, Any]) -> list[str]:
    """Render ``split_table`` as fixed-width lines."""
    total = sum(r["images"] for r in table["rows"])
    lines = [f"  {'split':<11}| {'images':>7} | {'share':>6} | {'per class':>11} | {'source':<29}| used for"]
    for r in table["rows"]:
        lines.append(
            f"  {r['split']:<11}| {r['images']:>7} | {r['images'] / total:>6.1%} | "
            f"{r['per_class_min']:>5}–{r['per_class_max']:<5} | {r['source']:<29}| {r['used_for']}"
        )
    return lines


def _tuning_fingerprint(cfg: dict[str, Any], num_train: int) -> dict[str, Any]:
    """Settings a cached tuning result must match to be reused."""
    return {
        "tuning": cfg["baseline_tuning"],
        "selection_sets": [list(s) for s in cfg["beta_selection_sets"]],
        "val_size": cfg["val_size"],
        "num_train": num_train,
        "architecture": [32, 64, 128, 64],
    }


def tune_baseline(
    cfg: dict[str, Any],
    data: Sequence[torch.Tensor],
    device: torch.device,
    *,
    retrain: bool = False,
) -> dict[str, Any] | None:
    """Grid-search the baseline's lr, batch size and epochs on validation data.

    Each (lr, batch_size) is trained once for ``max_epochs`` epochs and scored
    after every epoch, so every epoch count is a candidate. The score is the
    same one that selects beta for the Hopfield variant: mean head_acc over
    ``beta_selection_sets`` (clean + noisy validation, noise stream 1), so
    both variants are tuned on the same criterion and the test set is never
    used. Ties -> fewest epochs, then smallest lr, then smallest batch size.

    The winning values are written into ``cfg`` (lr, batch_size, epochs) and
    the result is cached as ``baseline_tuning.json``. Returns None when
    tuning is disabled.
    """
    tuning = cfg["baseline_tuning"]
    if tuning is None:
        return None
    sets = cfg["beta_selection_sets"]
    path = Path(cfg["reports_dir"]) / "baseline_tuning.json"
    first = make_splits(data, seed=tuning["seeds"][0], val_size=cfg["val_size"],
                        train_subset=cfg["train_subset"])
    fingerprint = _tuning_fingerprint(cfg, first.train_images.shape[0])

    result: dict[str, Any] | None = None
    if path.exists() and not retrain:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("fingerprint") == json.loads(json.dumps(fingerprint)):
            print(f"Baseline tuning: loaded cached result {path}")
            result = cached

    if result is None:
        scores: dict[tuple[float, int, int], list[dict[str, float]]] = {}
        for seed in tuning["seeds"]:
            splits = make_splits(data, seed=seed, val_size=cfg["val_size"],
                                 train_subset=cfg["train_subset"])
            # Clean validation is always evaluated first (reported as val_clean);
            # the score averages only the configured selection sets.
            val_sets = [splits.val_images] + [
                make_corrupted_test_set(
                    splits.val_images, seed=seed, kind=kind, level=level, stream=1
                )
                for kind, level in sets if kind != "clean"
            ]
            scored = val_sets if any(kind == "clean" for kind, _ in sets) else val_sets[1:]
            for lr, bs in itertools.product(tuning["lr_values"], tuning["batch_size_values"]):
                print(f"  tuning seed {seed}: lr={lr:g} batch={bs}")
                net = build_baseline_net(seed)

                def score(epoch: int, *, net: MnistNet = net, lr: float = lr, bs: int = bs,
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
        result = {
            "fingerprint": fingerprint,
            "selection_sets": [list(s) for s in sets],
            "seeds": tuning["seeds"],
            "grid": grid,
            "best": {k: best[k] for k in ("lr", "batch_size", "epochs")},
            "best_scores": {k: best[k] for k in ("val_clean", "val_score")},
            "untuned": untuned,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    cfg.update(result["best"])
    return result


def format_tuning_table(result: dict[str, Any], *, top: int = 8) -> list[str]:
    """Best ``top`` grid points by validation score, plus the untuned default."""
    seeds = ",".join(str(s) for s in result["seeds"])
    lines = [(f"  {'rank':>4} | {'lr':>7} | {'batch':>5} | {'epochs':>6} | "
              f"{'val clean':>9} | {'val score':>9}")]
    ranked = sorted(result["grid"], key=lambda r: (-r["val_score"], r["epochs"], r["lr"], r["batch_size"]))
    u = result.get("untuned")
    shown = ranked[:top] + ([u] if u is not None and u not in ranked[:top] else [])
    for r in shown:
        mark = "   ← previous untuned default" if r == u else ""
        lines.append(f"  {ranked.index(r) + 1:>4} | {r['lr']:>7g} | {r['batch_size']:>5} | "
                     f"{r['epochs']:>6} | {r['val_clean']:>9.4f} | {r['val_score']:>9.4f}{mark}")
    b = result["best"]
    lines.append(f"  Selected: lr={b['lr']:g}, batch={b['batch_size']}, epochs={b['epochs']} "
                 f"({len(ranked)} grid points, tuning seeds {seeds}; val score = mean val head_acc "
                 "over the β-selection sets)")
    return lines


# ---------------------------------------------------------------------------
# Embeddings, prototypes and readouts
# ---------------------------------------------------------------------------


@torch.inference_mode()
def encode(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Embed images [N, 1, 28, 28] -> z [N, D] (on CPU), in eval mode."""
    encoder.eval()
    chunks = [
        encoder(images[i : i + batch_size].to(device)).cpu()
        for i in range(0, images.shape[0], batch_size)
    ]
    return torch.cat(chunks)


def compute_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class means: features [N, D], labels [N] -> (prototypes [K, D], memory_labels [K]).

    Row c of the prototype matrix is the mean of all features with label c.
    """
    counts = torch.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"Cannot build prototypes: classes {missing} have no samples.")
    sums = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)
    sums.index_add_(0, labels, features)
    prototypes = sums / counts[:, None].to(features.dtype)
    return prototypes, torch.arange(num_classes)


def nearest_prototype(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Non-Hopfield prototype retrieval: index of the max-cosine prototype, z [N, D] -> [N]."""
    return (F.normalize(z, dim=-1) @ F.normalize(prototypes, dim=-1).T).argmax(dim=-1)


def _collect(
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> dict[str, torch.Tensor]:
    """Concatenate per-batch network outputs into CPU tensors."""
    out: dict[str, torch.Tensor] = {
        "head_pred": torch.cat([b[0].argmax(dim=-1).cpu() for b in batches]),
        "z": torch.cat([b[1].cpu() for b in batches]),
        "h": torch.cat([b[2].cpu() for b in batches]),
    }
    if "weights" in batches[0][3]:
        out["weights"] = torch.cat([b[3]["weights"].cpu() for b in batches])
        out["proto_pred"] = torch.cat([b[3]["proto_pred"].cpu() for b in batches])
        if "class_mass" in batches[0][3]:
            out["class_mass"] = torch.cat([b[3]["class_mass"].cpu() for b in batches])
    return out


@torch.inference_mode()
def forward_embeddings(
    net: MnistNet,
    z: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run L8+L9 on precomputed z [N, D].

    Returns head_pred [N], z [N, D], h [N, D] and, for bridges that expose
    retrieval weights, weights [N, K] and proto_pred [N].
    """
    net.eval()
    batches = [
        net.forward_from_embedding(z[i : i + batch_size].to(device))
        for i in range(0, z.shape[0], batch_size)
    ]
    return _collect(batches)


@torch.inference_mode()
def forward_images(
    net: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the full network on images [N, 1, 28, 28]; same keys as ``forward_embeddings``."""
    net.eval()
    batches = [
        net(images[i : i + batch_size].to(device)) for i in range(0, images.shape[0], batch_size)
    ]
    return _collect(batches)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def retrieval_confusion(
    labels: torch.Tensor,
    retrieved: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Counts [K, K]: row = true class, column = class of the retrieved prototype."""
    flat = labels * num_classes + retrieved
    return torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)


def summarize_outputs(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    memory_labels: torch.Tensor,
) -> tuple[dict[str, float | None], dict[str, torch.Tensor], torch.Tensor]:
    """Compute the Task 13 metrics for one model on one evaluation set.

    Args:
        outputs: from ``forward_embeddings``/``forward_images``.
        labels: [N] ground truth.
        prototypes: [K, D] embedding-space prototypes M.
        memory_labels: [K] class of each prototype.

    Returns:
        metrics: head_acc, proto_acc, dist_to_true_proto, and (only when
            retrieval weights exist, else None) entropy_norm, top1_weight,
            true_proto_weight.
        correct: per-sample bool vectors ``head`` and ``proto`` [N] (for McNemar).
        confusion: retrieval confusion counts [K, K].

    Without weights (baseline), prototype retrieval is the nearest prototype
    to z by cosine; with weights it is the bridge's ``proto_pred``. When the
    bridge reports ``class_mass`` [N, C] (several memories per class),
    true_proto_weight is the total weight on the true class.
    """
    weights = outputs.get("weights")
    if weights is None:
        proto_pred = memory_labels[nearest_prototype(outputs["z"], prototypes)]
    else:
        proto_pred = outputs["proto_pred"]

    head_correct = outputs["head_pred"] == labels
    proto_correct = proto_pred == labels
    dist = retrieval_distance(outputs["h"], prototypes[labels])

    metrics: dict[str, float | None] = {
        "head_acc": head_correct.float().mean().item(),
        "proto_acc": proto_correct.float().mean().item(),
        "dist_to_true_proto": dist.mean().item(),
        "entropy_norm": None,
        "top1_weight": None,
        "true_proto_weight": None,
    }
    if weights is not None:
        metrics["entropy_norm"] = attention_entropy(weights, normalize=True).mean().item()
        metrics["top1_weight"] = topk_mass(weights, k=1).mean().item()
        if "class_mass" in outputs:
            true_weight = outputs["class_mass"].gather(1, labels[:, None]).squeeze(1)
        else:
            true_mask = (memory_labels[None, :] == labels[:, None]).to(weights.dtype)
            true_weight = (weights * true_mask).sum(dim=-1)
        metrics["true_proto_weight"] = true_weight.mean().item()

    confusion = retrieval_confusion(labels, proto_pred, num_classes=prototypes.shape[0])
    return metrics, {"head": head_correct, "proto": proto_correct}, confusion


def robust_auc(levels: Sequence[float], accuracies: Sequence[float]) -> float:
    """Trapezoid mean of accuracy over noise levels (clean level included).

    AUC = integral(acc d level) / (max level - min level), so a model that is
    perfect at every level scores 1.0 regardless of the grid spacing.
    """
    if len(levels) != len(accuracies) or len(levels) < 2:
        raise ValueError("robust_auc needs matching level/accuracy lists of length >= 2.")
    area = sum(
        (levels[i + 1] - levels[i]) * (accuracies[i + 1] + accuracies[i]) / 2.0
        for i in range(len(levels) - 1)
    )
    return area / (levels[-1] - levels[0])


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample std across seeds (std is 0.0 for a single seed)."""
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Cost / efficiency harness (shared by both variants)
# ---------------------------------------------------------------------------


def count_trainable_params(model: nn.Module) -> int:
    """Number of parameter values with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_buffer_values(model: nn.Module, *, floating_only: bool = False) -> int:
    """Number of buffer values (optionally only floating-point buffers)."""
    return sum(
        b.numel() for b in model.buffers() if not floating_only or b.is_floating_point()
    )


def model_size_bytes(model: nn.Module) -> int:
    """Serialized size of ``state_dict`` written by torch.save into memory."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def analytical_macs(net: MnistNet) -> list[tuple[str, int]]:
    """Per-layer multiply-accumulates for one image, from the architecture alone.

    conv: H_out * W_out * C_out * k^2 * C_in;  linear: in * out.
    A bridge that defines ``macs_per_image()`` contributes that term at L8.
    Pooling, ReLU and input normalization are not counted.
    """
    enc = net.encoder
    s = enc.image_size
    k1 = enc.conv1.kernel_size[0] * enc.conv1.kernel_size[1]
    k2 = enc.conv2.kernel_size[0] * enc.conv2.kernel_size[1]
    breakdown = [
        ("L1 conv1", s * s * enc.c1 * k1 * 1),
        ("L3 conv2", (s // 2) * (s // 2) * enc.c2 * k2 * enc.c1),
        ("L6 fc1", enc.c2 * (s // 4) ** 2 * enc.hidden),
        ("L7 fc2", enc.hidden * enc.embed_dim),
    ]
    bridge_macs = getattr(net.bridge, "macs_per_image", None)
    if callable(bridge_macs):
        breakdown.append(("L8 bridge", int(bridge_macs())))
    breakdown.append(("L9 head", net.head.fc.in_features * net.head.fc.out_features))
    return breakdown


@torch.inference_mode()
def hook_macs(net: MnistNet, *, batch_size: int = 2) -> int:
    """Per-image MACs counted from real output shapes via forward hooks.

    Hooks cover every Conv2d and Linear module; a bridge exposing
    ``macs_per_image()`` adds its known term (it has no Conv2d/Linear layers).
    """
    total = 0

    def conv_hook(module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        nonlocal total
        assert isinstance(module, nn.Conv2d)
        kh, kw = module.kernel_size
        per_image = out[0].numel() * kh * kw * module.in_channels // module.groups
        total += per_image

    def linear_hook(module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        nonlocal total
        assert isinstance(module, nn.Linear)
        rows_per_image = out.numel() // out.shape[-1] // out.shape[0]
        total += rows_per_image * module.in_features * module.out_features

    handles = []
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        device = next(net.parameters()).device
        size = net.encoder.image_size
        net.eval()
        net(torch.zeros(batch_size, 1, size, size, device=device))
    finally:
        for handle in handles:
            handle.remove()

    bridge_macs = getattr(net.bridge, "macs_per_image", None)
    if callable(bridge_macs):
        total += int(bridge_macs())
    return total


def _tensor_bytes(obj: Any) -> int:
    """Total bytes of every tensor inside nested tuples/lists/dicts."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


@torch.inference_mode()
def activation_peak_bytes(model: nn.Module, x: torch.Tensor) -> int:
    """Upper-bound inference memory: weights + input + every leaf-module output.

    Used on CPU, where neither CUDA allocator stats nor tracemalloc (which
    cannot see torch's native tensor storage) measure tensor memory. It
    assumes nothing is freed during the forward pass, so it over-estimates.
    """
    total = _tensor_bytes(list(model.parameters())) + _tensor_bytes(list(model.buffers()))
    total += _tensor_bytes(x)
    outputs: list[int] = []

    def hook(_module: nn.Module, _inp: Any, out: Any) -> None:
        outputs.append(_tensor_bytes(out))

    handles = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total + sum(outputs)


def _percentile(values: list[float], q: float) -> float:
    return torch.quantile(torch.tensor(values, dtype=torch.float64), q).item()


@torch.inference_mode()
def measure_cost(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    *,
    warmup: int = 10,
    iters: int = 100,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Any]:
    """Time ``model`` on random [batch, 1, H, W] inputs in eval/inference mode.

    ``warmup`` untimed batches are followed by ``iters`` timed batches, each
    timed with ``time.perf_counter`` (with ``cuda.synchronize`` on GPU).

    Returns latency_ms_p50/p95/mean, latency_per_image_us (p50 / batch),
    throughput_img_s (batch / mean latency), peak_mem_mb and peak_mem_source.
    """
    model.eval()
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(batch_size, 1, image_size, image_size, generator=generator).to(device)
    is_cuda = device.type == "cuda"

    for _ in range(warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        if is_cuda:
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    if is_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
        source = "cuda.max_memory_allocated"
    else:
        peak_mb = activation_peak_bytes(model, x) / 2**20
        source = "activation upper bound (CPU)"

    p50 = _percentile(times_ms, 0.5)
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "batch_size": batch_size,
        "latency_ms_p50": p50,
        "latency_ms_p95": _percentile(times_ms, 0.95),
        "latency_ms_mean": mean_ms,
        "latency_per_image_us": p50 * 1000.0 / batch_size,
        "throughput_img_s": batch_size / (mean_ms / 1000.0),
        "peak_mem_mb": peak_mb,
        "peak_mem_source": source,
    }


def median_iqr(values: Sequence[float]) -> dict[str, Any]:
    """Median and inter-quartile range of repeated measurements."""
    vals = list(values)
    return {
        "median": _percentile(vals, 0.5),
        "iqr": _percentile(vals, 0.75) - _percentile(vals, 0.25),
        "values": vals,
    }


def interleaved_cost(
    models: dict[str, nn.Module],
    batch_size: int,
    device: torch.device,
    *,
    repeats: int = 5,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, dict[str, Any]]:
    """Run ``measure_cost`` for each model, alternating the order every repeat.

    Repeat r uses the given order when r is even and the reversed order when
    odd, so slow drift (thermal, background load) hits every model equally.
    Returns, per model, median/IQR for each numeric metric plus the source of
    the peak-memory figure.
    """
    names = list(models)
    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for r in range(repeats):
        for name in names if r % 2 == 0 else names[::-1]:
            raw[name].append(
                measure_cost(models[name], batch_size, device, warmup=warmup, iters=iters)
            )
    summary: dict[str, dict[str, Any]] = {}
    for name, runs in raw.items():
        summary[name] = {
            key: median_iqr([run[key] for run in runs])
            for key in runs[0]
            if key not in ("batch_size", "peak_mem_source")
        }
        summary[name]["peak_mem_source"] = runs[0]["peak_mem_source"]
    return summary


def time_full_evaluation(
    model: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Wall seconds to run the full network over ``images`` (e.g. the 10k test set)."""
    start = time.perf_counter()
    forward_images(model, images, batch_size=batch_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Per-seed baseline pipeline (shared with the Hopfield comparison)
# ---------------------------------------------------------------------------


@dataclass
class BaselineRun:
    """Everything one seed's experiments need, built once from the baseline net."""

    seed: int
    splits: MnistSplits
    net: MnistNet
    train_time_s: float
    from_cache: bool
    prototypes: torch.Tensor  # [K, D] embedding prototypes M
    memory_labels: torch.Tensor  # [K]
    prototype_time_s: float
    z_train: torch.Tensor  # [N_train, D]
    z_val: torch.Tensor  # [N_val, D]
    batch_size: int
    device: torch.device
    _embedded: dict[tuple[str, float], torch.Tensor] = field(default_factory=dict)
    _val_embedded: dict[tuple[str, float], torch.Tensor] = field(default_factory=dict)

    def val_embeddings(self, kind: str, level: float) -> torch.Tensor:
        """z [N_val, D] of a validation set (clean, or corrupted with noise stream 1)."""
        if kind == "clean" or level == 0:
            return self.z_val
        key = (kind, float(level))
        if key not in self._val_embedded:
            images = make_corrupted_test_set(
                self.splits.val_images, seed=self.seed, kind=kind, level=level, stream=1
            )
            self._val_embedded[key] = encode(
                self.net.encoder, images, batch_size=self.batch_size, device=self.device
            )
        return self._val_embedded[key]

    def val_images(self, kind: str, level: float) -> torch.Tensor:
        """Validation images for pixel-space evaluation (same noise stream as above)."""
        if kind == "clean" or level == 0:
            return self.splits.val_images
        return make_corrupted_test_set(
            self.splits.val_images, seed=self.seed, kind=kind, level=level, stream=1
        )

    def test_embeddings(self, kind: str, level: float) -> torch.Tensor:
        """z [N_test, D] of the (kind, level) test set, corrupted and encoded once.

        Every level-0 request maps to the same clean set. Both variants share
        L1-L7, so this one embedding feeds both models (paired comparison).
        """
        key = ("clean", 0.0) if level == 0 else (kind, float(level))
        if key not in self._embedded:
            images = (
                self.splits.test_images
                if level == 0
                else make_corrupted_test_set(
                    self.splits.test_images, seed=self.seed, kind=kind, level=level
                )
            )
            self._embedded[key] = encode(
                self.net.encoder, images, batch_size=self.batch_size, device=self.device
            )
        return self._embedded[key]


def prepare_baseline_run(
    cfg: dict[str, Any],
    data: Sequence[torch.Tensor],
    seed: int,
    device: torch.device,
    *,
    retrain: bool = False,
) -> BaselineRun:
    """Split data, train (or load) the frozen baseline, and build prototypes M."""
    splits = make_splits(
        data,
        seed=seed,
        val_size=cfg["val_size"],
        train_subset=cfg["train_subset"],
        test_subset=cfg["test_subset"],
    )
    net, train_time_s, from_cache = get_or_train_baseline(
        cfg, splits, seed, device, retrain=retrain
    )
    start = time.perf_counter()
    z_train = encode(net.encoder, splits.train_images, batch_size=cfg["eval_batch_size"], device=device)
    prototypes, memory_labels = compute_prototypes(z_train, splits.train_labels)
    prototype_time_s = time.perf_counter() - start
    z_val = encode(net.encoder, splits.val_images, batch_size=cfg["eval_batch_size"], device=device)
    return BaselineRun(
        seed=seed,
        splits=splits,
        net=net,
        train_time_s=train_time_s,
        from_cache=from_cache,
        prototypes=prototypes,
        memory_labels=memory_labels,
        prototype_time_s=prototype_time_s,
        z_train=z_train,
        z_val=z_val,
        batch_size=cfg["eval_batch_size"],
        device=device,
    )


def evaluate_baseline(run: BaselineRun, kind: str, level: float) -> dict[str, Any]:
    """Baseline metrics on one (kind, level) test set of ``run``'s seed."""
    outputs = forward_embeddings(
        run.net, run.test_embeddings(kind, level), batch_size=run.batch_size, device=run.device
    )
    metrics, _, _ = summarize_outputs(
        outputs, run.splits.test_labels, run.prototypes, run.memory_labels
    )
    return metrics


def noise_grids(cfg: dict[str, Any]) -> dict[str, list[float]]:
    """Noise type -> configured levels (E3a, E3b, E3c)."""
    return {kind: cfg[NOISE_GRID_KEYS[kind]] for kind in NOISE_TYPES}


def copy_net(net: nn.Module) -> nn.Module:
    """Deep copy (used to give the Hopfield variant its own copy of shared weights)."""
    return copy.deepcopy(net)


# ---------------------------------------------------------------------------
# Baseline-only entry point
# ---------------------------------------------------------------------------


def run_baseline_only(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """E1 and E3 baseline columns (head_acc, proto_acc, dist_to_true_proto, AUC)."""
    threads = configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    print(f"Device: {device}  threads: {threads}")
    data = load_mnist(cfg["data_dir"])
    header = ["Data splits", *format_split_table(split_table(cfg, data)), ""]
    print("\n".join(header))
    tuning = tune_baseline(cfg, data, device, retrain=retrain)
    if tuning is not None:
        header += ["Baseline tuning", *format_tuning_table(tuning), ""]

    per_seed: dict[str, list[list[dict[str, Any]]]] = {"clean": []}
    grids = noise_grids(cfg)
    for kind in NOISE_TYPES:
        per_seed[kind] = []
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        per_seed["clean"].append([evaluate_baseline(run, "gaussian", 0.0)])
        for kind in NOISE_TYPES:
            per_seed[kind].append([evaluate_baseline(run, kind, lvl) for lvl in grids[kind]])

    lines: list[str] = list(header)
    rows: list[dict[str, Any]] = []
    for kind in ("clean", *NOISE_TYPES):
        levels = [0.0] if kind == "clean" else grids[kind]
        title = "E1 clean" if kind == "clean" else f"E3 {kind}"
        lines.append(f"{title}   Varied: {'model (baseline only)' if kind == 'clean' else kind + ' level'}"
                     "   Held fixed: L8=Identity, clean training")
        lines.append(f"  {'level':>6} | {'head_acc':>15} | {'proto_acc':>15} | {'dist_to_true_proto':>18}")
        for i, level in enumerate(levels):
            stats = {
                m: mean_std([seed_rows[i][m] for seed_rows in per_seed[kind]])
                for m in ("head_acc", "proto_acc", "dist_to_true_proto")
            }
            lines.append(
                f"  {level:>6g} | {stats['head_acc'][0]:.4f} ± {stats['head_acc'][1]:.4f} | "
                f"{stats['proto_acc'][0]:.4f} ± {stats['proto_acc'][1]:.4f} | "
                f"{stats['dist_to_true_proto'][0]:>10.4f} ± {stats['dist_to_true_proto'][1]:.4f}"
            )
            rows.append({"experiment": title, "level": level, **{
                f"{m}_{s}": stats[m][j] for m in stats for j, s in enumerate(("mean", "std"))
            }})
        if kind != "clean":
            aucs = [
                robust_auc(levels, [r["head_acc"] for r in seed_rows])
                for seed_rows in per_seed[kind]
            ]
            auc_mean, auc_std = mean_std(aucs)
            lines.append(f"  {'AUC':>6} | {auc_mean:.4f} ± {auc_std:.4f}")
            rows.append({"experiment": title, "level": "AUC", "head_acc_mean": auc_mean,
                         "head_acc_std": auc_std})
        lines.append("")
    text = "\n".join(lines)
    print(text)

    out_dir = Path(cfg["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "environment": environment_info(device), "rows": rows}, f, indent=2)
    (out_dir / "baseline_summary.md").write_text(
        f"# Task 13 baseline only (L8 = Identity)\n\n```\n{text}\n```\n", encoding="utf-8"
    )
    print(f"Baseline reports saved to: {out_dir}")
    return {"rows": rows, "text": text}


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """CLI shared by both Task 13 entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to YAML configuration file.")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, 1 epoch, 2000 train / 1000 test images, reduced grids.")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore cached backbones and retrain.")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed.")
    parser.add_argument("--data-dir", type=str, default=None, help="MNIST download/cache dir.")
    parser.add_argument("--reports-dir", type=str, default=None, help="Output directory.")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Load and validate the config described by parsed CLI arguments."""
    return load_config(
        args.config,
        quick=args.quick,
        seed=args.seed,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Task 13 baseline (non-Hopfield) MNIST CNN.").parse_args(argv)
    cfg = config_from_args(args)
    run_baseline_only(cfg, retrain=args.retrain)


if __name__ == "__main__":
    main()

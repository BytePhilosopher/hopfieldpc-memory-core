"""Offline tests for the Task 13 MNIST benchmark (tiny random-init nets, synthetic tensors).

Nothing here downloads or asserts on real MNIST; the end-to-end test patches
the loader with a synthetic 10-class dataset.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import benchmarks.mnist_baseline as mb
from benchmarks.mnist_baseline import (
    ClassifierHead,
    MnistEncoder,
    MnistNet,
    analytical_macs,
    compute_prototypes,
    corrupt_images,
    count_trainable_params,
    hook_macs,
    make_corrupted_test_set,
    measure_cost,
    nearest_prototype,
    robust_auc,
    validate_mnist_config,
)
from benchmarks.mnist_prototype_retrieval import (
    HopfieldBridge,
    PixelHopfieldNet,
    build_hopfield_net,
    build_memory,
    decide_winner,
    kmeans,
    main,
    mcnemar_exact_p,
    mcnemar_test,
    shared_weights_equal,
    wilcoxon_signed_rank_p,
)
from hopfieldpc.diagnostics import attention_entropy


def tiny_baseline(seed: int = 0, embed_dim: int = 8) -> MnistNet:
    torch.manual_seed(seed)
    encoder = MnistEncoder(c1=4, c2=8, hidden=16, embed_dim=embed_dim)
    return MnistNet(encoder, nn.Identity(), ClassifierHead(embed_dim=embed_dim)).eval()


def tiny_prototypes(dim: int = 8, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(k, dim, generator=generator), torch.arange(k)


def images(n: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Shapes and bridges
# ---------------------------------------------------------------------------


def test_layer_shapes_full_architecture() -> None:
    encoder = MnistEncoder()
    x = (images(3) - mb.MNIST_MEAN) / mb.MNIST_STD
    expected = {
        "conv1": (3, 32, 28, 28), "pool1": (3, 32, 14, 14), "conv2": (3, 64, 14, 14),
        "pool2": (3, 64, 7, 7), "flatten": (3, 3136), "fc1": (3, 128), "fc2": (3, 64),
    }
    for name, module in encoder.named_children():
        x = module(x)
        if name in expected:
            assert tuple(x.shape) == expected[name], name
    assert count_trainable_params(encoder) + count_trainable_params(ClassifierHead()) == 429_258


@pytest.mark.parametrize("hopfield", [False, True])
def test_net_forward_shapes_both_bridges(hopfield: bool) -> None:
    base = tiny_baseline()
    net = build_hopfield_net(base, *tiny_prototypes(), beta=2.0) if hopfield else base
    logits, z, h, aux = net(images(5))
    assert logits.shape == (5, 10)
    assert z.shape == h.shape == (5, 8)
    if hopfield:
        assert aux["weights"].shape == (5, 10)
        assert aux["proto_pred"].shape == (5,)
    else:
        assert aux == {}


def test_hopfield_bridge_has_zero_trainable_params() -> None:
    bridge = HopfieldBridge(*tiny_prototypes(), beta=1.0)
    assert count_trainable_params(bridge) == 0
    assert sum(b.numel() for b in bridge.buffers()) == 10 * 8 + 10


def test_identity_bridge_h_equals_z() -> None:
    _, z, h, _ = tiny_baseline()(images(4))
    assert torch.equal(z, h)


def test_hopfield_bridge_h_is_convex_combination_of_memory() -> None:
    protos, labels = tiny_prototypes()
    bridge = HopfieldBridge(protos, labels, beta=3.0)
    h, aux = bridge(torch.randn(7, 8))
    weights = aux["weights"]
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6)
    assert torch.allclose(h, weights @ protos, atol=1e-5)


def test_s1_large_beta_matches_nearest_prototype() -> None:
    protos, labels = tiny_prototypes()
    z = torch.randn(200, 8, generator=torch.Generator().manual_seed(3))
    bridge = HopfieldBridge(protos, labels, beta=1e4)
    _, aux = bridge(z)
    assert torch.equal(aux["proto_pred"], labels[nearest_prototype(z, protos)])


def test_entropy_decreases_with_beta() -> None:
    protos, labels = tiny_prototypes()
    z = torch.randn(50, 8, generator=torch.Generator().manual_seed(4))
    entropies = []
    for beta in [0.5, 1.0, 5.0, 20.0, 100.0]:
        _, aux = HopfieldBridge(protos, labels, beta=beta)(z)
        entropies.append(attention_entropy(aux["weights"], normalize=True).mean().item())
    assert all(b < a for a, b in itertools.pairwise(entropies))


def test_pixel_placement_shapes_and_convexity() -> None:
    base = tiny_baseline()
    flat = images(20).flatten(1)
    labels = torch.arange(20) % 10
    protos, mem_labels = compute_prototypes(flat, labels)
    net = build_hopfield_net(base, protos, mem_labels, beta=5.0, placement="pixel",
                             center=flat.mean(dim=0))
    assert isinstance(net, PixelHopfieldNet)
    logits, z, h, aux = net(images(3, seed=9))
    assert logits.shape == (3, 10) and z.shape == (3, 8) and torch.equal(z, h)
    retrieved, _ = net.input_bridge(images(3, seed=9).flatten(1))
    assert torch.allclose(retrieved, aux["weights"] @ protos, atol=1e-5)


def test_centered_multi_prototype_bridge() -> None:
    generator = torch.Generator().manual_seed(5)
    z_train = torch.randn(400, 8, generator=generator)
    labels = torch.arange(400) % 10
    memory, mem_labels, center = build_memory(z_train, labels, per_class=3, center=True, seed=0)
    assert memory.shape == (30, 8) and mem_labels.tolist() == sorted(list(range(10)) * 3)
    assert center is not None and torch.allclose(center, z_train.mean(dim=0))
    again, _, _ = build_memory(z_train, labels, per_class=3, center=True, seed=0)
    assert torch.equal(memory, again)  # k-means is seeded

    bridge = HopfieldBridge(memory, mem_labels, beta=4.0, center=center)
    h, aux = bridge(torch.randn(6, 8, generator=generator))
    assert torch.allclose(h, aux["weights"] @ memory, atol=1e-5)  # convex combo of raw memory
    assert torch.allclose(aux["class_mass"].sum(dim=-1), torch.ones(6), atol=1e-6)
    assert torch.equal(aux["proto_pred"], aux["class_mass"].argmax(dim=-1))
    assert bridge.macs_per_image() == 2 * 30 * 8 + 8
    assert count_trainable_params(bridge) == 0


def test_kmeans_recovers_separated_clusters() -> None:
    generator = torch.Generator().manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = torch.cat([c + 0.1 * torch.randn(50, 2, generator=generator) for c in centers])
    found = kmeans(x, 3, generator=torch.Generator().manual_seed(1))
    dist = torch.cdist(centers, found).min(dim=1).values
    assert torch.all(dist < 0.2)


# ---------------------------------------------------------------------------
# Weight sharing
# ---------------------------------------------------------------------------


def test_nets_share_weights_and_beta_change_does_not_touch_them() -> None:
    base = tiny_baseline()
    before = copy.deepcopy(base.state_dict())
    hop = build_hopfield_net(base, *tiny_prototypes(), beta=1.0)
    assert shared_weights_equal(base, hop)
    assert hop.encoder is not base.encoder  # deep copy, not aliasing

    hop_state = copy.deepcopy(hop.encoder.state_dict())
    assert isinstance(hop.bridge, HopfieldBridge)
    hop.bridge.set_retrieval(beta=50.0, num_updates=3)
    assert hop.bridge.beta == 50.0 and hop.bridge.num_updates == 3
    for k, v in hop.encoder.state_dict().items():
        assert torch.equal(v, hop_state[k])
    for k, v in base.state_dict().items():
        assert torch.equal(v, before[k])
    assert shared_weights_equal(base, hop)


# ---------------------------------------------------------------------------
# Corruptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "level"), [("gaussian", 0.5), ("salt_pepper", 0.2), ("occlusion", 10)])
def test_corruptions_deterministic_and_in_range(kind: str, level: float) -> None:
    x = images(8)
    a = make_corrupted_test_set(x, seed=0, kind=kind, level=level)
    b = make_corrupted_test_set(x, seed=0, kind=kind, level=level)
    c = make_corrupted_test_set(x, seed=1, kind=kind, level=level)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert not torch.equal(a, x)


@pytest.mark.parametrize("kind", ["gaussian", "salt_pepper", "occlusion"])
def test_corruption_level_zero_is_identity(kind: str) -> None:
    x = images(4)
    assert torch.equal(corrupt_images(x, kind, 0, torch.Generator().manual_seed(0)), x)


def test_occlusion_zeros_a_square() -> None:
    side = 7
    out = corrupt_images(torch.ones(5, 1, 28, 28), "occlusion", side, torch.Generator().manual_seed(2))
    for img in out[:, 0]:
        zeros = (img == 0).nonzero()
        assert zeros.shape[0] == side * side
        rows, cols = zeros[:, 0], zeros[:, 1]
        assert rows.max() - rows.min() == side - 1
        assert cols.max() - cols.min() == side - 1


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_mcnemar_known_example() -> None:
    # n = 6 discordant, min(b, c) = 1: p = 2 * (C(6,0) + C(6,1)) / 2^6 = 14 / 64.
    assert mcnemar_exact_p(1, 5) == pytest.approx(14 / 64)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == 1.0
    # Large discordant counts (n > 1024) must not overflow float conversion.
    assert 0.0 <= mcnemar_exact_p(3000, 3100) <= 1.0
    assert mcnemar_exact_p(2500, 2500) == 1.0
    # Large discordant counts (n > 1024) must not overflow float.
    assert mcnemar_exact_p(5000, 5000) == 1.0
    assert 0.0 <= mcnemar_exact_p(4000, 3000) < 1e-20
    base = torch.tensor([True, True, False, False, True, False])
    hop = torch.tensor([False, True, True, True, True, False])
    assert mcnemar_test(base, hop) == {"b": 1, "c": 2, "p": pytest.approx(1.0)}


def test_winner_rule() -> None:
    assert decide_winner(0.004, 1e-6) == "tie"
    assert decide_winner(0.02, 0.05) == "tie"
    assert decide_winner(0.02, 0.01) == "Hopfield"
    assert decide_winner(-0.02, 0.01) == "Baseline"
    # One seed significant in the opposite direction: not a reproducible win.
    assert decide_winner(0.02, 0.01, [-1, 1, 1]) == "mixed"
    assert decide_winner(0.02, 0.01, [1, 1, 1]) == "Hopfield"


def test_wilcoxon_known_values() -> None:
    assert wilcoxon_signed_rank_p([1.0] * 10) == pytest.approx(2 / 1024)
    assert wilcoxon_signed_rank_p([0.5, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(2 / 32)
    assert wilcoxon_signed_rank_p([1.0, -1.0, 2.0, -2.0]) == pytest.approx(1.0)
    assert wilcoxon_signed_rank_p([0.0, 0.0]) == 1.0
    # One small negative among 9 positives: W+ = 54 of 55 -> p = 2*2/1024.
    assert wilcoxon_signed_rank_p([-0.1, *[float(i) for i in range(2, 11)]]) == pytest.approx(4 / 1024)


def test_robust_auc_trapezoid() -> None:
    assert robust_auc([0, 1], [1.0, 1.0]) == pytest.approx(1.0)
    assert robust_auc([0, 7, 14], [1.0, 0.5, 0.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_param_counts_and_macs() -> None:
    base = tiny_baseline()
    hop = build_hopfield_net(base, *tiny_prototypes(), beta=1.0, num_updates=2)
    assert count_trainable_params(base) == count_trainable_params(hop)
    for net in (base, hop):
        assert isinstance(net, MnistNet)
        assert sum(m for _, m in analytical_macs(net)) == hook_macs(net)
    # Hopfield term: T * (2 K D + D) with K=10, D=8, T=2.
    assert hook_macs(hop) - hook_macs(base) == 2 * (2 * 10 * 8 + 8)
    full = MnistNet(MnistEncoder(), nn.Identity(), ClassifierHead())
    assert sum(m for _, m in analytical_macs(full)) == 4_248_704


def test_measure_cost_returns_positive_finite_values() -> None:
    hop = build_hopfield_net(tiny_baseline(), *tiny_prototypes(), beta=1.0)
    out = measure_cost(hop, 4, torch.device("cpu"), warmup=1, iters=3)
    for key in ("latency_ms_p50", "latency_ms_p95", "latency_ms_mean", "latency_per_image_us",
                "throughput_img_s", "peak_mem_mb"):
        assert math.isfinite(out[key]) and out[key] > 0, key


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"seeds": []},
        {"seeds": [0, 0]},
        {"seeds": [-1]},
        {"epochs": 0},
        {"epochs": True},
        {"lr": 0},
        {"lr": float("nan")},
        {"val_size": 2.5},
        {"beta_values": [1.0, -2.0]},
        {"beta_values": [5.0, 1.0]},
        {"gaussian_sigmas": [0.1, 0.2]},
        {"salt_pepper_probs": [0.0, 1.5]},
        {"occlusion_sizes": [0, 30]},
        {"num_updates_values": [0, 1]},
        {"placements": ["pixel", "logits"]},
        {"device": "tpu"},
        {"cost": {"batch_size": 0}},
        {"cost": {"bogus": 1}},
        {"unknown_key": 1},
        {"data_dir": ""},
        {"center_prototypes": "yes"},
        {"prototypes_per_class": 0},
        {"beta_selection_sets": []},
        {"beta_selection_sets": [["clean", 0.3]]},
        {"beta_selection_sets": [["gaussian", 0]]},
        {"beta_selection_sets": [["blur", 0.2]]},
        {"beta_selection_sets": [["clean", 0], ["clean", 0]]},
        {"baseline_tuning": [1]},
        {"baseline_tuning": {"bogus": 1}},
        {"baseline_tuning": {"lr_values": []}},
        {"baseline_tuning": {"lr_values": [0.002, 0.001]}},
        {"baseline_tuning": {"batch_size_values": [0]}},
        {"baseline_tuning": {"max_epochs": 0}},
        {"baseline_tuning": {"seeds": [0, 0]}},
    ],
)
def test_config_validation_rejects_bad_values(override: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_mnist_config(override)


def test_config_file_and_quick_overrides_validate() -> None:
    cfg = mb.load_config(mb.DEFAULT_CONFIG_PATH)
    assert cfg["seeds"] == list(range(10)) and cfg["val_size"] == 5000 and cfg["epochs"] == 3
    assert cfg["center_prototypes"] and cfg["prototypes_per_class"] == 8
    assert cfg["beta_selection_sets"] == [("clean", 0.0), ("gaussian", 0.4), ("salt_pepper", 0.15)]
    quick = mb.load_config(mb.DEFAULT_CONFIG_PATH, quick=True)
    assert quick["seeds"] == [0] and quick["epochs"] == 1
    assert quick["train_subset"] == 2000 and quick["test_subset"] == 1000
    assert quick["cost"]["batch_size"] == cfg["cost"]["batch_size"]
    assert quick["reports_dir"] != cfg["reports_dir"]
    assert cfg["baseline_tuning"]["seeds"] == [0, 1] and cfg["baseline_tuning"]["max_epochs"] == 6
    assert quick["baseline_tuning"]["max_epochs"] == 2
    assert validate_mnist_config({})["baseline_tuning"] is None


def test_config_defaults_are_the_memory_protocol() -> None:
    cfg = validate_mnist_config({})
    assert cfg["center_prototypes"] is True and cfg["prototypes_per_class"] == 8
    assert [kind for kind, _ in cfg["beta_selection_sets"]] == ["clean", "gaussian", "salt_pepper"]


# ---------------------------------------------------------------------------
# End to end (synthetic data)
# ---------------------------------------------------------------------------


def synthetic_mnist(data_dir: Any, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """10 classes, each a distinct bright 7x7 block position plus noise, values in [0, 1]."""
    n = 3000 if train else 1000
    generator = torch.Generator().manual_seed(0 if train else 1)
    labels = torch.arange(n) % 10
    x = torch.rand(n, 1, 28, 28, generator=generator) * 0.2
    for c in range(10):
        r, col = (c // 4) * 9 + 2, (c % 4) * 6 + 1
        x[labels == c, :, r : r + 7, col : col + 7] += 0.7
    return x.clamp(0.0, 1.0), labels


TINY_CONFIG = """
data_dir: "unused"
prototypes_per_class: 3
quick:
  seeds: [0]
  epochs: 1
  baseline_tuning:
    seeds: [0]
    lr_values: [0.001, 0.003]
    batch_size_values: [64]
    max_epochs: 2
  val_size: 100
  train_subset: 300
  test_subset: 200
  eval_batch_size: 500
  beta_values: [1.0, 100.0]
  num_updates_values: [1, 2]
  gaussian_sigmas: [0.0, 0.5]
  salt_pepper_probs: [0.0, 0.2]
  occlusion_sizes: [0, 7]
  cost:
    batch_size: 16
    batch_sizes: [1, 16]
    warmup: 1
    iters: 2
    repeats: 2
"""


def test_end_to_end_quick_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Same CLI call as a user's --quick run; only the default config is shrunk for speed.
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    monkeypatch.setattr(mb, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(mb, "load_mnist_tensors", synthetic_mnist)
    main(["--quick", "--reports-dir", str(tmp_path)])

    for name in ("summary.md", "results.csv", "results.json", "plots.png", "cost.png",
                 "baseline_tuning.json"):
        assert (tmp_path / name).exists(), name
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for token in ("Varied:", "Held fixed:", "Baseline", "Hopfield", "VERDICT"):
        assert token in summary
    for i in range(1, 12):
        assert f"S{i} " in summary
    for exp in ("E1", "E2", "E3a", "E3b", "E3c", "E4", "E5", "E6", "E7", "E8", "E9a", "E9b"):
        assert f" {exp}  " in summary
    assert "memory=30 prototypes (3/class, centered)" in summary
    assert "mean of val clean + gaussian 0.4 + salt_pepper 0.15" in summary
    # Split table is the first section; tuning (4 grid points) comes right after it.
    assert summary.index("E0 Data splits") < summary.index("E0b Baseline tuning") < summary.index(" E1  ")
    for split in ("train", "validation", "test"):
        assert f"  {split} " in summary
    tuning = json.loads((tmp_path / "baseline_tuning.json").read_text(encoding="utf-8"))
    assert len(tuning["grid"]) == 4
    assert all(r["val_score"] <= tuning["best_scores"]["val_score"] for r in tuning["grid"])

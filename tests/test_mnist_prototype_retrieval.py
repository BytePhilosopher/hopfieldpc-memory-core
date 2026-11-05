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

import numpy as np
import pytest
import torch
from torch import nn

import benchmarks.cifar10.baseline as cifar_base
import benchmarks.cifar10.hopfield as cifar
import benchmarks.cifar10.tuning as cifar_tuning
import benchmarks.mnist.baseline as mb
import benchmarks.mnist.hopfield as mb_retrieval
import benchmarks.mnist.hopfield_mnist_c as mnist_c
import benchmarks.mnist.hopfield_recentering as recentering
import benchmarks.mnist.tuning as mnist_tuning
from benchmarks.common.tuned_params import TunedParamsError
from benchmarks.mnist.baseline import (
    ClassifierHead,
    EmbeddingClassifier,
    MnistEncoder,
    accuracy_auc_over_severity,
    analytical_macs,
    compute_prototypes,
    corrupt_images,
    count_trainable_params,
    hook_macs,
    make_corrupted_test_set,
    measure_cost,
    nearest_prototype,
    validate_mnist_config,
)
from benchmarks.mnist.hopfield import (
    HopfieldRetrievalLayer,
    PixelHopfieldNet,
    build_hopfield_net,
    build_prototype_memory,
    decide_winner,
    holm_adjusted_p_values,
    kmeans,
    main,
    mcnemar_exact_p,
    mcnemar_test,
    shared_weights_equal,
    spearman_correlation,
    wilcoxon_signed_rank_p,
)
from benchmarks.mnist.hopfield_recentering import BatchRecenteringLayer, mix_corruptions
from hopfieldpc.diagnostics import attention_entropy


def tiny_baseline(seed: int = 0, embed_dim: int = 8) -> EmbeddingClassifier:
    torch.manual_seed(seed)
    encoder = MnistEncoder(c1=4, c2=8, hidden=16, embed_dim=embed_dim)
    return EmbeddingClassifier(encoder, nn.Identity(), ClassifierHead(embed_dim=embed_dim)).eval()


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
    bridge = HopfieldRetrievalLayer(*tiny_prototypes(), beta=1.0)
    assert count_trainable_params(bridge) == 0
    assert sum(b.numel() for b in bridge.buffers()) == 10 * 8 + 10


def test_identity_bridge_h_equals_z() -> None:
    _, z, h, _ = tiny_baseline()(images(4))
    assert torch.equal(z, h)


def test_hopfield_bridge_h_is_convex_combination_of_memory() -> None:
    protos, labels = tiny_prototypes()
    bridge = HopfieldRetrievalLayer(protos, labels, beta=3.0)
    h, aux = bridge(torch.randn(7, 8))
    weights = aux["weights"]
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6)
    assert torch.allclose(h, weights @ protos, atol=1e-5)


def test_s1_large_beta_matches_nearest_prototype() -> None:
    protos, labels = tiny_prototypes()
    z = torch.randn(200, 8, generator=torch.Generator().manual_seed(3))
    bridge = HopfieldRetrievalLayer(protos, labels, beta=1e4)
    _, aux = bridge(z)
    assert torch.equal(aux["proto_pred"], labels[nearest_prototype(z, protos)])


def test_entropy_decreases_with_beta() -> None:
    protos, labels = tiny_prototypes()
    z = torch.randn(50, 8, generator=torch.Generator().manual_seed(4))
    entropies = []
    for beta in [0.5, 1.0, 5.0, 20.0, 100.0]:
        _, aux = HopfieldRetrievalLayer(protos, labels, beta=beta)(z)
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
    memory, mem_labels, center = build_prototype_memory(z_train, labels, per_class=3, center=True, seed=0)
    assert memory.shape == (30, 8) and mem_labels.tolist() == sorted(list(range(10)) * 3)
    assert center is not None and torch.allclose(center, z_train.mean(dim=0))
    again, _, _ = build_prototype_memory(z_train, labels, per_class=3, center=True, seed=0)
    assert torch.equal(memory, again)  # k-means is seeded

    bridge = HopfieldRetrievalLayer(memory, mem_labels, beta=4.0, center=center)
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
    assert isinstance(hop.bridge, HopfieldRetrievalLayer)
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
    assert accuracy_auc_over_severity([0, 1], [1.0, 1.0]) == pytest.approx(1.0)
    assert accuracy_auc_over_severity([0, 7, 14], [1.0, 0.5, 0.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_param_counts_and_macs() -> None:
    base = tiny_baseline()
    hop = build_hopfield_net(base, *tiny_prototypes(), beta=1.0, num_updates=2)
    assert count_trainable_params(base) == count_trainable_params(hop)
    for net in (base, hop):
        assert isinstance(net, EmbeddingClassifier)
        assert sum(m for _, m in analytical_macs(net)) == hook_macs(net)
    # Hopfield term: T * (2 K D + D) with K=10, D=8, T=2.
    assert hook_macs(hop) - hook_macs(base) == 2 * (2 * 10 * 8 + 8)
    full = EmbeddingClassifier(MnistEncoder(), nn.Identity(), ClassifierHead())
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


def tune_quick(config_path: Path, tmp_path: Path, *sections: str, data_dir: str | None = None) -> Path:
    """Run benchmarks.mnist.tuning --quick into a temporary tuned-parameter file."""
    tuned = tmp_path / "tuned_params.json"
    argv = ["--quick", "--config", str(config_path), "--reports-dir", str(tmp_path / "reports"),
            "--tuned-params", str(tuned), "--only", *sections]
    mnist_tuning.main(argv + (["--data-dir", data_dir] if data_dir else []))
    return tuned


def test_end_to_end_quick_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Same CLI calls as a user's --quick run; only the default config is shrunk for speed.
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    monkeypatch.setattr(mb, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(mb, "load_mnist_tensors", synthetic_mnist)
    reports = tmp_path / "reports"

    # Experiments never tune: without stored values they stop and name the command to run.
    with pytest.raises(TunedParamsError, match="benchmarks.mnist.tuning --quick"):
        main(["--quick", "--reports-dir", str(reports), "--tuned-params", str(tmp_path / "tuned_params.json")])

    tuned = tune_quick(config_path, tmp_path, "baseline", "hopfield_l8")
    stored = json.loads(tuned.read_text(encoding="utf-8"))["quick"]
    assert len(stored["baseline"]["values"]["grid"]) == 4
    best = stored["baseline"]["values"]["best_scores"]["val_score"]
    assert all(r["val_score"] <= best for r in stored["baseline"]["values"]["grid"])
    assert stored["hopfield_l8"]["values"]["beta_star"] in (1.0, 100.0)

    # Tuning again is a no-op: nothing is trained or scored.
    monkeypatch.setattr(mnist_tuning, "train_baseline", lambda *a, **k: pytest.fail("baseline was tuned again"))
    monkeypatch.setattr(mnist_tuning, "validation_accuracy_by_beta", lambda *a, **k: pytest.fail("beta tuned again"))
    assert tune_quick(config_path, tmp_path, "baseline", "hopfield_l8").read_text(encoding="utf-8") \
        == tuned.read_text(encoding="utf-8")

    monkeypatch.setattr(mb_retrieval, "validation_accuracy_by_beta",
                        lambda *a, **k: pytest.fail("the experiment selected beta itself"))
    main(["--quick", "--reports-dir", str(reports), "--tuned-params", str(tuned)])

    for name in ("summary.md", "results.csv", "results.json", "plots.png", "cost.png"):
        assert (reports / name).exists(), name
    summary = (reports / "summary.md").read_text(encoding="utf-8")
    for token in ("Varied:", "Held fixed:", "Baseline", "Hopfield", "VERDICT"):
        assert token in summary
    for i in range(1, 12):
        assert f"S{i} " in summary
    for exp in ("E1", "E2", "E3a", "E3b", "E3c", "E4", "E5", "E6", "E7", "E8", "E9a", "E9b"):
        assert f" {exp}  " in summary
    assert "memory=30 prototypes (3/class, centered)" in summary
    assert "mean of val clean + gaussian 0.4 + salt_pepper 0.15" in summary
    # Split table is the first section; tuning comes right after it.
    assert summary.index("E0 Data splits") < summary.index("E0b Baseline tuning") < summary.index(" E1  ")
    for split in ("train", "validation", "test"):
        assert f"  {split} " in summary


def test_changed_settings_make_stored_tuning_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    monkeypatch.setattr(mb, "load_mnist_tensors", synthetic_mnist)
    tuned = tune_quick(config_path, tmp_path, "baseline")
    config_path.write_text(TINY_CONFIG.replace("lr_values: [0.001, 0.003]", "lr_values: [0.001]"), encoding="utf-8")
    cfg = mb.load_config(config_path, quick=True, tuned_params_path=str(tuned))
    with pytest.raises(TunedParamsError, match="changed: tuning"):
        mb.apply_baseline_params(cfg, mb.load_mnist("unused"))


def test_holm_adjustment_is_monotone_and_keeps_input_order() -> None:
    adjusted = holm_adjusted_p_values([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])
    assert holm_adjusted_p_values([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_spearman_correlation_handles_order_and_constant_input() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) is None


def test_load_mnist_c_matches_mnist_tensor_format(tmp_path: Path) -> None:
    folder = tmp_path / "mnist_c" / "fog"
    folder.mkdir(parents=True)
    np.save(folder / "test_images.npy", np.full((3, 28, 28, 1), 255, dtype=np.uint8))
    np.save(folder / "test_labels.npy", np.array([1, 2, 3]))
    images, labels = mb.load_mnist_c_test_set(tmp_path, "fog", test_subset=2)
    assert images.shape == (2, 1, 28, 28)
    assert images.dtype == torch.float32 and images.max() == 1.0
    assert labels.tolist() == [1, 2] and labels.dtype == torch.long


def test_summarize_corruption_suite_uses_holm_for_the_winner() -> None:
    groups = {"a": ("strong", "weak"), "b": ("flat",)}
    gains = {"strong": 0.05, "weak": 0.0049, "flat": 0.0}
    rows = []
    for seed in range(10):
        for name, gain in gains.items():
            step = seed * 1e-4 if gain else 0.0
            for readout, acc in (("baseline", 0.8), ("hopfield", 0.8 + gain + step),
                                 ("nearest_prototype", 0.8 + gain + step)):
                rows.append({
                    "seed": seed, "corruption": name, "group": "x", "readout": readout,
                    "head_acc": acc, "embedding_shift": 0.1, "entropy_norm": 0.1,
                    "true_proto_weight": 0.9, "mcnemar_b": 0, "mcnemar_c": 5, "mcnemar_p": 0.01,
                })
    summary = mnist_c.summarize_corruption_suite(rows, groups)
    winners = {c["corruption"]: c["winner"] for c in summary["per_corruption"]}
    assert winners == {"strong": "Hopfield", "weak": "Hopfield", "flat": "tie"}
    assert summary["overall"]["hopfield_vs_baseline"]["n_pos"] == 10
    assert summary["overall"]["hopfield_vs_nearest_prototype"]["delta"] == pytest.approx(0.0)


def install_fake_mnist_c(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Patch MNIST with quantized synthetic data and write a matching fake MNIST-C folder."""

    def quantized_mnist(data_dir: Any, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = synthetic_mnist(data_dir, train=train)
        return images.mul(255).round().to(torch.uint8).float().div(255.0), labels

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TINY_CONFIG, encoding="utf-8")
    monkeypatch.setattr(mb, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(mb, "load_mnist_tensors", quantized_mnist)
    images, labels = quantized_mnist(None, train=False)
    pixels = images.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for name in [mb.MNIST_C_UNCORRUPTED_NAME, *mb.mnist_c_corruption_names()]:
        folder = tmp_path / "data" / "mnist_c" / name
        folder.mkdir(parents=True)
        np.save(folder / "test_images.npy", pixels)
        np.save(folder / "test_labels.npy", labels.numpy())


def test_mnist_c_end_to_end_quick_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_mnist_c(monkeypatch, tmp_path)

    reports = tmp_path / "reports"
    tuned = tune_quick(tmp_path / "tiny.yaml", tmp_path, "baseline", "hopfield_mnist_c", data_dir=str(tmp_path / "data"))
    mnist_c.main(["--quick", "--reports-dir", str(reports), "--data-dir", str(tmp_path / "data"),
                  "--tuned-params", str(tuned)])

    for name in ("mnist_c_summary.md", "mnist_c_results.csv", "mnist_c_results.json", "mnist_c_plot.png"):
        assert (reports / name).exists(), name
    assert not (reports / "summary.md").exists()
    summary = (reports / "mnist_c_summary.md").read_text(encoding="utf-8")
    assert "VERDICT" in summary and "FAIL" not in summary
    for name in mb.mnist_c_corruption_names():
        assert name in summary
    row_count = len((reports / "mnist_c_results.csv").read_text(encoding="utf-8").splitlines()) - 1
    assert row_count == 16 * 3  # 15 corruptions + uncorrupted, 3 readouts, 1 seed


def test_test_batch_centering_cancels_a_shift_shared_by_the_batch() -> None:
    protos, labels = tiny_prototypes()
    center = protos.mean(dim=0)
    z = protos[:4] + 0.01 * torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    z = z - z.mean(dim=0) + center
    bridge = HopfieldRetrievalLayer(protos, labels, beta=20.0, center=center)
    reference, _ = bridge(z)

    # When the batch mean is the train mean, the two modes are identical.
    bridge.set_retrieval(query_centering="test_batch_mean")
    assert torch.allclose(bridge(z)[0], reference)
    # A shift shared by the whole batch changes nothing in this mode, only in train_mean mode.
    shift = 5.0 * torch.ones(8)
    assert torch.allclose(bridge(z + shift)[0], reference, atol=1e-5)
    bridge.set_retrieval(query_centering="train_mean")
    assert torch.allclose(bridge(z)[0], reference)
    assert not torch.allclose(bridge(z + shift)[0], reference, atol=1e-3)


def test_test_batch_centering_rejects_bad_settings() -> None:
    protos, labels = tiny_prototypes()
    with pytest.raises(ValueError, match="requires center"):
        HopfieldRetrievalLayer(protos, labels, beta=1.0).set_retrieval(query_centering="test_batch_mean")
    with pytest.raises(ValueError, match="query_centering"):
        HopfieldRetrievalLayer(protos, labels, beta=1.0).set_retrieval(query_centering="median")


def test_batch_recentering_layer_moves_the_batch_mean_to_the_train_mean() -> None:
    train_mean = torch.arange(8.0)
    z = torch.randn(6, 8, generator=torch.Generator().manual_seed(0))
    h = BatchRecenteringLayer(train_mean)(z)
    assert torch.allclose(h.mean(dim=0), train_mean, atol=1e-5)
    assert torch.allclose(h - h[0], z - z[0], atol=1e-5)


def test_mix_corruptions_keeps_image_order_and_is_seeded() -> None:
    sets = [torch.full((50, 2), float(i)) for i in range(3)]
    sets = [s + torch.arange(50.0)[:, None] * 10 for s in sets]
    mixed = mix_corruptions(sets, seed=0)
    assert torch.equal(mixed, mix_corruptions(sets, seed=0))
    assert torch.equal((mixed[:, 0] // 10), torch.arange(50.0))
    assert len(set((mixed[:, 0] % 10).tolist())) == 3


def test_mnist_c_recentering_end_to_end_quick_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_mnist_c(monkeypatch, tmp_path)
    reports = tmp_path / "reports"
    tuned = tune_quick(tmp_path / "tiny.yaml", tmp_path, "baseline", "hopfield_mnist_c", data_dir=str(tmp_path / "data"))
    recentering.main(["--quick", "--reports-dir", str(reports), "--data-dir", str(tmp_path / "data"),
                      "--tuned-params", str(tuned)])

    for name in ("summary.md", "results.csv", "results.json", "plot.png"):
        assert (reports / f"mnist_c_recentering_{name}").exists(), name
    assert not (reports / "mnist_c_summary.md").exists()
    summary = (reports / "mnist_c_recentering_summary.md").read_text(encoding="utf-8")
    assert "VERDICT" in summary and "FAIL" not in summary and "Mixed corruptions" in summary
    row_count = len((reports / "mnist_c_recentering_results.csv").read_text(encoding="utf-8").splitlines()) - 1
    assert row_count == 17 * (2 + 2 * 4)  # 15 + uncorrupted + mixed, 2 fixed readouts, 2 x 4 batch sizes


def test_random_crop_and_flip_keeps_shape_and_uses_source_pixels() -> None:
    torch.manual_seed(0)
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    out = cifar_base.random_crop_and_flip(images, padding=2)
    assert out.shape == images.shape
    assert set(out.flatten().tolist()) <= set(images.flatten().tolist())
    # All channels of one pixel come from the same source position.
    assert torch.equal(out[:, 1] - out[:, 0], torch.full_like(out[:, 0], 64.0))


def test_cifar_encoder_output_shape_and_channel_order() -> None:
    encoder = cifar_base.CifarResNetEncoder(base_width=2, embed_dim=8).eval()
    assert encoder(torch.rand(4, 3, 32, 32)).shape == (4, 8)


def test_load_cifar10_c_selects_the_requested_severity_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cifar_base, "IMAGES_PER_SEVERITY", 2)
    root = tmp_path / "CIFAR-10-C"
    root.mkdir()
    pixels = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    for index in range(10):
        pixels[index] = index
    np.save(root / "fog.npy", pixels)
    np.save(root / "labels.npy", np.arange(10) % 10)
    images, labels = cifar_base.load_cifar10_c_test_set(tmp_path, "fog", 3)
    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [4, 5]
    assert images[0, 0, 0, 0].item() == pytest.approx(4 / 255)
    with pytest.raises(ValueError, match="severity"):
        cifar_base.load_cifar10_c_test_set(tmp_path, "fog", 6)


def test_collapse_to_severity_auc_includes_clean_as_severity_zero() -> None:
    rows = [{"seed": 0, "test_set": "identity", "severity": 0, "readout": "baseline",
             "batch_size": None, "head_acc": 1.0}]
    rows += [{"seed": 0, "test_set": "fog", "severity": s, "readout": "baseline", "batch_size": None,
              "head_acc": 1.0 - s * 0.2} for s in cifar_base.SEVERITIES]
    collapsed = cifar.collapse_to_severity_auc(rows)
    fog = next(r for r in collapsed if r["test_set"] == "fog")
    assert fog["head_acc"] == pytest.approx(0.5)
    assert len(collapsed) == 2


def _tiny_cifar_data(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    n = 300 if train else 40
    generator = torch.Generator().manual_seed(0 if train else 1)
    labels = torch.arange(n) % 10
    images = torch.rand(n, 3, 32, 32, generator=generator) * 0.2
    for c in range(10):
        images[labels == c, :, c * 3 : c * 3 + 2] += 0.7
    return images.clamp(0, 1).mul(255).round().to(torch.uint8).float().div(255), labels


def install_tiny_cifar(monkeypatch: pytest.MonkeyPatch) -> None:
    train_images, train_labels = _tiny_cifar_data(True)
    test_images, test_labels = _tiny_cifar_data(False)
    data = (train_images, train_labels, test_images, test_labels)
    for module in (cifar, cifar_tuning):
        monkeypatch.setattr(module, "load_cifar10", lambda _: data)


def test_cifar10_c_end_to_end_resumes_without_rework(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for module in (cifar, cifar_base):
        monkeypatch.setattr(module, "IMAGES_PER_SEVERITY", 40)
    install_tiny_cifar(monkeypatch)
    test_images, test_labels = _tiny_cifar_data(False)
    root = tmp_path / "data" / "CIFAR-10-C"
    root.mkdir(parents=True)
    pixels = test_images.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for name in cifar_base.CORRUPTION_NAMES:
        np.save(root / f"{name}.npy", np.concatenate([pixels] * 5))
    np.save(root / "labels.npy", np.concatenate([test_labels.numpy()] * 5))

    cfg = cifar.Cifar10CConfig(
        data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"), seeds=[0], base_width=2,
        embed_dim=8, epochs=1, batch_size=32, val_size=50, min_clean_val_accuracy=0.0,
        prototypes_per_class=2, beta_values=[1.0, 100.0], eval_batch_size=64,
        tuned_params=str(tmp_path / "tuned_params.json"),
    )
    with pytest.raises(TunedParamsError, match="benchmarks.cifar10.tuning"):
        cifar.run_experiment(cfg)
    tuned = cifar_tuning.tune(cfg)
    assert tuned["beta_star"] in (1.0, 100.0) and tuned["num_cached_backbones"] == 0
    raw, text = cifar.run_experiment(cfg)
    for name in ("cifar10_c_summary.md", "cifar10_c_results.csv", "cifar10_c_results.json", "cifar10_c_plot.png",
                 "backbone_seed0.pt", "validation_seed0.json", "results_seed0.json"):
        assert (tmp_path / "out" / name).exists(), name
    assert "CIFAR-10-C" in text and "VERDICT" in text and "FAIL" not in text
    assert "accuracy AUC over severity" in text and "Soft vs hard retrieval" in text
    assert raw["num_cached_backbones"] == 0 and raw["beta_star"] == tuned["beta_star"]

    # A rerun after a disconnect must not train, tune or evaluate again.
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("work was repeated on resume")

    monkeypatch.setattr(cifar_base, "train_backbone", fail)
    monkeypatch.setattr(cifar, "prepare_seed_run", fail)
    monkeypatch.setattr(cifar_tuning, "prepare_seed_run", fail)
    assert cifar_tuning.tune(cfg) == tuned
    _, resumed = cifar.run_experiment(cfg)
    assert resumed == text


def test_cifar10_c_stops_before_test_data_when_validation_accuracy_is_low(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_tiny_cifar(monkeypatch)

    def fail_download(_: Any) -> None:
        raise AssertionError("corrupted data was requested")

    monkeypatch.setattr(cifar, "ensure_cifar10_c_downloaded", fail_download)
    cfg = cifar.Cifar10CConfig(
        data_dir=str(tmp_path), output_dir=str(tmp_path / "out"), seeds=[0], base_width=2, embed_dim=8,
        epochs=1, batch_size=32, val_size=50, min_clean_val_accuracy=1.01, prototypes_per_class=2,
        beta_values=[1.0], eval_batch_size=64, tuned_params=str(tmp_path / "tuned_params.json"),
    )
    with pytest.raises(SystemExit, match="below"):
        cifar_tuning.tune(cfg)
    assert not (tmp_path / "tuned_params.json").exists()


def test_cifar10_c_refuses_to_run_without_a_gpu_unless_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="No GPU"):
        cifar.main(["--quick"])

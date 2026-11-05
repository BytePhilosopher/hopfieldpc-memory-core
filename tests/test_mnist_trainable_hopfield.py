"""Offline tests for the trainable Hopfield benchmark (random-init nets, synthetic tensors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import benchmarks.mnist.hopfield_trainable as th
import benchmarks.mnist.tuning_trainable as tuning_trainable
from benchmarks.common import tuned_params
from benchmarks.mnist.hopfield_l5 import L5_SECTION, l5_fingerprint
from benchmarks.mnist.baseline import (
    build_baseline_net,
    checkpoint_fingerprint,
    make_splits,
)
from benchmarks.mnist.hopfield import HopfieldRetrievalLayer

REPO = Path(__file__).resolve().parents[1]


def synthetic_data() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(400, 1, 28, 28, generator=generator)
    labels = torch.arange(400) % 10
    return images[:300], labels[:300], images[300:], labels[300:]


def tiny_config(reports_dir: Path) -> dict[str, Any]:
    cfg, _ = th.load_trainable_config(REPO / "configs" / "mnist" / "hopfield_trainable.yaml", quick=True)
    cfg.update(
        reports_dir=str(reports_dir), seeds=[0], baseline_tuning=None, val_size=50, train_subset=None, test_subset=None,
        epochs=1, batch_size=32, lr=0.001, eval_batch_size=64, gaussian_sigmas=[0.0, 0.5], salt_pepper_probs=[0.0, 0.1],
        occlusion_sizes=[0, 7], device="cpu", prior_rows={}, tuned_params=str(reports_dir / "tuned_params.json"),
    )
    cfg["tuning"] = {"seed": 0, "train_images": 100, "epochs": 1, "memory_lr_values": [0.001], "beta_factors": [1.0]}
    cfg["cost_settings"] = {"batch_size": 2, "repeats": 1, "warmup": 1, "iters": 1}
    return cfg


def save_backbone(cfg: dict[str, Any], data: tuple[torch.Tensor, ...], seed: int = 0) -> None:
    splits = make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"])
    torch.save(
        {"state_dict": build_baseline_net(seed).state_dict(),
         "fingerprint": checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0]), "train_time_s": 0.0},
        Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt",
    )


def layer(dim: int = 16, beta: float = 20.0) -> th.TrainableHopfieldLayer:
    memory = torch.randn(20, dim, generator=torch.Generator().manual_seed(1))
    labels = torch.arange(20) % 10
    return th.TrainableHopfieldLayer(memory, labels, memory.mean(dim=0), beta=beta)


def test_layer_matches_the_fixed_retrieval_before_training() -> None:
    trainable = layer()
    fixed = HopfieldRetrievalLayer(trainable.memory.detach(), trainable.memory_labels, beta=20.0, center=trainable.center)
    z = torch.randn(5, 16, generator=torch.Generator().manual_seed(2))
    h, aux = trainable(z)
    fixed_h, fixed_aux = fixed(z)
    assert torch.allclose(h, fixed_h, atol=1e-5)
    assert torch.allclose(aux["weights"], fixed_aux["weights"], atol=1e-5)
    assert torch.equal(aux["proto_pred"], fixed_aux["proto_pred"])


def test_gradients_reach_memory_and_beta() -> None:
    trainable = layer()
    h, _ = trainable(torch.randn(5, 16))
    h.sum().backward()
    assert trainable.memory.grad.abs().sum() > 0
    assert trainable.log_beta.grad is not None


def test_center_only_moves_when_tracking_in_training_mode() -> None:
    trainable = layer()
    before = trainable.center.clone()
    trainable.train()
    trainable(torch.ones(5, 16))
    assert torch.equal(trainable.center, before)
    trainable.track_center = True
    trainable.eval()
    trainable(torch.ones(5, 16))
    assert torch.equal(trainable.center, before)
    trainable.train()
    trainable(torch.ones(5, 16))
    assert not torch.equal(trainable.center, before)


def test_augment_batch_keeps_a_clean_third_and_corrupts_the_rest() -> None:
    x = torch.full((9, 1, 28, 28), 0.5)
    out = th.augment_batch(x, [0.25], [0.15], torch.Generator().manual_seed(0))
    assert torch.equal(out[:3], x[:3])
    assert not torch.equal(out[3:6], x[3:6]) and not torch.equal(out[6:], x[6:])
    assert torch.equal(x, torch.full((9, 1, 28, 28), 0.5))


def test_memory_loss_prefers_the_true_class_prototype() -> None:
    cfg = {"cluster_weight": 1.0, "separation_weight": 1.0}
    labels = torch.tensor([0])
    memory_labels = torch.tensor([0, 1])
    logits = torch.zeros(1, 2)
    near = {"cosine": torch.tensor([[0.9, 0.1]]), "memory_labels": memory_labels}
    far = {"cosine": torch.tensor([[0.1, 0.9]]), "memory_labels": memory_labels}
    assert th.memory_loss(logits, near, labels, cfg) < th.memory_loss(logits, far, labels, cfg)


def test_set_trainable_freezes_each_group() -> None:
    base = build_baseline_net(0)
    net = th.PlacedNet(base.encoder, base.head, layer(3136), "l5")
    th.set_trainable(net, memory=True)
    assert net.bridge.memory.requires_grad and not net.encoder.fc1.weight.requires_grad
    th.set_trainable(net, memory=True, finetune=True)
    assert net.encoder.fc1.weight.requires_grad and not net.encoder.conv1.weight.requires_grad


def test_end_to_end_resumes_and_leaves_backbones_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = synthetic_data()
    backbones = tmp_path / "backbones"
    backbones.mkdir()
    cfg = tiny_config(backbones)
    save_backbone(cfg, data)
    saved = (backbones / "backbone_seed0.pt").read_bytes()
    monkeypatch.setattr(th, "load_mnist", lambda _: data)
    output = tmp_path / "output"
    # The fixed betas come from the stored L5 tuning; store them directly to keep the test small.
    num_train = make_splits(data, seed=0, val_size=cfg["val_size"], train_subset=None).train_images.shape[0]
    tuned_params.save_entry(cfg["tuned_params"], "quick", L5_SECTION, l5_fingerprint(cfg, num_train),
                            {"beta_l5": 10.0, "beta_l8": 10.0})
    best = tuning_trainable.tune_hopfield_trainable(cfg, data, torch.device("cpu"))["best"]

    raw = th.run_experiment(cfg, output)
    assert raw["tuning"]["best"] == best
    text, summary = th.build_report(raw)
    th.save_reports(raw, text, summary, output)
    assert (backbones / "backbone_seed0.pt").read_bytes() == saved
    assert {"summary.md", "results.csv", "results.json", "seeds"} <= {p.name for p in output.iterdir()}
    assert set(raw["seeds"][0]["rows"]["noise_gaussian"][0]["metrics"]) == set(th.MODELS)
    assert raw["seeds"][0]["sanity"]["convs_frozen"]

    monkeypatch.setattr(th, "run_seed", lambda *a, **k: pytest.fail("finished seed was run again"))
    assert th.run_experiment(cfg, output)["seeds"][0]["seed"] == 0

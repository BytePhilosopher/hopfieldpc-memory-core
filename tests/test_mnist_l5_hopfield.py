"""Offline tests for the L5 Hopfield benchmark (random-init nets, synthetic tensors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import benchmarks.mnist.hopfield_l5 as l5
import benchmarks.mnist.tuning as tuning
from benchmarks.common.tuned_params import TunedParamsError
from benchmarks.mnist.baseline import (
    build_baseline_net,
    checkpoint_fingerprint,
    make_splits,
)
from benchmarks.mnist.hopfield import shared_weights_equal

REPO = Path(__file__).resolve().parents[1]


def synthetic_data() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(400, 1, 28, 28, generator=generator)
    labels = torch.arange(400) % 10
    return images[:300], labels[:300], images[300:], labels[300:]


def tiny_config(reports_dir: Path) -> dict[str, Any]:
    cfg, _ = l5.load_l5_config(REPO / "configs" / "mnist" / "hopfield_l5.yaml", quick=True)
    cfg.update(
        reports_dir=str(reports_dir),
        seeds=[0],
        baseline_tuning=None,
        val_size=50,
        train_subset=None,
        test_subset=None,
        epochs=1,
        batch_size=32,
        lr=0.001,
        eval_batch_size=64,
        l5_betas=[1.0, 100.0],
        gaussian_sigmas=[0.0, 0.5],
        salt_pepper_probs=[0.0, 0.1],
        occlusion_sizes=[0, 7],
        noisy_sigma=0.5,
        num_updates_values=[1, 2],
        device="cpu",
        tuned_params=str(reports_dir / "tuned_params.json"),
    )
    cfg["cost"] = {**cfg["cost"], "batch_sizes": [2], "batch_size": 2, "warmup": 1, "iters": 1, "repeats": 1}
    return cfg


def save_backbone(cfg: dict[str, Any], data: tuple[torch.Tensor, ...], seed: int = 0) -> Path:
    splits = make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"])
    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt"
    torch.save(
        {
            "state_dict": build_baseline_net(seed).state_dict(),
            "fingerprint": checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0]),
            "train_time_s": 0.0,
        },
        path,
    )
    return path


def test_split_reproduces_the_encoder() -> None:
    encoder = build_baseline_net(0).encoder.eval()
    x = torch.rand(4, 1, 28, 28)
    features = l5.flatten_features(encoder, x)
    assert features.shape == (4, 3136)
    assert torch.equal(l5.embed_features(encoder, features), encoder(x))


def test_l5_net_adds_only_buffers_and_keeps_weights() -> None:
    baseline = build_baseline_net(0).eval()
    train = torch.rand(60, 3136)
    labels = torch.arange(60) % 10
    net = l5.build_l5_net(baseline, train, labels, per_class=1, center=True, beta=10.0, seed=0)
    assert shared_weights_equal(baseline, net)
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in baseline.parameters())
    assert net.bridge.prototypes.shape == (10, 3136)
    logits, z, h, aux = net(torch.rand(3, 1, 28, 28))
    assert logits.shape == (3, 10) and z.shape == (3, 64) and h is z
    assert aux["weights"].shape == (3, 10)


def test_l5_output_is_built_from_the_prototypes_at_high_beta() -> None:
    baseline = build_baseline_net(0).eval()
    train = torch.rand(60, 3136)
    labels = torch.arange(60) % 10
    net = l5.build_l5_net(baseline, train, labels, per_class=1, center=True, beta=1e4, seed=0)
    prototype = net.bridge.prototypes[3:4]
    _, z, _, aux = net.forward_from_embedding(prototype)
    assert aux["proto_pred"].item() == 3
    assert torch.allclose(z, l5.embed_features(net.encoder, prototype), atol=1e-4)


def test_missing_or_mismatched_backbone_stops_the_run(tmp_path: Path) -> None:
    data = synthetic_data()
    cfg = tiny_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        l5.require_cached_backbone(cfg, data, 0)
    save_backbone(cfg, data)
    l5.require_cached_backbone(cfg, data, 0)
    with pytest.raises(ValueError, match="trained with"):
        l5.require_cached_backbone({**cfg, "epochs": 99}, data, 0)


def test_compare_marks_small_or_unreliable_differences_as_ties() -> None:
    assert l5.winner(l5.compare([0.02] * 3), "hopfield_l5", "baseline") == "tie"  # too few seeds
    assert l5.winner(l5.compare([0.001 + i * 1e-5 for i in range(10)]), "hopfield_l5", "baseline") == "tie"
    big = l5.compare([0.02 + i * 1e-3 for i in range(10)])
    assert l5.winner(big, "hopfield_l5", "baseline") == "Hopfield@L5"
    assert l5.winner(l5.compare([-0.02 - i * 1e-3 for i in range(10)]), "hopfield_l5", "baseline") == "Baseline"


def test_end_to_end_writes_only_to_the_output_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = synthetic_data()
    backbones = tmp_path / "backbones"
    backbones.mkdir()
    cfg = tiny_config(backbones)
    save_backbone(cfg, data)
    before = sorted(p.name for p in backbones.iterdir())
    monkeypatch.setattr(l5, "load_mnist", lambda _: data)

    with pytest.raises(TunedParamsError, match="benchmarks.mnist.tuning"):
        l5.run_experiment(cfg)
    tuned = tuning.tune_hopfield_l5(cfg, data, torch.device("cpu"))
    before = sorted(p.name for p in backbones.iterdir())
    monkeypatch.setattr(l5, "validation_curve", lambda *a, **k: pytest.fail("the experiment selected beta itself"))
    raw = l5.run_experiment(cfg)
    assert (raw["beta_l5"], raw["beta_l8"]) == (tuned["beta_l5"], tuned["beta_l8"])
    text, summary = l5.build_report(raw)
    output = tmp_path / "output"
    l5.save_reports(raw, text, summary, output)

    assert sorted(p.name for p in backbones.iterdir()) == before
    assert {"summary.md", "results.csv", "results.json"} <= {p.name for p in output.iterdir()}
    assert summary["verdict"] and "Hopfield@L5" in text
    assert set(raw["results"]["clean"][0]["metrics"]) == set(l5.MODELS)

"""Task 13: MNIST prototype retrieval, Baseline vs Hopfield as a controlled experiment.

Both variants are the same CNN (``benchmarks.mnist_baseline``). The only
difference is layer L8, between the embedding z (L7) and the classifier head
(L9):

    Baseline: h = z                                   (nn.Identity)
    Hopfield: p = softmax(beta * cos(z, M)), h = p M  (hopfieldpc.HopfieldMemory)

M holds 8 k-means prototypes per class of the training embeddings, centered
by the train-mean embedding, as a frozen buffer, so the Hopfield variant adds
zero trainable parameters and nothing is retrained. beta is selected on clean
+ noisy validation sets. Every experiment varies one parameter, every
Baseline-vs-Hopfield pair is evaluated on the same (seeded) test images and
compared with an exact McNemar test per seed and an exact Wilcoxon
signed-rank test across seeds.

Run from the repository root:

    python -m benchmarks.mnist_prototype_retrieval            # everything
    python -m benchmarks.mnist_prototype_retrieval --quick    # smoke run
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmarks.mnist_baseline import (
    NOISE_TYPES,
    BaselineRun,
    ClassifierHead,
    MnistEncoder,
    MnistNet,
    analytical_macs,
    build_arg_parser,
    compute_prototypes,
    config_from_args,
    configure_threads,
    copy_net,
    count_buffer_values,
    count_trainable_params,
    environment_info,
    format_split_table,
    format_tuning_table,
    forward_embeddings,
    forward_images,
    hook_macs,
    interleaved_cost,
    load_mnist,
    mean_std,
    median_iqr,
    model_size_bytes,
    nearest_prototype,
    noise_grids,
    prepare_baseline_run,
    resolve_device,
    robust_auc,
    split_table,
    summarize_outputs,
    time_full_evaluation,
    tune_baseline,
)
from hopfieldpc import HopfieldMemory
from hopfieldpc.api import validate_beta

WINNER_DELTA = 0.005
ALPHA = 0.05
S4_MIN_PROTO_ACC = 0.95
S7_TOLERANCE = 0.01
S11_MAX_OVERHEAD_PCT = 10.0

BASELINE_COLOR = "#2a78d6"
HOPFIELD_COLOR = "#eb6834"

RULE = "═" * 96
MODELS = ("baseline", "hopfield")
NOISE_LABELS = {"gaussian": "σ", "salt_pepper": "p", "occlusion": "side"}
E3_IDS = {"gaussian": "E3a", "salt_pepper": "E3b", "occlusion": "E3c"}


# ---------------------------------------------------------------------------
# Hopfield variant
# ---------------------------------------------------------------------------


class HopfieldBridge(nn.Module):
    """Layer L8 of the Hopfield variant: retrieval over frozen class prototypes.

    Wraps ``HopfieldMemory(dim, beta, num_updates, normalize=True,
    return_diagnostics=True)`` and always calls it with the prototype buffer
    as external memory. There are no trainable parameters.

    Shapes:
        prototypes:    [K, D]  (buffer; K = classes x prototypes per class)
        memory_labels: [K]     (buffer; class of each stored pattern)
        input z:       [B, D]  ->  h [B, D], aux dict

    With ``center`` [D] (required for ``placement="pixel"``, optional for the
    embedding), query and prototypes are centered before scoring and the
    center is added back, so h is still a convex combination of the raw
    prototypes. Centering removes the component shared by all prototypes,
    which increases their separation (Ramsauer et al. 2020, A.6.2).

    aux holds the HopfieldMemory diagnostics plus ``weights`` [B, K],
    ``class_mass`` [B, C] (weight summed per class) and ``proto_pred`` [B] =
    argmax class_mass, which equals memory_labels[argmax weights] when each
    class has one prototype.
    """

    def __init__(
        self,
        prototypes: torch.Tensor,
        memory_labels: torch.Tensor,
        *,
        beta: float,
        num_updates: int = 1,
        placement: str = "embedding",
        center: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if prototypes.ndim != 2:
            raise ValueError(f"prototypes must be [K, D], got {tuple(prototypes.shape)}.")
        if memory_labels.shape != (prototypes.shape[0],):
            raise ValueError(
                f"memory_labels must be [{prototypes.shape[0]}], got {tuple(memory_labels.shape)}."
            )
        if placement not in ("embedding", "pixel"):
            raise ValueError(f"placement must be 'embedding' or 'pixel', got {placement!r}.")
        if placement == "pixel" and center is None:
            raise ValueError("center is required for placement='pixel'.")

        self.placement = placement
        self.register_buffer("prototypes", prototypes.detach().clone().float())
        self.register_buffer("memory_labels", memory_labels.detach().clone().long())
        self.num_classes = int(memory_labels.max().item()) + 1
        self.center: torch.Tensor | None
        if center is not None:
            if center.shape != (prototypes.shape[1],):
                raise ValueError(f"center must be [{prototypes.shape[1]}], got {tuple(center.shape)}.")
            self.register_buffer("center", center.detach().clone().float())
        else:
            self.center = None
        self.memory = HopfieldMemory(
            dim=prototypes.shape[1],
            beta=beta,
            num_updates=num_updates,
            normalize=True,
            return_diagnostics=True,
        )

    @property
    def beta(self) -> float:
        return self.memory.beta

    @property
    def num_updates(self) -> int:
        return self.memory.num_updates

    def set_retrieval(self, *, beta: float | None = None, num_updates: int | None = None) -> None:
        """Change beta and/or T in place (buffers and the shared layers are untouched)."""
        if beta is not None:
            self.memory.beta = validate_beta(beta)
        if num_updates is not None:
            if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
                raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")
            self.memory.num_updates = num_updates

    def macs_per_image(self) -> int:
        """Per step: K*D (scores) + K*D (readout) + D (query L2 norm).

        The prototype L2 norm (K*D) is computed once per batch, not per image,
        and is excluded.
        """
        k, d = self.prototypes.shape
        return self.num_updates * (2 * k * d + d)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.center is not None:
            retrieved, weights, diagnostics = self.memory(
                z - self.center, memory=self.prototypes - self.center
            )
            h = retrieved + self.center
        else:
            h, weights, diagnostics = self.memory(z, memory=self.prototypes)
        class_mass = weights.new_zeros(weights.shape[0], self.num_classes)
        class_mass.index_add_(1, self.memory_labels, weights)
        aux = {
            **diagnostics,
            "weights": weights,
            "class_mass": class_mass,
            "proto_pred": class_mass.argmax(dim=-1),
        }
        return h, aux


class PixelHopfieldNet(nn.Module):
    """Ablation E6: Hopfield on raw pixels, then the same encoder + head.

    image [B, 1, 28, 28] -> flatten [B, 784] -> HopfieldBridge(pixel) ->
    reshape [B, 1, 28, 28] -> encoder -> z [B, D] -> head -> logits.
    Returns ``(logits, z, h, aux)`` with h = z (L8 is the identity here).
    """

    def __init__(self, input_bridge: HopfieldBridge, encoder: MnistEncoder, head: ClassifierHead) -> None:
        super().__init__()
        self.input_bridge = input_bridge
        self.encoder = encoder
        self.head = head

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        retrieved, aux = self.input_bridge(images.flatten(1))
        z = self.encoder(retrieved.view_as(images))
        return self.head(z), z, z, aux


def build_hopfield_net(
    baseline_net: MnistNet,
    prototypes: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    beta: float,
    num_updates: int = 1,
    placement: str = "embedding",
    center: torch.Tensor | None = None,
) -> nn.Module:
    """Hopfield variant: deep copies of the baseline's L1-L7 and L9 plus a HopfieldBridge.

    ``prototypes`` is [K, 64] for ``placement="embedding"`` (returns MnistNet;
    ``center`` optional) and [10, 784] with ``center`` [784] for
    ``placement="pixel"`` (returns PixelHopfieldNet). Nothing is retrained.
    """
    device = next(baseline_net.parameters()).device
    encoder = copy_net(baseline_net.encoder)
    head = copy_net(baseline_net.head)
    bridge = HopfieldBridge(
        prototypes,
        memory_labels,
        beta=beta,
        num_updates=num_updates,
        placement=placement,
        center=center,
    ).to(device)
    if placement == "pixel":
        return PixelHopfieldNet(bridge, encoder, head).eval()
    return MnistNet(encoder, bridge, head).eval()


def kmeans(
    x: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator,
    iters: int = 25,
) -> torch.Tensor:
    """Seeded k-means++ / Lloyd on rows of x [N, D]; returns centroids [k, D]."""
    if k >= x.shape[0]:
        raise ValueError(f"k ({k}) must be smaller than the number of points ({x.shape[0]}).")
    first = torch.randint(0, x.shape[0], (1,), generator=generator)
    centroids = x[first]
    while centroids.shape[0] < k:
        d2 = torch.cdist(x, centroids).min(dim=1).values.pow(2)
        nxt = torch.multinomial(d2 / d2.sum(), 1, generator=generator)
        centroids = torch.cat([centroids, x[nxt]])
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        for j in range(k):
            members = x[assign == j]
            if members.shape[0] > 0:
                centroids[j] = members.mean(dim=0)
    return centroids


def build_memory(
    z_train: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    center: bool,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Stored patterns for the embedding bridge.

    per_class = 1 gives the class means; larger
    values store k-means centroids of each class's training embeddings
    (Ramsauer et al. 2020, Table A.3: 1 or 8 stored patterns per class).
    ``center`` returns the train-mean embedding [D] used to center scores.

    Returns (memory [C*per_class, D], memory_labels [C*per_class], center | None).
    """
    generator = torch.Generator().manual_seed(seed)
    blocks, block_labels = [], []
    for c in range(num_classes):
        zc = z_train[labels == c]
        blocks.append(zc.mean(dim=0, keepdim=True) if per_class == 1
                      else kmeans(zc, per_class, generator=generator))
        block_labels.append(torch.full((per_class,), c, dtype=torch.long))
    mean = z_train.mean(dim=0) if center else None
    return torch.cat(blocks), torch.cat(block_labels), mean


def shared_weights_equal(a: nn.Module, b: nn.Module) -> bool:
    """True when L1-L7 (encoder) and L9 (head) tensors are exactly equal."""
    for part in ("encoder", "head"):
        sa, sb = getattr(a, part).state_dict(), getattr(b, part).state_dict()
        if sa.keys() != sb.keys() or not all(torch.equal(sa[k], sb[k]) for k in sa):
            return False
    return True


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    # C(n, i+1) = C(n, i) * (n - i) / (i + 1), kept as exact integers.
    term = tail = 1
    for i in range(min(b, c)):
        term = term * (n - i) // (i + 1)
        tail += term
    # Exact integer division: tail and 2**n overflow float for n > ~1000.
    return min(1.0, 2 * tail / 2**n)


def mcnemar_test(correct_baseline: torch.Tensor, correct_hopfield: torch.Tensor) -> dict[str, Any]:
    """Paired comparison of two bool vectors [N] on the same samples.

    b = baseline right & Hopfield wrong; c = baseline wrong & Hopfield right.
    """
    b = int((correct_baseline & ~correct_hopfield).sum().item())
    c = int((~correct_baseline & correct_hopfield).sum().item())
    return {"b": b, "c": c, "p": mcnemar_exact_p(b, c)}


def wilcoxon_signed_rank_p(diffs: Sequence[float]) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value for paired per-seed differences.

    Zero differences are dropped; tied |d| get average ranks. The null
    distribution is enumerated exactly over all 2^n sign patterns (via a sum
    DP on doubled ranks), so no normal approximation or scipy is needed.
    With all n seeds in the same direction, p = 2 / 2^n.
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n  # doubled average ranks keep the DP on integers
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks2[order[t]] = i + j + 2
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks2, d) if x > 0)
    total = sum(ranks2)
    counts = {0: 1}
    for r in ranks2:
        nxt: dict[int, int] = {}
        for value, cnt in counts.items():
            nxt[value] = nxt.get(value, 0) + cnt
            nxt[value + r] = nxt.get(value + r, 0) + cnt
        counts = nxt
    observed = abs(2 * w_plus - total)
    extreme = sum(cnt for value, cnt in counts.items() if abs(2 * value - total) >= observed)
    return min(1.0, extreme / 2**n)


MIN_SEEDS_FOR_WILCOXON = 6


def decide_winner(
    delta: float,
    p_value: float,
    seed_signs: Sequence[int] = (),
    wilcoxon_p: float | None = None,
) -> str:
    """Winner of a paired comparison pooled over seeds.

    With ``wilcoxon_p`` (enough seeds for an across-seed test): 'tie' if
    |delta| < 0.005 or wilcoxon_p >= 0.05, else the better model. This tests
    reproducibility across independently trained networks.

    Without it (few seeds): 'tie' if |delta| < 0.005 or p >= 0.05 (p is the
    max McNemar p over seeds); 'mixed' if any seed's discordant counts point
    the other way; otherwise the model with the higher accuracy.
    """
    if wilcoxon_p is not None:
        if abs(delta) < WINNER_DELTA or wilcoxon_p >= ALPHA:
            return "tie"
        return "Hopfield" if delta > 0 else "Baseline"
    if abs(delta) < WINNER_DELTA or p_value >= ALPHA:
        return "tie"
    direction = 1 if delta > 0 else -1
    if any(s != direction for s in seed_signs):
        return "mixed"
    return "Hopfield" if delta > 0 else "Baseline"


# ---------------------------------------------------------------------------
# Experiment bookkeeping
# ---------------------------------------------------------------------------


def _record(
    results: dict[str, list[dict[str, Any]]],
    exp: str,
    index: int,
    value: Any,
    pair: dict[str, Any],
    **extra: Any,
) -> None:
    """Append one seed's paired evaluation to point ``index`` of experiment ``exp``."""
    points = results.setdefault(exp, [])
    if len(points) <= index:
        points.append({"value": value, "baseline": [], "hopfield": [], "mcnemar_head": [],
                       "mcnemar_proto": [], **{k: [] for k in extra}})
    point = points[index]
    for key in ("baseline", "hopfield", "mcnemar_head", "mcnemar_proto"):
        point[key].append(pair[key])
    for key, val in extra.items():
        point[key].append(val)


def compare_outputs(
    run: BaselineRun,
    base_out: dict[str, torch.Tensor],
    hop_out: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Metrics for both models plus McNemar on head and prototype correctness."""
    bm, bc, bconf = summarize_outputs(base_out, labels, run.prototypes, run.memory_labels)
    hm, hc, hconf = summarize_outputs(hop_out, labels, run.prototypes, run.memory_labels)
    return {
        "baseline": bm,
        "hopfield": hm,
        "mcnemar_head": mcnemar_test(bc["head"], hc["head"]),
        "mcnemar_proto": mcnemar_test(bc["proto"], hc["proto"]),
        "confusion_baseline": bconf,
        "confusion_hopfield": hconf,
    }


def compare_on_test(run: BaselineRun, hop_net: MnistNet, kind: str, level: float) -> dict[str, Any]:
    """Paired Baseline-vs-Hopfield evaluation on one (kind, level) test set."""
    z = run.test_embeddings(kind, level)
    base_out = forward_embeddings(run.net, z, batch_size=run.batch_size, device=run.device)
    hop_out = forward_embeddings(hop_net, z, batch_size=run.batch_size, device=run.device)
    return compare_outputs(run, base_out, hop_out, run.splits.test_labels)


def memory_variants(configured: tuple[int, bool]) -> list[tuple[str, int, bool]]:
    """E9 memory designs: class means raw, class means centered, 8/class centered,
    plus the configured design if it is none of these."""
    designs = [(1, False), (1, True), (8, True)]
    if configured not in designs:
        designs.append(configured)
    return [(f"k={k}{' centered' if c else ' raw'}", k, c) for k, c in designs]


def build_embedding_net(run: BaselineRun, per_class: int, center: bool, *, beta: float) -> MnistNet:
    """Hopfield net whose L8 memory is built from ``run``'s training embeddings."""
    memory, labels, offset = build_memory(
        run.z_train, run.splits.train_labels, per_class=per_class, center=center, seed=run.seed
    )
    net = build_hopfield_net(run.net, memory, labels, beta=beta, center=offset)
    assert isinstance(net, MnistNet) and isinstance(net.bridge, HopfieldBridge)
    return net


def selection_label(sets: Sequence[tuple[str, float]]) -> str:
    """Human-readable description of the beta-selection validation sets."""
    parts = ["clean" if kind == "clean" else f"{kind} {level:g}" for kind, level in sets]
    return "clean val" if parts == ["clean"] else "mean of val " + " + ".join(parts)


def beta_curve(
    net: nn.Module,
    run: BaselineRun,
    betas: list[float],
    sets: Sequence[tuple[str, float]],
    *,
    pixel: bool = False,
) -> list[float]:
    """Validation score per beta: mean head_acc over the selection sets."""
    bridge = net.input_bridge if pixel else net.bridge
    assert isinstance(bridge, HopfieldBridge)
    labels = run.splits.val_labels
    curve = []
    for beta in betas:
        bridge.set_retrieval(beta=beta)
        accs = []
        for kind, level in sets:
            if pixel:
                out = forward_images(net, run.val_images(kind, level),
                                     batch_size=run.batch_size, device=run.device)
            else:
                assert isinstance(net, MnistNet)
                out = forward_embeddings(net, run.val_embeddings(kind, level),
                                         batch_size=run.batch_size, device=run.device)
            accs.append((out["head_pred"] == labels).float().mean().item())
        curve.append(sum(accs) / len(accs))
    return curve


def select_beta(betas: list[float], curves: list[list[float]]) -> tuple[float, list[float]]:
    """argmax over betas of the seed-mean validation accuracy; ties -> smallest beta."""
    mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(len(betas))]
    best = max(mean_curve)
    return betas[mean_curve.index(best)], mean_curve


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """Run E0-E8 for every configured seed and return all raw results."""
    threads = configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    env = environment_info(device)
    print(f"Device: {device}  threads: {threads}  torch {torch.__version__}")
    data = load_mnist(cfg["data_dir"])
    betas = cfg["beta_values"]

    # Stage 0: split sizes (logged first), then tune the baseline before any
    # backbone is trained; the winning lr / batch / epochs are written into cfg.
    splits_info = split_table(cfg, data)
    print("Data splits\n" + "\n".join(format_split_table(splits_info)))
    tuning = tune_baseline(cfg, data, device, retrain=retrain)
    if tuning is not None:
        print("Baseline tuning\n" + "\n".join(format_tuning_table(tuning)))

    # Stage 1: frozen baselines, memories, Hopfield nets, validation curves.
    sel_sets = cfg["beta_selection_sets"]
    configured = (cfg["prototypes_per_class"], cfg["center_prototypes"])
    variants = memory_variants(configured)
    runs: list[BaselineRun] = []
    hop_nets: list[MnistNet] = []
    variant_nets: dict[str, list[MnistNet]] = {name: [] for name, _, _ in variants}
    pixel_nets: list[PixelHopfieldNet | None] = []
    setup_times: list[float] = []
    val_curves: dict[str, list[list[float]]] = {"embedding": [], "pixel": []}
    variant_curves: dict[str, list[list[float]]] = {name: [] for name, _, _ in variants}
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        start = time.perf_counter()
        hop = build_embedding_net(run, *configured, beta=betas[0])
        setup_times.append(run.prototype_time_s + time.perf_counter() - start)
        val_curves["embedding"].append(beta_curve(hop, run, betas, sel_sets))

        for name, per_class, center in variants:
            net = hop if (per_class, center) == configured else build_embedding_net(
                run, per_class, center, beta=betas[0]
            )
            variant_nets[name].append(net)
            variant_curves[name].append(
                val_curves["embedding"][-1] if net is hop
                else beta_curve(net, run, betas, sel_sets)
            )

        pixel_net = None
        if "pixel" in cfg["placements"]:
            flat = run.splits.train_images.flatten(1)
            pixel_protos, pixel_labels = compute_prototypes(flat, run.splits.train_labels)
            pixel_net = build_hopfield_net(
                run.net, pixel_protos, pixel_labels, beta=betas[0],
                placement="pixel", center=flat.mean(dim=0),
            )
            assert isinstance(pixel_net, PixelHopfieldNet)
            val_curves["pixel"].append(beta_curve(pixel_net, run, betas, sel_sets, pixel=True))
        runs.append(run)
        hop_nets.append(hop)
        pixel_nets.append(pixel_net)

    beta_star, mean_val_curve = select_beta(betas, val_curves["embedding"])
    beta_pixel = select_beta(betas, val_curves["pixel"])[0] if val_curves["pixel"] else None
    variant_betas = {name: select_beta(betas, c)[0] for name, c in variant_curves.items()}
    print(f"beta* = {beta_star:g} (validation head_acc on {selection_label(sel_sets)},"
          " ties -> smallest beta)")

    # Stage 2: paired test-set experiments.
    results: dict[str, list[dict[str, Any]]] = {}
    confusions: dict[str, torch.Tensor] = {}
    sanity_raw: dict[str, list[Any]] = {"s1": [], "s2": [], "s3": []}
    grids = noise_grids(cfg)
    sigma = cfg["noisy_sigma"]
    for run, hop, pixel_net in zip(runs, hop_nets, pixel_nets):
        assert isinstance(hop.bridge, HopfieldBridge)
        bridge = hop.bridge
        print(f"  seed {run.seed}: evaluating")
        labels = run.splits.test_labels

        def keep_confusion(name: str, pair: dict[str, Any], model: str) -> None:
            conf = pair[f"confusion_{model}"]
            confusions[name] = confusions.get(name, torch.zeros_like(conf)) + conf

        bridge.set_retrieval(beta=beta_star, num_updates=1)
        pair = compare_on_test(run, hop, "gaussian", 0.0)
        _record(results, "E1", 0, "clean", pair)
        keep_confusion("hopfield_clean", pair, "hopfield")
        keep_confusion("baseline_clean", pair, "baseline")

        for kind in NOISE_TYPES:
            for i, level in enumerate(grids[kind]):
                pair = compare_on_test(run, hop, kind, level)
                _record(results, E3_IDS[kind], i, level, pair)

        pair = compare_on_test(run, hop, "gaussian", sigma)
        keep_confusion("hopfield_noisy", pair, "hopfield")
        keep_confusion("baseline_noisy", pair, "baseline")

        for i, beta in enumerate(betas):
            bridge.set_retrieval(beta=beta, num_updates=1)
            _record(results, "E2", i, beta, compare_on_test(run, hop, "gaussian", 0.0))
            _record(results, "E5", i, beta, compare_on_test(run, hop, "gaussian", sigma))

        for i, steps in enumerate(cfg["num_updates_values"]):
            bridge.set_retrieval(beta=beta_star, num_updates=steps)
            _record(results, "E4", i, steps, compare_on_test(run, hop, "gaussian", sigma))

        bridge.set_retrieval(beta=beta_star, num_updates=1)
        for i, placement in enumerate(cfg["placements"]):
            if placement == "embedding":
                pair = compare_on_test(run, hop, "gaussian", 0.0)
                _record(results, "E6", i, placement, pair, beta=beta_star)
            else:
                assert pixel_net is not None and beta_pixel is not None
                pixel_net.input_bridge.set_retrieval(beta=beta_pixel)
                base_out = forward_embeddings(
                    run.net, run.test_embeddings("gaussian", 0.0),
                    batch_size=run.batch_size, device=run.device,
                )
                pix_out = forward_images(
                    pixel_net, run.splits.test_images, batch_size=run.batch_size, device=run.device
                )
                pair = compare_outputs(run, base_out, pix_out, labels)
                _record(results, "E6", i, placement, pair, beta=beta_pixel)

        # E9: memory design, each variant at its own validation-selected beta.
        seed_idx = runs.index(run)
        for i, (name, _, _) in enumerate(variants):
            vnet = variant_nets[name][seed_idx]
            assert isinstance(vnet.bridge, HopfieldBridge)
            vnet.bridge.set_retrieval(beta=variant_betas[name], num_updates=1)
            _record(results, "E9a", i, name, compare_on_test(run, vnet, "gaussian", 0.0),
                    beta=variant_betas[name])
            _record(results, "E9b", i, name, compare_on_test(run, vnet, "gaussian", sigma),
                    beta=variant_betas[name])
        bridge.set_retrieval(beta=beta_star, num_updates=1)

        # S1: at very large beta the softmax readout must equal the nearest
        # stored pattern under the same (centered) cosine scoring.
        z_clean = run.test_embeddings("gaussian", 0.0)
        bridge.set_retrieval(beta=cfg["s1_beta"])
        hop_out = forward_embeddings(hop, z_clean, batch_size=run.batch_size, device=run.device)
        memory = bridge.prototypes.cpu()
        offset = bridge.center.cpu() if bridge.center is not None else torch.zeros(memory.shape[1])
        nearest = bridge.memory_labels.cpu()[
            nearest_prototype(z_clean - offset, memory - offset)
        ]
        sanity_raw["s1"].append((hop_out["proto_pred"] == nearest).float().mean().item())
        bridge.set_retrieval(beta=beta_star)

        # S2: identical shared tensors, and identical z from both full networks.
        with torch.inference_mode():
            x = run.splits.test_images[:256].to(run.device)
            same_z = torch.equal(run.net(x)[1], hop(x)[1])
        sanity_raw["s2"].append(shared_weights_equal(run.net, hop) and same_z)

        # S3: the head maps every prototype to its own class.
        with torch.inference_mode():
            head_pred = run.net.head(run.prototypes.to(run.device)).argmax(dim=-1).cpu()
        sanity_raw["s3"].append(int((head_pred == run.memory_labels).sum().item()))

    cost = measure_all_costs(
        cfg, runs[0], hop_nets[0], beta_star, device,
        train_times=[r.train_time_s for r in runs], setup_times=setup_times,
    )
    return {
        "config": cfg,
        "environment": env,
        "splits": splits_info,
        "baseline_tuning": tuning,
        "beta_star": beta_star,
        "beta_pixel": beta_pixel,
        "variant_betas": variant_betas,
        "beta_selection": selection_label(sel_sets),
        "val_curves": val_curves,
        "mean_val_curve": mean_val_curve,
        "n_test": int(runs[0].splits.test_labels.shape[0]),
        "results": results,
        "confusions": {k: v.tolist() for k, v in confusions.items()},
        "sanity_raw": sanity_raw,
        "cost": cost,
    }


def measure_all_costs(
    cfg: dict[str, Any],
    run: BaselineRun,
    hop: MnistNet,
    beta_star: float,
    device: torch.device,
    *,
    train_times: list[float],
    setup_times: list[float],
) -> dict[str, Any]:
    """E7 (batch 256), E8 (batch sweep) and E4 latency per T, on the first seed's nets."""
    c = cfg["cost"]
    assert isinstance(hop.bridge, HopfieldBridge)
    hop.bridge.set_retrieval(beta=beta_star, num_updates=1)
    models: dict[str, nn.Module] = {"baseline": run.net, "hopfield": hop}
    timing = {"repeats": c["repeats"], "warmup": c["warmup"], "iters": c["iters"]}

    print(f"  cost: E7 at batch {c['batch_size']} ({c['repeats']} interleaved repeats)")
    e7 = interleaved_cost(models, c["batch_size"], device, **timing)

    eval_times: dict[str, list[float]] = {name: [] for name in models}
    names = list(models)
    for r in range(c["repeats"]):
        for name in names if r % 2 == 0 else names[::-1]:
            eval_times[name].append(time_full_evaluation(
                models[name], run.splits.test_images, batch_size=c["batch_size"], device=device
            ))

    print(f"  cost: E8 over batch sizes {c['batch_sizes']}")
    e8 = {bs: interleaved_cost(models, bs, device, **timing) for bs in c["batch_sizes"]}

    print(f"  cost: E4 latency for T in {cfg['num_updates_values']}")
    t_models: dict[str, nn.Module] = {"baseline": run.net}
    for steps in cfg["num_updates_values"]:
        net = build_hopfield_net(
            run.net, run.prototypes, run.memory_labels, beta=beta_star, num_updates=steps
        )
        t_models[f"T={steps}"] = net
    e4 = interleaved_cost(t_models, c["batch_size"], device, **timing)

    static: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        assert isinstance(model, MnistNet)
        breakdown = analytical_macs(model)
        static[name] = {
            "trainable_params": count_trainable_params(model),
            "buffer_values": count_buffer_values(model),
            "frozen_memory_values": count_buffer_values(model, floating_only=True),
            "model_size_bytes": model_size_bytes(model),
            "macs_breakdown": breakdown,
            "macs_per_image": sum(m for _, m in breakdown),
            "hook_macs_per_image": hook_macs(model),
        }
    return {
        "num_threads": torch.get_num_threads(),
        "device": str(device),
        "e7": e7,
        "eval_time_s": {name: median_iqr(v) for name, v in eval_times.items()},
        "e8": {str(bs): v for bs, v in e8.items()},
        "e4_latency": e4,
        "static": static,
        "bridge_trainable_params": count_trainable_params(hop.bridge),
        "train_time_s": mean_std(train_times),
        "setup_time_s": mean_std(setup_times),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_point(point: dict[str, Any], metric: str) -> dict[str, Any]:
    """Mean ± std per model, Δ (H−B), max-over-seeds McNemar p, and winner."""
    out: dict[str, Any] = {"value": point["value"]}
    for model in MODELS:
        vals = [m[metric] for m in point[model]]
        if any(v is None for v in vals):
            out[model] = None
        else:
            out[model] = mean_std(vals)
    if out["baseline"] is None or out["hopfield"] is None:
        out["delta"] = out["p"] = out["winner"] = None
        return out
    out["delta"] = out["hopfield"][0] - out["baseline"][0]
    out.update(seed_stats([h[metric] - b[metric] for h, b in zip(point["hopfield"], point["baseline"])]))
    test_key = {"head_acc": "mcnemar_head", "proto_acc": "mcnemar_proto"}.get(metric)
    if test_key is None:
        out["p"] = out["winner"] = None
    else:
        out["p"] = max(t["p"] for t in point[test_key])
        signs = [(t["c"] > t["b"]) - (t["c"] < t["b"]) for t in point[test_key]]
        out["winner"] = decide_winner(out["delta"], out["p"], signs, out["wilcoxon_p"])
    return out


def seed_stats(seed_deltas: Sequence[float]) -> dict[str, Any]:
    """Per-seed direction counts and (with >= 6 seeds) the exact Wilcoxon p."""
    eps = 1e-12
    return {
        "seed_deltas": list(seed_deltas),
        "n_pos": sum(d > eps for d in seed_deltas),
        "n_neg": sum(d < -eps for d in seed_deltas),
        "wilcoxon_p": (
            wilcoxon_signed_rank_p([0.0 if abs(d) <= eps else d for d in seed_deltas])
            if len(seed_deltas) >= MIN_SEEDS_FOR_WILCOXON else None
        ),
    }


def auc_summary(points: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """robust_auc of ``metric`` per seed and model; mean ± std and Δ (H−B)."""
    levels = [float(p["value"]) for p in points]
    num_seeds = len(points[0]["baseline"])
    out: dict[str, Any] = {}
    per_seed: dict[str, list[float]] = {}
    for model in MODELS:
        per_seed[model] = [
            robust_auc(levels, [p[model][s][metric] for p in points]) for s in range(num_seeds)
        ]
        out[model] = mean_std(per_seed[model])
    out["delta"] = out["hopfield"][0] - out["baseline"][0]
    out.update(seed_stats([h - b for h, b in zip(per_seed["hopfield"], per_seed["baseline"])]))
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _ms(stat: tuple[float, float] | None, digits: int = 4) -> str:
    return "—" if stat is None else f"{stat[0]:.{digits}f} ± {stat[1]:.{digits}f}"


def _signed(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−")


def _p(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def _fmt_value(value: Any, fmt: str) -> str:
    if isinstance(value, str):
        return value
    return format(value, fmt)


def table_header(exp: str, title: str, varied: str, held: str, context: str) -> list[str]:
    """Title, ``Varied:`` / ``Held fixed:`` line and context, framed by rules."""
    return [RULE, f" {exp}  {title}", f"     Varied: {varied}     Held fixed: {held}",
            f"     {context}", RULE]


def metric_block(
    points: list[dict[str, Any]],
    metric: str,
    varied_col: str,
    value_fmt: str,
    *,
    summary: tuple[str, dict[str, Any]] | None = None,
) -> list[str]:
    """``<varied> | Baseline | Hopfield | Δ | McNemar p | Seeds +/− | Wilcoxon p | Winner``."""
    rule = f" {'─' * 13}┼{'─' * 26}┼{'─' * 26}┼{'─' * 11}┼{'─' * 11}┼{'─' * 11}┼{'─' * 12}┼{'─' * 9}"
    lines = [
        (f"  {varied_col:>11} │ {'Baseline ' + metric:>24} │ {'Hopfield ' + metric:>24} │"
         f" {'Δ (H−B)':>9} │ {'McNemar p':>9} │ {'Seeds +/−':>9} │ {'Wilcoxon p':>10} │ Winner"),
        rule,
    ]

    def seeds(agg: dict[str, Any]) -> str:
        return f"{agg['n_pos']}/{agg['n_neg']}"

    for point in points:
        agg = aggregate_point(point, metric)
        lines.append(
            f"  {_fmt_value(agg['value'], value_fmt):>11} │ {_ms(agg['baseline']):>24} │"
            f" {_ms(agg['hopfield']):>24} │ {_signed(agg['delta']):>9} │ {_p(agg['p']):>9} │"
            f" {seeds(agg):>9} │ {_p(agg['wilcoxon_p']):>10} │ {agg['winner'] or '—'}"
        )
    if summary is not None:
        label, s = summary
        lines.append(rule)
        lines.append(
            f"  {label:>11} │ {_ms(s['baseline']):>24} │ {_ms(s['hopfield']):>24} │"
            f" {_signed(s['delta']):>9} │ {'':>9} │ {seeds(s):>9} │ {_p(s['wilcoxon_p']):>10} │"
        )
    return lines


def diagnostics_block(points: list[dict[str, Any]], varied_col: str, value_fmt: str) -> list[str]:
    """Hopfield-only retrieval diagnostics (baseline makes a hard choice: '—')."""
    lines = [
        (f"  {varied_col:>9} │ {'entropy_norm':>17} │ {'top1_weight':>17} │ {'true_proto_weight':>17} │"
        f" {'dist_to_true_proto B / H':>26}"),
        f" {'─' * 11}┼{'─' * 19}┼{'─' * 19}┼{'─' * 19}┼{'─' * 28}",
    ]
    for point in points:
        ent = aggregate_point(point, "entropy_norm")["hopfield"]
        top = aggregate_point(point, "top1_weight")["hopfield"]
        tru = aggregate_point(point, "true_proto_weight")["hopfield"]
        dist = aggregate_point(point, "dist_to_true_proto")
        lines.append(
            f"  {_fmt_value(point['value'], value_fmt):>9} │ {_ms(ent, 3):>17} │ {_ms(top, 3):>17} │"
            f" {_ms(tru, 3):>17} │ {dist['baseline'][0]:>11.3f} / {dist['hopfield'][0]:<11.3f}"
        )
    lines.append("  Baseline: entropy_norm / top1_weight / true_proto_weight = — (hard nearest-prototype choice)")
    return lines


def format_e7(raw: dict[str, Any], e3a_auc_delta: float) -> tuple[list[str], dict[str, Any]]:
    """E7 cost table: Baseline | Hopfield | Δ | Overhead % plus the MAC breakdown."""
    cost = raw["cost"]
    cfg = raw["config"]
    sb, sh = cost["static"]["baseline"], cost["static"]["hopfield"]
    eb, eh = cost["e7"]["baseline"], cost["e7"]["hopfield"]

    def pct(b: float, h: float) -> str:
        return "—" if b == 0 else _signed((h - b) / b * 100.0, 2) + " %"

    def med(stats: dict[str, Any], key: str) -> float:
        return float(stats[key]["median"])

    lat_b, lat_h = med(eb, "latency_ms_p50"), med(eh, "latency_ms_p50")
    overhead = (lat_h - lat_b) / lat_b * 100.0
    train_mean = cost["train_time_s"][0]
    setup_mean = cost["setup_time_s"][0]
    ev_b, ev_h = cost["eval_time_s"]["baseline"], cost["eval_time_s"]["hopfield"]
    head = table_header(
        "E7", "Cost of the Hopfield layer", "model",
        f"batch={cfg['cost']['batch_size']}, device={cost['device']}, threads={cost['num_threads']},"
        f" β=β*={raw['beta_star']:g}, T=1, clean",
        f"{cfg['cost']['repeats']} interleaved repeats × {cfg['cost']['iters']} timed batches"
        f" (+{cfg['cost']['warmup']} warm-up), median ± IQR",
    )
    sep = f" {'─' * 26}┼{'─' * 24}┼{'─' * 24}┼{'─' * 14}┼{'─' * 11}"
    rows = [f"  {'Metric':<24} │ {'Baseline':>22} │ {'Hopfield':>22} │ {'Δ (H−B)':>12} │ Overhead", sep]

    def row(name: str, b: str, h: str, d: str, o: str) -> None:
        rows.append(f"  {name:<24} │ {b:>22} │ {h:>22} │ {d:>12} │ {o}")

    def mi(stats: dict[str, Any], key: str, digits: int) -> str:
        return f"{stats[key]['median']:.{digits}f} ± {stats[key]['iqr']:.{digits}f}"

    row("Trainable params", f"{sb['trainable_params']:,}", f"{sh['trainable_params']:,}",
        f"{sh['trainable_params'] - sb['trainable_params']:+,}",
        pct(sb["trainable_params"], sh["trainable_params"]))
    row("Frozen memory values", f"{sb['frozen_memory_values']:,}",
        f"{sh['frozen_memory_values']:,} (+{sh['buffer_values'] - sh['frozen_memory_values']} labels)",
        f"{sh['frozen_memory_values'] - sb['frozen_memory_values']:+,}", "—")
    row("Model size (bytes)", f"{sb['model_size_bytes']:,}", f"{sh['model_size_bytes']:,}",
        f"{sh['model_size_bytes'] - sb['model_size_bytes']:+,}",
        pct(sb["model_size_bytes"], sh["model_size_bytes"]))
    row("MACs / image", f"{sb['macs_per_image']:,}", f"{sh['macs_per_image']:,}",
        f"{sh['macs_per_image'] - sb['macs_per_image']:+,}",
        pct(sb["macs_per_image"], sh["macs_per_image"]))
    row("Latency p50 (ms)", mi(eb, "latency_ms_p50", 3), mi(eh, "latency_ms_p50", 3),
        _signed(lat_h - lat_b, 3), pct(lat_b, lat_h))
    row("Latency p95 (ms)", mi(eb, "latency_ms_p95", 3), mi(eh, "latency_ms_p95", 3),
        _signed(med(eh, "latency_ms_p95") - med(eb, "latency_ms_p95"), 3),
        pct(med(eb, "latency_ms_p95"), med(eh, "latency_ms_p95")))
    row("Latency / image (µs)", mi(eb, "latency_per_image_us", 2), mi(eh, "latency_per_image_us", 2),
        _signed(med(eh, "latency_per_image_us") - med(eb, "latency_per_image_us"), 2),
        pct(med(eb, "latency_per_image_us"), med(eh, "latency_per_image_us")))
    row("Throughput (img/s)", mi(eb, "throughput_img_s", 0), mi(eh, "throughput_img_s", 0),
        _signed(med(eh, "throughput_img_s") - med(eb, "throughput_img_s"), 0),
        pct(med(eb, "throughput_img_s"), med(eh, "throughput_img_s")))
    row("Peak memory (MB)", mi(eb, "peak_mem_mb", 3), mi(eh, "peak_mem_mb", 3),
        _signed(med(eh, "peak_mem_mb") - med(eb, "peak_mem_mb"), 3),
        pct(med(eb, "peak_mem_mb"), med(eh, "peak_mem_mb")))
    row("Train time (s)", f"{train_mean:.1f}", "0 (reused)", "", "")
    row("Setup time (s)", "—", f"{setup_mean:.2f}", f"+{setup_mean:.2f}", "")
    row(f"Eval time, {raw['n_test']} imgs (s)", f"{ev_b['median']:.3f} ± {ev_b['iqr']:.3f}",
        f"{ev_h['median']:.3f} ± {ev_h['iqr']:.3f}", _signed(ev_h["median"] - ev_b["median"], 3),
        pct(ev_b["median"], ev_h["median"]))
    rows.append(sep)

    if overhead > 0:
        per_cost = e3a_auc_delta * 100.0 / overhead
        acc_line = (f"  Accuracy per cost: Δ robust_auc(E3a) per +1 % latency = {per_cost:+.4f} pp"
                    f"  (Δ AUC {e3a_auc_delta * 100:+.2f} pp / {overhead:+.2f} % latency)")
    else:
        per_cost = None
        acc_line = (f"  Accuracy per cost: n/a, no measurable latency overhead"
                    f" ({overhead:+.2f} %); Δ robust_auc(E3a) = {e3a_auc_delta * 100:+.2f} pp")
    rows.append(acc_line)
    rows.append(f"  Peak memory source: {eb['peak_mem_source']}")
    rows.append("  MACs per image (analytical; conv H·W·Cout·k²·Cin, linear in·out,"
                " Hopfield per step 2·K·D + D):")
    for name, macs in sh["macs_breakdown"]:
        base = dict(sb["macs_breakdown"]).get(name)
        rows.append(f"    {name:<10} baseline {base if base is not None else 0:>11,}   hopfield {macs:>11,}")
    rows.append(f"    {'total':<10} baseline {sb['macs_per_image']:>11,}   hopfield {sh['macs_per_image']:>11,}")
    return head + rows, {"latency_overhead_pct": overhead, "accuracy_per_cost": per_cost}


def format_e8(raw: dict[str, Any]) -> list[str]:
    """E8: latency per image and throughput across batch sizes."""
    cost = raw["cost"]
    head = table_header(
        "E8", "Cost scaling with batch size", "batch size",
        f"β=β*={raw['beta_star']:g}, T=1, clean, device={cost['device']}, threads={cost['num_threads']}",
        f"{raw['config']['cost']['repeats']} interleaved repeats, median",
    )
    lines = [
        (f"  {'batch':>6} │ {'Baseline µs/img':>16} │ {'Hopfield µs/img':>16} │ {'Δ (H−B)':>9} │"
        f" {'Overhead':>9} │ {'B img/s':>9} │ {'H img/s':>9}"),
        f" {'─' * 8}┼{'─' * 18}┼{'─' * 18}┼{'─' * 11}┼{'─' * 11}┼{'─' * 11}┼{'─' * 10}",
    ]
    for bs, entry in cost["e8"].items():
        b = entry["baseline"]["latency_per_image_us"]["median"]
        h = entry["hopfield"]["latency_per_image_us"]["median"]
        lines.append(
            f"  {bs:>6} │ {b:>16.2f} │ {h:>16.2f} │ {_signed(h - b, 2):>9} │"
            f" {_signed((h - b) / b * 100, 1) + ' %':>9} │"
            f" {entry['baseline']['throughput_img_s']['median']:>9.0f} │"
            f" {entry['hopfield']['throughput_img_s']['median']:>9.0f}"
        )
    return head + lines


def build_report(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render E1-E9, the sanity report and the verdict; return (text, summary dict)."""
    cfg = raw["config"]
    res = raw["results"]
    bstar = raw["beta_star"]
    n_seeds = len(cfg["seeds"])
    k = cfg["prototypes_per_class"]
    centered = cfg["center_prototypes"]
    memory_desc = f"memory={10 * k} prototypes ({k}/class{', centered' if centered else ''})"
    ctx = f"seeds={n_seeds}  n_test={raw['n_test']}  {memory_desc}  placement=L8"
    selection = raw.get("beta_selection", "clean val")
    sigma = cfg["noisy_sigma"]
    sections: list[tuple[str, list[str]]] = []
    summary: dict[str, Any] = {"beta_star": bstar}

    lines = [" E0  Data splits   Varied: —   Held fixed: split sizes are identical for every seed",
             f"     {ctx}", *format_split_table(raw["splits"])]
    sections.append(("E0 Data splits", lines))
    if raw.get("baseline_tuning") is not None:
        lines = [(" E0b Baseline tuning   Varied: lr, batch size, epochs   Held fixed: architecture,"
                  " Adam, clean training, validation data only"),
                 *format_tuning_table(raw["baseline_tuning"])]
        sections.append(("E0b Baseline tuning", lines))
        summary["baseline_tuning"] = raw["baseline_tuning"]["best"]

    e1 = res["E1"]
    lines = table_header("E1", "Clean performance: does Hopfield cost accuracy?", "model",
                         f"β=β*={bstar:g}, T=1, clean", ctx)
    lines += metric_block(e1, "head_acc", "data", "s")
    lines += [""] + metric_block(e1, "proto_acc", "data", "s")
    lines += [""] + diagnostics_block(e1, "data", "s")
    if n_seeds >= MIN_SEEDS_FOR_WILCOXON:
        lines.append("  Winner rule: |Δ| < 0.005 or Wilcoxon p ≥ 0.05 → tie (exact signed-rank test"
                     " on per-seed Δ: is the effect reproducible across trained networks?).")
        lines.append("  McNemar p = max over seeds (per-image paired test within each seed).")
    else:
        lines.append("  Winner rule: |Δ| < 0.005 or p ≥ 0.05 → tie; seeds disagreeing in direction"
                     " → mixed. McNemar p = max over seeds (exact binomial).")
    if k == 1 and not centered:
        lines.append("  Note: at T=1 Hopfield proto_acc equals nearest-prototype proto_acc by"
                     " construction (softmax keeps the argmax of cos(z, M)); only head_acc,"
                     " dist and T>1 can differ.")
    else:
        lines.append("  Note: Baseline proto_acc is the nearest class mean (cosine on raw z);"
                     " Hopfield proto_acc is the class with the most retrieval weight.")
    sections.append(("E1 Clean performance", lines))
    summary["E1"] = aggregate_point(e1[0], "head_acc")

    e2 = res["E2"]
    lines = table_header("E2", "How does β shape retrieval?", "β", "clean, T=1",
                         ctx + "  (Baseline column = reference, independent of β)")
    lines += metric_block(e2, "head_acc", "β", "g")
    lines += [""] + metric_block(e2, "proto_acc", "β", "g")
    lines += [""] + diagnostics_block(e2, "β", "g")
    val = ", ".join(f"{b:g}:{v:.4f}" for b, v in zip(cfg["beta_values"], raw["mean_val_curve"]))
    lines.append(f"  Selection score ({selection}, seed mean): {val}  → β*={bstar:g}")
    sections.append(("E2 β sweep (clean)", lines))

    auc_deltas: dict[str, float] = {}
    for kind in NOISE_TYPES:
        exp = E3_IDS[kind]
        pts = res[exp]
        sym = NOISE_LABELS[kind]
        fmt = "d" if kind == "occlusion" else ".2f"
        name = {"gaussian": "Gaussian noise", "salt_pepper": "salt-and-pepper noise",
                "occlusion": "occlusion (square side, px)"}[kind]
        lines = table_header(exp, f"Robustness to {name}", sym, f"β=β*={bstar:g}, T=1", ctx)
        head_auc = auc_summary(pts, "head_acc")
        proto_auc = auc_summary(pts, "proto_acc")
        lines += metric_block(pts, "head_acc", sym, fmt, summary=("AUC", head_auc))
        lines += [""] + metric_block(pts, "proto_acc", sym, fmt, summary=("AUC", proto_auc))
        lines += [""] + diagnostics_block(pts, sym, fmt)
        drops = []
        for model in MODELS:
            clean = aggregate_point(pts[0], "head_acc")[model][0]
            worst = aggregate_point(pts[-1], "head_acc")[model][0]
            drops.append(f"{model} {clean - worst:.4f}")
        lines.append(f"  acc_drop (clean − {sym}={_fmt_value(pts[-1]['value'], fmt)}): " + ", ".join(drops))
        sections.append((f"{exp} {name}", lines))
        auc_deltas[kind] = head_auc["delta"]
        summary[exp] = {"head_auc": head_auc, "proto_auc": proto_auc}

    e4 = res["E4"]
    lines = table_header("E4", "Do extra attractor steps help a noisy query?", "T",
                         f"β=β*={bstar:g}, Gaussian σ={sigma:g}",
                         ctx + "  (Baseline column = reference, independent of T)")
    lines += metric_block(e4, "head_acc", "T", "d")
    lines += [""] + metric_block(e4, "proto_acc", "T", "d")
    lines += [""] + diagnostics_block(e4, "T", "d")
    lat = raw["cost"]["e4_latency"]
    lines.append(f"  latency_ms p50 at batch {cfg['cost']['batch_size']}: baseline"
                 f" {lat['baseline']['latency_ms_p50']['median']:.3f}; " + ", ".join(
                     f"T={t} {lat[f'T={t}']['latency_ms_p50']['median']:.3f}"
                     for t in cfg["num_updates_values"]))
    sections.append(("E4 attractor steps", lines))

    e5 = res["E5"]
    lines = table_header("E5", "Does β matter more under noise?", "β",
                         f"Gaussian σ={sigma:g}, T=1",
                         ctx + "  (Baseline column = reference, independent of β)")
    lines += metric_block(e5, "head_acc", "β", "g")
    lines += [""] + metric_block(e5, "proto_acc", "β", "g")
    lines += [""] + diagnostics_block(e5, "β", "g")
    sections.append(("E5 β sweep (σ fixed)", lines))

    e6 = res["E6"]
    tuned = ", ".join(f"{p['value']} β={p['beta'][0]:g}" for p in e6)
    lines = table_header("E6", "Placement ablation: pixel vs embedding", "placement",
                         f"β tuned per placement on val ({tuned}), clean, T=1",
                         f"seeds={n_seeds}  n_test={raw['n_test']}  pixel memory=10 class means,"
                         f" embedding {memory_desc}  (Baseline column = reference)")
    lines += metric_block(e6, "head_acc", "placement", "s")
    lines += [""] + metric_block(e6, "proto_acc", "placement", "s")
    sections.append(("E6 placement ablation", lines))

    if "E9a" in res:
        betas_v = ", ".join(f"{name}: β={b:g}" for name, b in raw["variant_betas"].items())
        for exp, cond, label in (("E9a", "clean", "clean"),
                                 ("E9b", f"Gaussian σ={sigma:g}", f"σ={sigma:g}")):
            lines = table_header(exp, f"Memory design ablation ({label})", "memory design",
                                 f"{cond}, T=1, β selected per design on {selection} ({betas_v})",
                                 f"seeds={n_seeds}  n_test={raw['n_test']}  placement=L8"
                                 "  (Baseline column = reference, no Hopfield layer)")
            lines += metric_block(res[exp], "head_acc", "design", "s")
            sections.append((f"{exp} memory design ({label})", lines))
            summary[exp] = [aggregate_point(p, "head_acc") for p in res[exp]]

    e7_lines, e7_summary = format_e7(raw, auc_deltas["gaussian"])
    sections.append(("E7 cost", e7_lines))
    sections.append(("E8 cost scaling", format_e8(raw)))
    summary["E7"] = e7_summary

    sanity = sanity_checks(raw)
    summary["sanity"] = sanity
    lines = [f" {'SANITY CHECKS':<64} {'expected':<16} {'measured':<22} status"]
    for s in sanity:
        lines.append(f" {s['id']:<4}{s['description']:<60} {s['expected']:<16} {s['measured']:<22} {s['status']}")
    verdict = build_verdict(summary, auc_deltas, raw)
    summary["verdict"] = verdict
    lines.append(f" {verdict}")
    sections.append(("Sanity checks and verdict", lines))

    text = "\n\n".join("\n".join(body) for _, body in sections)
    md = [(f"# Task 13: MNIST prototype retrieval, Baseline vs Hopfield"
          f"{' (quick mode)' if cfg.get('quick') else ''}"), "",
          (f"β* = {bstar:g} (validation head_acc on {selection}). Device {raw['environment']['device']},"
          f" {raw['environment']['cpu_model']}, torch {raw['environment']['torch_version']},"
          f" threads {raw['cost']['num_threads']}."), ""]
    for heading, body in sections:
        md += [f"## {heading}", "", "```", *body, "```", ""]
    summary["markdown"] = "\n".join(md)
    return text, summary


# ---------------------------------------------------------------------------
# Sanity checks and verdict
# ---------------------------------------------------------------------------


def _check(sid: str, description: str, expected: str, measured: str, ok: bool | None,
           *, warn: bool = False) -> dict[str, Any]:
    status = "SKIP" if ok is None else "PASS" if ok else "WARN" if warn else "FAIL"
    return {"id": sid, "description": description, "expected": expected,
            "measured": measured, "status": status}


def sanity_checks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """S1-S11 evaluated on the run's real results (PASS / FAIL / WARN / SKIP)."""
    res, sraw, cost = raw["results"], raw["sanity_raw"], raw["cost"]
    checks = []

    s1 = min(sraw["s1"])
    checks.append(_check("S1", f"Hopfield(β={raw['config']['s1_beta']:g}) proto preds == nearest stored pattern",
                         "identical", f"{s1 * 100:.2f} % agree", s1 == 1.0))
    checks.append(_check("S2", "Baseline vs Hopfield share L1–L7, L9 tensors (and z)", "identical",
                         "identical" if all(sraw["s2"]) else "differ", all(sraw["s2"])))
    s3 = min(sraw["s3"])
    checks.append(_check("S3", "head(M[c]) == c for all 10 prototypes", "10/10",
                         f"{s3}/10 (min over seeds)", s3 == 10))
    s4 = aggregate_point(res["E1"][0], "proto_acc")["hopfield"][0]
    checks.append(_check("S4", "Clean proto_acc (Hopfield, β*)", f"≥ {S4_MIN_PROTO_ACC}",
                         f"{s4:.4f}", s4 >= S4_MIN_PROTO_ACC))
    conf = torch.tensor(raw["confusions"]["hopfield_clean"])
    off = conf.clone()
    off.fill_diagonal_(-1)
    rows_ok = int((conf.diagonal() > off.max(dim=1).values).sum().item())
    checks.append(_check("S5", "Clean digit retrieves its own prototype most (confusion)",
                         "diag > row-max", f"{rows_ok}/{conf.shape[0]} rows", rows_ok == conf.shape[0]))
    ent = [aggregate_point(p, "entropy_norm")["hopfield"][0] for p in res["E2"]]
    mono = all(b < a for a, b in itertools.pairwise(ent))
    checks.append(_check("S6", "entropy_norm strictly decreases with β (E2)", "monotone",
                         f"{ent[0]:.3f} → {ent[-1]:.3f}", mono))
    worst_rise = -math.inf
    for kind in NOISE_TYPES:
        for model in MODELS:
            accs = [aggregate_point(p, "head_acc")[model][0] for p in res[E3_IDS[kind]]]
            worst_rise = max(worst_rise, *(b - a for a, b in itertools.pairwise(accs)))
    checks.append(_check("S7", "head_acc non-increasing with noise (E3a–c, both models)",
                         f"monotone ±{S7_TOLERANCE}", f"max rise {worst_rise:+.4f}",
                         worst_rise <= S7_TOLERANCE))
    e6 = {p["value"]: aggregate_point(p, "head_acc")["hopfield"][0] for p in res["E6"]}
    if {"pixel", "embedding"} <= set(e6):
        checks.append(_check("S8", "Embedding placement (L8) beats pixel placement (E6)", "L8 > L0",
                             f"{e6['embedding']:.4f} vs {e6['pixel']:.4f}",
                             e6["embedding"] > e6["pixel"]))
    else:
        checks.append(_check("S8", "Embedding placement (L8) beats pixel placement (E6)", "L8 > L0",
                             "placement not run", None))
    sb, sh = cost["static"]["baseline"], cost["static"]["hopfield"]
    s9 = sb["trainable_params"] == sh["trainable_params"] and cost["bridge_trainable_params"] == 0
    checks.append(_check("S9", "Trainable params equal; Hopfield adds 0 trainable params", "equal",
                         f"{sb['trainable_params']:,} vs {sh['trainable_params']:,}", s9))
    s10 = all(cost["static"][m]["macs_per_image"] == cost["static"][m]["hook_macs_per_image"]
              for m in MODELS)
    checks.append(_check("S10", "Analytical MACs == forward-hook count (both nets)", "exact",
                         f"{sh['macs_per_image']:,} / {sh['hook_macs_per_image']:,}", s10))
    eb, eh = cost["e7"]["baseline"], cost["e7"]["hopfield"]
    lat_b, lat_h = eb["latency_ms_p50"]["median"], eh["latency_ms_p50"]["median"]
    overhead = (lat_h - lat_b) / lat_b * 100.0
    checks.append(_check("S11", f"Hopfield latency overhead at batch {raw['config']['cost']['batch_size']} (E7)",
                         f"< {S11_MAX_OVERHEAD_PCT:g} %", f"{overhead:+.2f} %",
                         overhead < S11_MAX_OVERHEAD_PCT, warn=cost["device"] == "cpu"))
    return checks


def build_verdict(summary: dict[str, Any], auc_deltas: dict[str, float], raw: dict[str, Any]) -> str:
    """One line from E1 (clean winner), the E3 AUC deltas and the E7 overheads."""
    e1 = summary["E1"]
    clean = {"tie": "is neutral", "mixed": "is inconsistent across seeds",
             "Hopfield": "helps", "Baseline": "hurts"}[e1["winner"]]

    def word(kind: str, delta: float) -> str:
        auc = summary[E3_IDS[kind]]["head_auc"]
        counts = f"{auc['n_pos']}/{auc['n_neg']} seeds"
        if abs(delta) < WINNER_DELTA:
            return f"neutral, {counts}"
        if auc["wilcoxon_p"] is not None:
            if auc["wilcoxon_p"] >= ALPHA:
                return f"not significant, {counts}, Wilcoxon p={_p(auc['wilcoxon_p'])}"
            word_ = "helps" if delta > 0 else "hurts"
            return f"{word_}, {counts}, Wilcoxon p={_p(auc['wilcoxon_p'])}"
        if any((d > 0) != (delta > 0) for d in auc["seed_deltas"]):
            return f"inconsistent across seeds, {counts}"
        return "helps" if delta > 0 else "hurts"

    noise = ", ".join(
        f"{_signed(d * 100, 2)} pp AUC under {k} ({word(k, d)})" for k, d in auc_deltas.items()
    )
    sb = raw["cost"]["static"]["baseline"]
    sh = raw["cost"]["static"]["hopfield"]
    macs_pct = (sh["macs_per_image"] - sb["macs_per_image"]) / sb["macs_per_image"] * 100.0
    return (
        f"VERDICT: Hopfield {clean} on clean data (Δ head_acc {_signed(e1['delta'])}, p={_p(e1['p'])});"
        f" {noise}; for {_signed(summary['E7']['latency_overhead_pct'], 2)} % latency,"
        f" +{sh['model_size_bytes'] - sb['model_size_bytes']:,} bytes,"
        f" {_signed(macs_pct, 3)} % MACs, +0 trainable params"
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def flatten_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (experiment, model, varied value, seed), plus cost rows."""
    varied = {"E1": "model", "E2": "beta", "E3a": "sigma", "E3b": "p", "E3c": "occlusion",
              "E4": "T", "E5": "beta", "E6": "placement", "E9a": "memory_design",
              "E9b": "memory_design"}
    rows: list[dict[str, Any]] = []
    for exp, points in raw["results"].items():
        for point in points:
            for s_idx, seed in enumerate(raw["config"]["seeds"]):
                for model in MODELS:
                    row: dict[str, Any] = {"experiment": exp, "model": model, "varied": varied[exp],
                                           "value": point["value"], "seed": seed,
                                           **point[model][s_idx]}
                    if model == "hopfield":
                        for test in ("head", "proto"):
                            mc = point[f"mcnemar_{test}"][s_idx]
                            row.update({f"mcnemar_{test}_b": mc["b"], f"mcnemar_{test}_c": mc["c"],
                                        f"mcnemar_{test}_p": mc["p"]})
                    if "beta" in point:
                        row["beta"] = point["beta"][s_idx]
                    rows.append(row)
    cost = raw["cost"]
    seed0 = raw["config"]["seeds"][0]
    for model in MODELS:
        rows.append({"experiment": "E7", "model": model, "varied": "model", "value": model,
                     "seed": seed0, **{k: v["median"] for k, v in cost["e7"][model].items()
                                       if isinstance(v, dict)},
                     **{k: v for k, v in cost["static"][model].items() if not isinstance(v, list)}})
        for bs, entry in cost["e8"].items():
            rows.append({"experiment": "E8", "model": model, "varied": "batch_size", "value": int(bs),
                         "seed": seed0, **{k: v["median"] for k, v in entry[model].items()
                                           if isinstance(v, dict)}})
    return rows


def save_reports(raw: dict[str, Any], summary: dict[str, Any], output_dir: Path) -> None:
    """Write summary.md, results.csv, results.json, plots.png and cost.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(summary["markdown"], encoding="utf-8")

    rows = flatten_rows(raw)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {k: v for k, v in raw.items() if k != "results"}
    payload["results"] = raw["results"]
    payload["summary"] = {k: v for k, v in summary.items() if k != "markdown"}
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    plot_results(raw, output_dir / "plots.png")
    plot_costs(raw, output_dir / "cost.png")


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed; skipping plots.", UserWarning, stacklevel=3)
        return None
    return plt


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _series(points: list[dict[str, Any]], model: str, metric: str) -> tuple[list[float], list[float]]:
    stats = [aggregate_point(p, metric)[model] for p in points]
    return [s[0] for s in stats], [s[1] for s in stats]


def _band_line(ax: Any, xs: list[float], points: list[dict[str, Any]], model: str,
               metric: str = "head_acc", **kwargs: Any) -> None:
    color = BASELINE_COLOR if model == "baseline" else HOPFIELD_COLOR
    mean, std = _series(points, model, metric)
    ax.plot(xs, mean, color=color, linewidth=2, marker="o", markersize=5,
            label=model.capitalize(), **kwargs)
    ax.fill_between(xs, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                    color=color, alpha=0.18, linewidth=0)


def plot_results(raw: dict[str, Any], path: Path) -> bool:
    """plots.png (2×3): E3a/E3b/E3c, E2 accuracy+entropy vs β, E4 vs T, confusions."""
    plt = _import_pyplot()
    if plt is None:
        return False
    from matplotlib.colors import LinearSegmentedColormap

    res = raw["results"]
    bstar = raw["beta_star"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle(f"Task 13: Baseline (L8 = Identity) vs Hopfield (L8 = HopfieldMemory), β*={bstar:g}"
                 f"  · mean ± std over {len(raw['config']['seeds'])} seed(s)", fontsize=13, x=0.01,
                 ha="left")

    for ax, kind in zip(axes[0], NOISE_TYPES):
        pts = res[E3_IDS[kind]]
        xs = [float(p["value"]) for p in pts]
        for model in MODELS:
            _band_line(ax, xs, pts, model)
        label = {"gaussian": "Gaussian σ", "salt_pepper": "salt-and-pepper p",
                 "occlusion": "occlusion side (px)"}[kind]
        _style(ax, f"{E3_IDS[kind]} head_acc vs {label}", label, "head_acc")
        ax.legend(frameon=False)

    ax = axes[1, 0]
    pts = res["E2"]
    xs = [float(p["value"]) for p in pts]
    ent, _ = _series(pts, "hopfield", "entropy_norm")
    base = _series(pts, "baseline", "head_acc")[0][0]
    _band_line(ax, xs, pts, "hopfield")
    ax.plot(xs, ent, color=HOPFIELD_COLOR, linewidth=2, linestyle=":", marker="s", markersize=5,
            label="Hopfield entropy_norm")
    ax.axhline(base, color=BASELINE_COLOR, linestyle="--", linewidth=2, label="Baseline head_acc")
    ax.axvline(bstar, color="#8a8984", linewidth=1, linestyle="-.")
    ax.text(bstar, 0.02, f" β*={bstar:g}", color="#52514e", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    _style(ax, "E2 clean: head_acc and entropy_norm vs β", "β (log scale)", "value (both on [0, 1])")
    ax.legend(frameon=False, fontsize=8, loc="center right")

    ax = axes[1, 1]
    pts = res["E4"]
    xs = [float(p["value"]) for p in pts]
    _band_line(ax, xs, pts, "hopfield")
    ax.axhline(_series(pts, "baseline", "head_acc")[0][0], color=BASELINE_COLOR, linestyle="--",
               linewidth=2, label="Baseline (reference)")
    ax.set_xticks(xs)
    _style(ax, f"E4 head_acc vs T (σ={raw['config']['noisy_sigma']:g})", "update steps T", "head_acc")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    cmap.set_bad("#ffffff")

    def row_norm(name: str) -> list[list[float]]:
        conf = torch.tensor(raw["confusions"][name], dtype=torch.float64)
        return (conf / conf.sum(dim=1, keepdim=True).clamp_min(1)).tolist()

    clean, noisy = row_norm("hopfield_clean"), row_norm("hopfield_noisy")
    grid = [c + [float("nan")] + n for c, n in zip(clean, noisy)]
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(list(range(10)) + list(range(11, 21)))
    ax.set_xticklabels([str(i) for i in range(10)] * 2, fontsize=7)
    ax.set_yticks(range(10))
    ax.set_title(f"Hopfield retrieval confusion: clean | σ={raw['config']['noisy_sigma']:g}",
                 fontsize=11, loc="left")
    ax.set_xlabel("retrieved prototype")
    ax.set_ylabel("true digit")
    fig.colorbar(im, ax=ax, fraction=0.025, label="row fraction")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return True


def plot_costs(raw: dict[str, Any], path: Path) -> bool:
    """cost.png (1×3): latency/image vs batch, throughput vs batch, accuracy vs latency (E4)."""
    plt = _import_pyplot()
    if plt is None:
        return False
    cost = raw["cost"]
    batches = [int(b) for b in cost["e8"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(f"Task 13 cost on {cost['device']} ({cost['num_threads']} threads), median of"
                 f" {raw['config']['cost']['repeats']} interleaved repeats", fontsize=13, x=0.01, ha="left")
    for model in MODELS:
        color = BASELINE_COLOR if model == "baseline" else HOPFIELD_COLOR
        lat = [cost["e8"][str(b)][model]["latency_per_image_us"]["median"] for b in batches]
        thr = [cost["e8"][str(b)][model]["throughput_img_s"]["median"] for b in batches]
        axes[0].plot(batches, lat, color=color, linewidth=2, marker="o", label=model.capitalize())
        axes[1].plot(batches, thr, color=color, linewidth=2, marker="o", label=model.capitalize())
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    _style(axes[0], "E8 latency per image vs batch", "batch size (log)", "µs / image (log)")
    _style(axes[1], "E8 throughput vs batch", "batch size (log)", "images / s")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    ax = axes[2]
    lat = cost["e4_latency"]
    e4 = raw["results"]["E4"]
    ts = raw["config"]["num_updates_values"]
    hop_acc = _series(e4, "hopfield", "head_acc")[0]
    hop_lat = [lat[f"T={t}"]["latency_ms_p50"]["median"] for t in ts]
    ax.plot(hop_lat, hop_acc, color=HOPFIELD_COLOR, linewidth=2, marker="o", label="Hopfield (T)")
    for t, x, y in zip(ts, hop_lat, hop_acc):
        ax.annotate(f"T={t}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9,
                    color="#52514e")
    ax.plot([lat["baseline"]["latency_ms_p50"]["median"]], [_series(e4, "baseline", "head_acc")[0][0]],
            color=BASELINE_COLOR, marker="D", markersize=9, linestyle="none", label="Baseline")
    _style(ax, f"E4 accuracy vs latency (σ={raw['config']['noisy_sigma']:g},"
               f" batch {raw['config']['cost']['batch_size']})", "latency p50 (ms)", "head_acc")
    ax.legend(frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser(
        "Task 13: MNIST prototype retrieval, Baseline vs Hopfield (one variable at a time)."
    ).parse_args(argv)
    cfg = config_from_args(args)
    print(f"Running Task 13 from {args.config}{' (quick)' if cfg['quick'] else ''}")
    raw = run_experiment(cfg, retrain=args.retrain)
    text, summary = build_report(raw)
    print()
    print(text)
    output_dir = Path(cfg["reports_dir"])
    save_reports(raw, summary, output_dir)
    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    main()

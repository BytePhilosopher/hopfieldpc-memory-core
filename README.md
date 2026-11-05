# hopfieldpc-memory-core

Standalone modern Hopfield memory core for the HopfieldPC project.

<p align="justify">
This repository implements and validates the core Hopfield memory mechanism before integration into FabricPC. It focuses on beta-controlled associative retrieval, tensor-shape validation, diagnostics, synthetic benchmarks, and unit tests.

## Setup

Clone the repository:

```bash
git clone https://github.com/iCog-Labs-Dev/hopfieldpc-memory-core.git
cd hopfieldpc-memory-core
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import hopfieldpc; print('hopfieldpc import OK')"
```

## Current Implementation Stage

Current task:

```text
Task 2: Implement one-step Hopfield retrieval
```

Implemented so far:

```text
src/hopfieldpc/api.py
src/hopfieldpc/functional.py
src/hopfieldpc/memory.py
src/hopfieldpc/__init__.py
tests/test_api_contract.py
tests/test_one_step_retrieval.py
tests/test_beta_behavior.py
```

The module now validates tensor shapes and performs one-step modern Hopfield retrieval.

## Expected Tensor Contract

```text
query:     [batch_size, dim]
memory:    [num_memories, dim]
retrieved: [batch_size, dim]
weights:   [batch_size, num_memories]
```

## Core Retrieval Formula

```python
scores = beta * query @ memory.T
weights = softmax(scores)
retrieved = weights @ memory
```

Equivalent mathematical form:

```text
p = softmax(β qMᵀ)
h_memory = pM
```

## Run Tests

Run all tests:

```bash
pytest -q
```

Run only the API contract tests:

```bash
pytest tests/test_api_contract.py -q
```

Run only one-step retrieval tests:

```bash
pytest tests/test_one_step_retrieval.py tests/test_beta_behavior.py -q
```

## Benchmark Layout and Tuning

Each dataset has its own folder. `tuning.py` is the only code that tunes anything. It writes the
tuned values to the `tuned_params.json` next to it, which is committed. Baseline and Hopfield
experiments only read that file, so tuning runs once per dataset and settings.

```text
benchmarks/
  common/       tuned_params.py (tuned-parameter store), validation.py
  mnist/        tuning.py → tuned_params.json      (baseline, L8, MNIST-C, L5)
                baseline.py               CNN, L8 = Identity (shared data, training, metrics)
                hopfield.py               Hopfield at L8, frozen memory (E1–E9)
                hopfield_mnist_c.py       the L8 networks on MNIST-C
                hopfield_recentering.py   layer change: test-batch centering at L8
                hopfield_l5.py            layer change: Hopfield after L5 (Flatten)
  cifar10/      tuning.py → tuned_params.json
                baseline.py               ResNet-18, L8 = Identity
                hopfield.py               Hopfield at L8, train-mean and test-batch centering
  synthetic/    synthetic_retrieval.py, beta_sweep.py, cluster_retrieval.py (sweeps, nothing tuned)
configs/
  mnist/        hopfield.yaml, hopfield_l5.yaml
  synthetic/    beta_sweep.yaml, cluster_retrieval.yaml, synthetic_retrieval.yaml
```

MNIST `tuning.py` tunes these sections, in dependency order, all on validation data only:

| Section              | Tuned values                                  | Read by                                            |
| -------------------- | --------------------------------------------- | -------------------------------------------------- |
| `baseline`           | lr, batch size, epochs                        | every MNIST experiment                             |
| `hopfield_l8`        | β\*, pixel-placement β, memory-variant βs     | `hopfield.py`                                      |
| `hopfield_mnist_c`   | β\* on the wider MNIST-C grid                 | `hopfield_mnist_c.py`, `hopfield_recentering.py`   |
| `hopfield_l5`        | β of Hopfield@L5 (every memory variant) and @L8 | `hopfield_l5.py`        |

```bash
python -m benchmarks.mnist.tuning                      # skips every section that is already tuned
python -m benchmarks.mnist.tuning --quick              # smoke settings, stored under a separate "quick" profile
python -m benchmarks.mnist.tuning --only hopfield_l5   # one section
python -m benchmarks.mnist.tuning --force              # tune again even though stored values match
```

Each stored entry records the settings it was tuned with. If you change a relevant setting (a grid,
the selection sets, the backbone recipe, the memory design), the entry becomes stale. Experiments
then stop and print the tuning command to run, and `tuning.py` retunes only the stale sections. The
current MNIST values (full and quick) and the CIFAR-10 quick values were carried over from earlier
runs, so nothing needs to be tuned again.

## Task 13: MNIST Prototype Retrieval (Baseline vs Hopfield)

A controlled experiment with one CNN and two variants that differ only in layer L8, between the
embedding `z` (L7) and the classifier head (L9):

```text
L1 Conv 1→32 → L2 Pool → L3 Conv 32→64 → L4 Pool → L5 Flatten → L6 Linear 3136→128 → L7 Linear 128→64 = z
L8  Baseline: h = z (Identity)    Hopfield: h = softmax(β·cos(z − μ, M − μ))·M   (frozen memory M)
L9 Linear 64→10
```

The baseline is trained once per seed on clean MNIST (cached as
`reports/mnist_prototype_retrieval/backbone_seed{s}.pt`). The Hopfield variant copies the same
L1–L7/L9 weights and adds 0 trainable parameters. Its memory follows Ramsauer et al. (2020):

- M = 8 k-means prototypes per class of the training embeddings (80 × 64), frozen.
- Scores are centered by the train-mean embedding μ, which increases pattern separation.
- β\* is selected on the validation split, averaged over clean, Gaussian σ = 0.4 and
  salt-and-pepper p = 0.15 sets (noise levels that are not in the test grids). `tuning.py` does
  this once and stores the result.

```bash
python -m benchmarks.mnist.tuning              # once (already done: values are in tuned_params.json)
python -m benchmarks.mnist.hopfield            # everything (10 seeds, E1–E9)
python -m benchmarks.mnist.hopfield --quick    # 1 seed, 1 epoch, reduced grids
python -m benchmarks.mnist.baseline            # baseline only (E1/E3 baseline columns)
# options: --config, --retrain, --seed N, --data-dir DIR, --reports-dir DIR, --tuned-params FILE
```

Settings are in `configs/mnist/hopfield.yaml`. Outputs go to
`reports/mnist_prototype_retrieval/` (`--quick` writes to `reports/mnist_prototype_retrieval_quick/`):

- `summary.md`: the file to read. Each table (E1–E9) varies exactly one parameter, stated in its
  `Varied:` / `Held fixed:` header. Columns: Baseline | Hopfield | Δ (H−B) | McNemar p (max over
  seeds) | Seeds +/− | Wilcoxon p (across seeds) | Winner. Winner is `tie` when |Δ| < 0.005 or
  Wilcoxon p ≥ 0.05. E9 ablates the memory design (class means raw / centered, 8 per class
  centered). The file ends with the S1–S11 sanity report and a one-line VERDICT.
- `results.csv` / `results.json`: every (experiment, model, value, seed) row, plus the config,
  environment, β\*, confusion matrices and cost measurements.
- `plots.png` (accuracy vs noise, β, T and retrieval confusion) and `cost.png` (latency and
  throughput vs batch, accuracy vs latency).

Result (10 seeds, tuned baseline, β\* = 100): clean accuracy is unchanged (98.62 % vs 98.66 %,
tie), while robustness AUC rises by +5.55 pp under Gaussian noise and +5.35 pp under
salt-and-pepper noise, better in all 10 seeds (Wilcoxon p = 0.002). Occlusion shows no significant
effect (+0.41 pp). The layer adds 0 trainable parameters and +0.24 % MACs. A hard
nearest-centered-prototype readout matches the Hopfield layer, so the gain comes from the memory
design rather than from soft retrieval.

### Placement: Hopfield after L5 vs at L8

```bash
python -m benchmarks.mnist.hopfield_l5            # 10 seeds, reuses the cached backbones
python -m benchmarks.mnist.hopfield_l5 --quick    # smoke run
```

The same frozen memory (8 centered k-means prototypes per class) is placed on the 3136-d flatten
features (L5, β = 50) instead of the 64-d embedding (L8, β = 100), with identical frozen weights.
Result (10 seeds): Gaussian AUC +14.53 pp and salt-and-pepper AUC +15.11 pp over the baseline
(Holm p = 0.004), against +5.55 / +5.35 pp at L8. The cost is −2.48 pp clean accuracy, +11.9 % MACs
and about +30 % latency. The full report, with architecture, protocol, every table and the verdict,
is in [docs/mnist_hopfield_placement_noise_report.md](docs/mnist_hopfield_placement_noise_report.md).

Timings only compare within one run; on CPU, peak memory is an activation upper bound.

### MNIST-C (unseen corruptions)

```bash
python -m benchmarks.mnist.hopfield_mnist_c            # 10 seeds, reuses cached backbones
python -m benchmarks.mnist.hopfield_mnist_c --quick    # smoke run
```

Evaluates the same frozen networks on the 15 MNIST-C test corruptions (Mu & Gilmer 2019), with β\* selected on
validation data only. Outputs are `mnist_c_summary.md`, `mnist_c_results.csv`, `mnist_c_results.json` and
`mnist_c_plot.png` in the reports directory; the main experiment's files are not touched.

Result (10 seeds, β\* = 100): no overall gain, with mean accuracy over the 15 corruptions at 84.59 % vs 84.72 %
(tie, Wilcoxon p = 0.49). Hopfield wins on impulse noise, shot noise, zigzag and canny edges, and loses heavily on
fog (−20.1 pp); the other 10 corruptions are ties after Holm correction.

Test-batch recentering centers each query batch on its own mean instead of the train mean, with no new parameters
and no retraining:

```bash
python -m benchmarks.mnist.hopfield_recentering
```

Result (10 seeds, batch 256): mean accuracy over the 15 corruptions rises from 84.72 % (baseline) to 90.98 %
(+6.26 pp, better in all 10 seeds, Holm p = 0.008), fog goes from 69.65 % to 85.38 %, and uncorrupted accuracy is
unchanged. The gain shrinks to +1.14 pp when one batch mixes corruptions and turns into a loss (−1.59 pp) in mixed
batches of 16. This method was chosen after seeing MNIST-C results, so it still needs an independent test.

### CIFAR-10-C (independent test, run on Colab)

`benchmarks/cifar10/` tests the same method on a ResNet-18 trained on clean CIFAR-10, with the main result fixed in
advance: Hopfield recentered vs Baseline at batch 256, mean accuracy AUC over severities 0-5 across the 15 standard
corruptions. `python -m benchmarks.cifar10.tuning` trains the backbones and stores β\* once, then
`python -m benchmarks.cifar10.hopfield` evaluates. Both need a GPU and resume from the output directory. The
docstring of `benchmarks/cifar10/hopfield.py` lists the Colab cells. Full-run results are not available yet.

## Next Planned Tasks

```text
Task 3: Add memory modes and prepare module extension points.
Task 4: Add optional multi-step retrieval.
Task 5: Add diagnostics such as entropy, top-k mass, and retrieval distance.
Task 6: Add synthetic retrieval benchmarks.
```

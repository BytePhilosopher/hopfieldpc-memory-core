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
  salt-and-pepper p = 0.15 sets (noise levels that are not in the test grids).

```bash
python -m benchmarks.mnist_prototype_retrieval            # everything (10 seeds, E1–E9)
python -m benchmarks.mnist_prototype_retrieval --quick    # 1 seed, 1 epoch, reduced grids
python -m benchmarks.mnist_baseline                       # baseline only (E1/E3 baseline columns)
# options: --config, --retrain, --seed N, --data-dir DIR, --reports-dir DIR
```

Settings are in `configs/mnist_prototype_retrieval.yaml`. Outputs go to
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

Result (10 seeds, β\* = 100): clean accuracy is unchanged (98.86 % vs 98.89 %, tie), while
robustness AUC rises by +7.2 pp under Gaussian noise and +7.8 pp under salt-and-pepper noise,
better in all 10 seeds (Wilcoxon p = 0.002). Occlusion shows no significant effect. The layer adds
0 trainable parameters and +0.24 % MACs. A hard nearest-centered-prototype readout matches the
Hopfield layer, so the gain comes from the memory design rather than from soft retrieval.

Timings only compare within one run; on CPU, peak memory is an activation upper bound.

## Next Planned Tasks

```text
Task 3: Add memory modes and prepare module extension points.
Task 4: Add optional multi-step retrieval.
Task 5: Add diagnostics such as entropy, top-k mass, and retrieval distance.
Task 6: Add synthetic retrieval benchmarks.
```

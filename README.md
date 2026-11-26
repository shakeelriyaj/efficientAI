# FinBERT: early exit and compact students

Compare a full 12-layer financial sentiment classifier, dynamic early exit,
and fixed eight-layer students trained with gold labels (CE) or distillation (KD).
This shared branch contains the experiment code, raw results and report.

## Start here

- **Latest results:** [controlled A100 comparison](reports/controlled_a100/results.md),
  including all training seeds, batches, overhead controls and limitations.
- **Report:** [PDF](reports/controlled_a100/Efficient_AI_Final_Report.pdf) or
  [editable Overleaf project](https://www.overleaf.com/project/6b07a79c4b26e8098a123ad3).
- **W&B:** [verified summary run](https://wandb.ai/wk2376-efficient-ai/finbert-controlled-a100/runs/435ttxp3)
  and [project](https://wandb.ai/wk2376-efficient-ai/finbert-controlled-a100).
  Team membership is required to view private runs.
- **Code:** read [run_controlled.py](run_controlled.py), then
  [controlled_routes.py](controlled_routes.py) and [train.py](train.py).
  Settings are in [controlled.json](controlled.json); the frozen plan and
  preemption amendment are in [CONTROLLED_PROTOCOL.md](CONTROLLED_PROTOCOL.md).
- **Evidence:** [raw CSVs and provenance](reports/controlled_a100/raw/),
  [file checksums](reports/controlled_a100/checksums.json), and
  [independent audit script](tools/analyze_controlled.py).

## Main finding

The predefined primary seed is 4142. At batch size one, KD students reduce
median end-to-end latency by **27.9% on SEntFiN** and **26.4% on PhraseBank**.
Their observed macro-F1 losses are **-0.02 pp** and **0.80 pp**, respectively,
within the 1 percentage point budget. A negative loss is a small observed gain.

Dynamic exit reduces mean latency but makes batch-one median latency
**2.8% higher on SEntFiN** and **8.5% higher on PhraseBank**. Intermediate
classification and CPU confidence checks cost time when a sentence continues.
Fewer executed layers do not guarantee a faster response.

This is not a guarantee across training runs: PhraseBank KD passes the observed
test budget in **two of three seeds**. SEntFiN's seed-2 CE and KD checkpoints
fail the stricter validation margin, even though their test losses are below
1 pp. All candidates are retained in the results. The reused test sets are
development evidence, not fresh holdouts or population guarantees.

## Same-configuration comparison

Both datasets use NVIDIA A100-SXM4-80GB GPUs, driver 610.43.02, FP32 eager
inference, TF32 off, one CPU thread and a 96-token cap. Each dataset's methods
and batches are paired within one uninterrupted allocation. The two datasets
use separate allocations; absolute times from older studies are not pooled.

The comparison includes full, dynamic, CE and KD at batch sizes 1, 4, 16 and 32;
three training seeds per objective; and forced-full-depth overhead controls.
All **288,000 timed calls** were checked against saved predictions and depths.
Teacher and exit-head weights remained frozen; every student parameter was
trainable. W&B logging and uploads happen outside measured inference calls.

## Reproduce or inspect

For a CPU-only results audit, install NumPy, pandas and scikit-learn and run:

```bash
python tools/analyze_controlled.py reports/controlled_a100/raw
```

This recomputes F1, timing summaries, seed variation, overhead and bootstrap
intervals from the committed CSVs. It does not require checkpoints or a GPU.

For training and timing, prepare the existing teacher, splits and exit heads
using [the input/reproduction guide](REPRODUCTION.md). Exact input hashes are
recorded in each dataset's `asset_manifest.json`. Weights and input bundles
stay outside Git. A fresh teacher rebuild is a different training run.

Inside an A100-SXM4-80GB allocation with the environment from
[hpc/env.sh](hpc/env.sh):

```bash
python -m pip install -r requirements-controlled.txt
python preflight_controlled.py
python run_controlled.py --sentfin-assets /path/to/sentfin-assets \
  --phrasebank-assets /path/to/phrasebank-assets \
  --phrasebank-exits /path/to/phrasebank-exits --out results/new-controlled
```

Checkpoint selection uses validation before test labels are read. Failures
are recorded alongside passes. The W&B preflight uses an existing login; if
unavailable, it records offline mode and cloud synchronization remains necessary.

NYU wrappers are [hpc/controlled_setup.sbatch](hpc/controlled_setup.sbatch) and
[hpc/controlled.sbatch](hpc/controlled.sbatch). The setup wrapper points to the
original Torch environment; change those paths for another account. Pass three
asset directories, then optionally `sentfin` or `phrasebank` as the fourth
argument. For an interrupted timing run, the fifth argument is the original
result root containing locked student checkpoints. Partial timing sessions must
be discarded and all methods for that dataset rerun together.

## Earlier work

These remain available for provenance and are distinct from the latest protocol:

- [SEntFiN ten-seed A100 study](reports/nyu_hpc/a100-student-results.md)
- [Multi-exit and small-batch study](reports/nyu_hpc/batch-exit-results.md)
- [Earlier PhraseBank A100 comparison](reports/phrasebank/a100-results.md)
- [Original notebook snapshot](notebooks/FinBERT_Colab.ipynb),
  [source record](notebooks/source.json), and [reproduction guide](REPRODUCTION.md)

Use the latest controlled comparison for the report's CE-versus-KD and
cross-dataset conclusions. Older latency values come from other configurations
or sessions and should not be combined with the new timings.

## Code checks

`python -m pytest tests -q` passes all 21 tests. They cover frozen-teacher
training, real layer skipping, batch compaction/order, forced-depth controls
and rejection of mismatched timed predictions. The results audit above also
passes for both datasets. Exact executed code snapshots are retained with the
raw results; the published runner only removes one redundant error check and
clarifies its description after execution.

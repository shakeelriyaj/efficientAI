# FinBERT: reproduce the three-model comparison

Classify financial headlines with the full 12-layer teacher, dynamic layer-8
exit, and an eight-layer distilled student. English code; five small core files.

## Start here

- **Review or reproduce the final comparison:** start with `run.py` and the
  five small Python files described below.
- **Colab:** [downloaded notebook](notebooks/FinBERT_Colab.ipynb) or
  [original live Colab](https://colab.research.google.com/drive/124xteylTSfZKCAi61_v_XUhIa7ss25Mn).
  The downloaded notebook is an unchanged snapshot of the full experiment
  history, including discarded trials. It is longer than the final codebase.
- **Results:** [simple HPC report](reports/HPC_Reproduction_Report.txt), with
  metrics, predictions, raw timings, logs, and verification in `reports/nyu_hpc/`.
- **Team context:** this work lives on the `finbert-hpc-reproduction` branch of
  [the shared repository](https://github.com/shakeelriyaj/efficientAI).

Large checkpoints and training inputs stay outside Git. Their exact input hashes
are in `reference/assets_manifest.json`; the existing Torch copy is listed below.
`notebooks/source.json` records the downloaded notebook's origin and checksum.

Read `run.py` first. It connects these four helpers:

| File | What to review |
|---|---|
| `models.py` | One real early-return path; copy eight teacher blocks for the student |
| `train.py` | Student training and validation-only checkpoint selection |
| `benchmark.py` | Paired, randomized timing with every label/depth checked |
| `data.py` | Exact saved splits, checksums, tokenization, and scores |

`experiment.json` contains the settings. No notebook history, plotting framework,
or hyperparameter search is needed to run this comparison.

The proposal's multi-exit study has its own entry point, `run_exits.py`, with
the exit logic in `exits.py` and settings in `exits.json` (see below).

## What is reproduced

1. **Saved-model replay:** load the exact Colab weights and compare every test
   prediction and exit depth with `reference/colab_predictions.csv`.
2. **Student retraining:** initialize from teacher blocks 1, 2, 4, 5, 7, 8, 10,
   and 12; train all student parameters for five epochs, seed 4142. Choose the
   checkpoint on validation only, then evaluate once without changing settings.

The original teacher and dynamic head remain frozen. The student's backbone
**is trainable**. Teacher mistakes use gold labels alone. Otherwise the loss is
75% gold-label CE plus 25% teacher KL, with distillation temperature 2.

The saved splits contain 4704 train, 786 validation, 782 calibration, and 1573
test headlines. Reuse them: do not randomly split the data again.

Reference Colab results:

| Model | Macro-F1 | Median end-to-end latency |
|---|---:|---:|
| Full 12 layers | 0.878955 | 9.55 ms |
| Dynamic layer-8 exit | 0.875099 | 9.43 ms |
| Compact 8-layer student | 0.876573 | 6.77 ms |

HPC hardware differs. Do not expect identical milliseconds or bitwise identical
new training. Compare quality and latency ratios against the full model measured
on the same allocated GPU. Saved checkpoint predictions are checked exactly.

## Inputs

The checkpoints are kept outside the source code. On NYU Torch they are already
in `/scratch/wk2376/efficient-ai-repro-20261105/assets`.

To reconstruct inputs elsewhere, use the three previously delivered archives:

```bash
python tools/prepare_assets.py \
  --teacher /path/to/sentfin_full_seed42_20261105_results.zip \
  --head /path/to/finbert_layer8_training_results_20261105.zip \
  --student /path/to/finbert_compact8_student_checkpoint_20261105.zip
```

The preparation script checks all inputs against the frozen reference hashes.
No account, Hugging Face access, or model download is needed after preparation.

### Rebuilding inputs from public sources (no archives or shared folder)

`tools/build_assets.py` recreates `assets/` from the pinned public SEntFiN commit
and `ProsusAI/finbert` revision. The rebuilt splits are byte-identical to Colab's
(all six split checksums match). The teacher is fine-tuned again with the
notebook's recipe (3 epochs, seed 42), so it is a new run, not Colab's weights.

```bash
mkdir -p logs
sbatch --account=torch_pr_1349_general hpc/setup.sbatch       # once: Python environment
.venv/bin/python tools/build_assets.py download --out assets   # login node: downloads only
sbatch --account=torch_pr_1349_general hpc/build_assets.sbatch # GPU: splits + teacher
sbatch --account=torch_pr_1349_general hpc/exits.sbatch        # then the multi-exit study
```

`assets/build_record.json` records the data audit, split check, and teacher
history. Rebuilt assets work with `run_exits.py`. They lack Colab's `head8.pt`,
student, and teacher weights, so `run.py verify` and `run.py train` (which
compare against Colab predictions) are for the original assets only.

## Run on NYU Cloud Bursting (course allocation)

Use Open OnDemand at https://ood.burst.hpc.nyu.edu/ (NYU VPN off campus), start
a Jupyter session, and open its terminal. Every job needs the course account;
choose the partition per job: `n2c48m24` (CPU) or `c12m85-a100-1` (1 × A100 40GB).

```bash
git clone -b finbert-hpc-reproduction https://github.com/shakeelriyaj/efficientAI.git
cd efficientAI && mkdir -p logs
ACCOUNT=csci_ga_3033-2026fa; GPU=c12m85-a100-1; CPU=n2c48m24

# 1. Check the GPU driver once. PyTorch's default build needs driver 580+.
srun --account=$ACCOUNT --partition=$GPU --gres=gpu:1 --time=00:10:00 nvidia-smi
# 2. Python environment (CPU node). If the driver is below 580, prefix with
#    FINBERT_TORCH_CUDA=cu128 to install the CUDA 12.8 PyTorch build instead.
sbatch --account=$ACCOUNT --partition=$CPU hpc/setup.sbatch
# 3. After setup finishes: download public inputs, then build splits + teacher.
.venv/bin/python tools/build_assets.py download --out assets
sbatch --account=$ACCOUNT --partition=$GPU hpc/build_assets.sbatch
# 4. After the build finishes: the multi-exit study.
sbatch --account=$ACCOUNT --partition=$GPU hpc/exits.sbatch
```

Cloud nodes start on demand, so a job may wait a few minutes before running.
`squeue -u $USER` shows job state; logs are in `logs/`. Each GPU log begins with
the GPU name and driver. The scripts run without a container here; on Torch
they use NYU's CUDA image automatically.

## Run on NYU Torch

From this codebase's root, after copying it and its `assets/` folder to Torch:

```bash
mkdir -p logs
setup_job=$(sbatch --parsable --account=torch_pr_1349_general hpc/setup.sbatch)
sbatch --account=torch_pr_1349_general --dependency=afterok:$setup_job hpc/run.sbatch
```

On Torch, add `--constraint=cuda` to GPU jobs, as the original runs did.
Use your own active account from `my_slurm_accounts` if running under another
allocation. The setup job installs Python 3.13.16 and the pinned ML packages on
a CPU compute node. The reproduction job requests one NVIDIA GPU and runs both
commands. It uses NYU's CUDA 13 image and records resolved package versions.

Direct commands inside that GPU environment:

```bash
.venv/bin/python run.py verify --assets assets --out results/replay
.venv/bin/python run.py train  --assets assets --out results/retrain
```

Output folders must be new, preventing accidental overwrites. Each contains
metrics, per-headline predictions, raw timings, environment details, and a
verification record. Training also saves the selected standard Hugging Face
checkpoint, training history, and validation selection record.

### Several training seeds

`--seed N` replaces only the training seed from `experiment.json`; each output's
`config.json` records the seed used. To retrain the student with ten seeds
fixed in advance (4142 repeats the original run), one GPU task per seed:

```bash
seeds_job=$(sbatch --parsable --account=torch_pr_1349_general hpc/seeds.sbatch)
# after all tasks finish:
.venv/bin/python tools/summarize_seeds.py results/seeds-$seeds_job
```

The summary writes `seed_summary.csv` and `seed_summary.txt`: mean, spread, and
worst-seed student F1 drop, runs within budget, selected epochs, and per-run
latency reduction against the full model on the same GPU. Report every seed.
Seed spread measures training variation only, not test-set uncertainty.

Timing uses FP32 eager inference, batch size one, one CPU thread, 512 unique
headlines, three repeats per timing mode, and ten excluded warmup predictions.
Both modes include a CPU label. End-to-end timing also includes tokenization
and input transfer. Three routes produce 9,216 checked timing records per run.

## Multi-exit study (layers 2, 4, 8; max-probability or entropy)

`run_exits.py` implements the proposal's method on the frozen teacher:

1. Train one exit head per layer in `exit_layers` (a copy of the teacher's
   pooler and classifier) on that layer's [CLS] vector. Loss: 75% gold-label CE
   plus 25% teacher KL; teacher mistakes use gold labels alone. Validation F1
   picks each head's epoch. Only training labels enter backward().
2. Fit one temperature per head on the calibration split.
3. Record, for every validation and test headline at batch size one, each
   exit's calibrated max probability, normalized entropy (0 certain, 1 uniform),
   and label. Any policy can then be scored exactly without rerunning the model.
4. Sweep both rules over the exit sets {8}, {4,8}, {2,8}, {2,4,8} and the
   threshold grids in `exits.json`. Keep policies within the validation F1 limit
   (0.75 points, leaving margin for the 1-point test budget), take the fewest
   mean layers per rule and exit set, time those finalists on validation, and
   select the fastest median per rule.
5. Evaluate once on test: full model, static exits at each layer, and the two
   selected dynamic policies. Real batch-one predictions must equal the
   simulation exactly, and every timed prediction is checked.

```bash
sbatch --account=torch_pr_1349_general hpc/exits.sbatch        # default seed 3142
sbatch --account=torch_pr_1349_general hpc/exits.sbatch 7      # another head seed
.venv/bin/python run_exits.py --assets assets --out results/exits-debug --device cpu --skip-timing
```

Outputs: `metrics.csv` (quality, exit distribution, latency, speedup),
`validation_sweep.csv` (every candidate policy, for trade-off curves),
`validation_timing.csv`, `selected_policies.json`, `calibration.csv`,
`head_history.json`, `predictions.csv` (with text, for error analysis),
`test_exit_scores.csv` (per-headline exit confidences), `timings.csv`,
`exit_heads.pt`, `temperatures.json`, and `verification.json`. With
`--skip-timing`, policies are chosen by mean layers, so it is not a speed result.

## Limits

The 1 percentage point macro-F1 rule is checked against the full model on this
same development test set. It is not a population guarantee. The Colab student
F1-drop interval extends above one point; fresh holdout data and more seeds are
needed before a final research claim. This code repeats the selected experiment,
not the earlier teacher fine-tuning or the discarded weight-edit experiments.

NYU instructions: [connection](https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/),
[job submission](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/),
[containers on compute nodes](https://services.rt.nyu.edu/docs/hpc/tutorial_apptainer/running_containers/).

## Checks

`python -m pytest tests` runs twelve focused tests (requires pytest). They check
that teacher errors use gold labels, student training preserves the teacher,
and an early exit actually skips later blocks. `source_sha256.json` records
the reviewed code version. The full local saved-model replay also matches all
Colab test labels and exit depths exactly.

NYU Torch CPU replay passed (job 22013816, node cs638): all 1,573 labels and
exit depths match for each model. GPU job 22013353 also completed successfully
on an NVIDIA B200 (node gb002). Saved-model GPU predictions match exactly.

Fresh student training selected epoch 4 of 5 on validation. Its paired benchmark:

| Model | Test macro-F1 | Median end-to-end latency |
|---|---:|---:|
| Full 12 layers | 0.878955 | 2.559 ms |
| Dynamic layer-8 exit | 0.875099 | 2.623 ms |
| Fresh compact student | 0.879177 | 1.885 ms |

The fresh student has 26.3% lower median latency and meets the observed F1
budget. The tiny F1 increase is not evidence of a reliable accuracy improvement.
Its predictions differ from the Colab-trained student on 42 headlines; a fresh
training run is distinct from replaying the saved weights. Both GPU runs checked
all 9,216 timed predictions. Reports and checkpoints remain under
`results/replay-22013353` and `results/retrain-22013353` on Torch.

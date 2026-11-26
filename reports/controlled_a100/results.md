# Controlled A100 results — November 26, 2026

Compare each route with its matching full teacher. Positive F1 loss means lower
macro-F1; negative latency change means faster. CE uses gold labels; KD uses
75% gold-label CE plus 25% temperature-2 teacher KL when the teacher is correct,
and gold labels alone when it is wrong. All student parameters are trainable.

## Primary comparison (seed 4142, batch size one)

Median end-to-end latency includes tokenization, input transfer and CPU labels.
Values and percentage changes are the median of three session statistics; each
percentage change is computed against the full model in the same session.

| Dataset | Route | Macro-F1 | F1 loss (pp) | Median (ms) | Median change | Mean change | Test budget |
|---|---|---:|---:|---:|---:|---:|---|
| SEntFiN | Full 12 layers | 0.87896 | +0.00 | 5.368 | +0.0% | +0.0% | Pass |
| SEntFiN | Dynamic exit | 0.87510 | +0.39 | 5.518 | +2.8% | -8.9% | Pass |
| SEntFiN | CE 8-layer student | 0.87691 | +0.20 | 3.873 | -27.9% | -27.7% | Pass |
| SEntFiN | KD 8-layer student | 0.87918 | -0.02 | 3.871 | -27.9% | -27.7% | Pass |
| PhraseBank | Full 12 layers | 0.82327 | +0.00 | 6.091 | +0.0% | +0.0% | Pass |
| PhraseBank | Dynamic exit | 0.81727 | +0.60 | 6.596 | +8.5% | -11.3% | Pass |
| PhraseBank | CE 8-layer student | 0.81119 | +1.21 | 4.495 | -26.2% | -26.2% | **Fail** |
| PhraseBank | KD 8-layer student | 0.81524 | +0.80 | 4.484 | -26.4% | -26.3% | Pass |

**The primary KD student meets the observed 1 pp budget on both datasets.**
The primary PhraseBank CE student fails. Dynamic exit improves mean latency but
increases batch-one median latency: saved work on easy examples does not remove
the checking cost paid by examples that continue.

## All training seeds

Seeds 4142, 1 and 2 were fixed before this run; seed 4142 was the predefined
timing comparison. All twelve checkpoints are reported, including failures.
Validation eligibility uses a stricter 0.25 pp margin; test budget uses 1 pp.

| Dataset | Objective | Seed | Selected epoch | Test macro-F1 | Test F1 loss (pp) | Validation eligible | Test budget |
|---|---|---:|---:|---:|---:|---|---|
| SEntFiN | CE | 4142 | 5 | 0.87691 | +0.20 | Yes | Pass |
| SEntFiN | KD | 4142 | 4 | 0.87918 | -0.02 | Yes | Pass |
| SEntFiN | CE | 1 | 4 | 0.87996 | -0.10 | Yes | Pass |
| SEntFiN | KD | 1 | 5 | 0.88405 | -0.51 | Yes | Pass |
| SEntFiN | CE | 2 | 5 | 0.87673 | +0.22 | **No** | Pass |
| SEntFiN | KD | 2 | 2 | 0.87847 | +0.05 | **No** | Pass |
| PhraseBank | CE | 4142 | 4 | 0.81119 | +1.21 | Yes | **Fail** |
| PhraseBank | KD | 4142 | 3 | 0.81524 | +0.80 | Yes | Pass |
| PhraseBank | CE | 1 | 2 | 0.81418 | +0.91 | Yes | Pass |
| PhraseBank | KD | 1 | 2 | 0.81221 | +1.11 | Yes | **Fail** |
| PhraseBank | CE | 2 | 1 | 0.81386 | +0.94 | Yes | Pass |
| PhraseBank | KD | 2 | 5 | 0.82688 | -0.36 | Yes | Pass |

SEntFiN: all three seeds per objective pass the observed test budget, but both
seed-2 checkpoints fail validation eligibility. PhraseBank: each objective passes
the test budget in two of three seeds. KD improves mean F1 but is more variable
on PhraseBank and is worse than CE for seed 1. This is evidence of a promising
compression recipe, not a reliable accuracy gain or an all-seed guarantee.

## Batch-size comparison

Median end-to-end change relative to the full teacher in the same session.
The same seed-4142 checkpoints are used at every batch size; these are per-batch
latencies. See the CSVs for model-only timings, p95 and throughput.

| Dataset | Batch size | Dynamic | CE student | KD student |
|---|---:|---:|---:|---:|
| SEntFiN | 1 | +2.8% | -27.9% | -27.9% |
| SEntFiN | 4 | +6.0% | -25.1% | -25.1% |
| SEntFiN | 16 | +1.1% | -24.9% | -25.0% |
| SEntFiN | 32 | -4.3% | -26.2% | -26.4% |
| PhraseBank | 1 | +8.5% | -26.2% | -26.4% |
| PhraseBank | 4 | +13.9% | -24.5% | -24.6% |
| PhraseBank | 16 | -4.4% | -26.5% | -26.6% |
| PhraseBank | 32 | -14.2% | -28.4% | -28.3% |

The compact students reduce median latency across all four tested batch sizes.
Dynamic exit is workload-dependent: the batch-32 median improves on both
datasets, while small-batch medians can get worse.

## Where the overhead comes from

Forced-full-depth controls always run all twelve blocks and return the same
labels as full inference. Differences below are pooled paired median model-only
milliseconds: manual loop minus standard full call; intermediate head plus
confidence scoring minus loop; CPU confidence check minus GPU scoring; and
complete check path minus standard full call. Medians do not add algebraically.

| Dataset | Manual loop | Head and scoring | Host check | Net check-path cost |
|---|---:|---:|---:|---:|
| SEntFiN | -0.120 ms | +0.214 ms | +0.080 ms | +0.178 ms |
| PhraseBank | -0.142 ms | +0.445 ms | +0.169 ms | +0.470 ms |

These include dispatch and synchronization costs, not isolated GPU arithmetic.
SEntFiN checks layer 8 (threshold 0.90); PhraseBank checks layers 4 and 8
(threshold 0.92), using their existing calibrated temperatures. Dynamic routing
compacts unfinished examples and transfers confidence and labels together.

## Quality uncertainty

Group bootstrap: 2,000 resamples, seed 10143, 95% pointwise intervals for
macro-F1 loss in percentage points. These do not account for repeated development
on the same test sets. Intervals above 1 pp prevent a population-level budget
guarantee even when the observed point estimate passes.

| Dataset | Dynamic | CE student | KD student |
|---|---:|---:|---:|
| SEntFiN | [0.00, 0.77] | [-0.96, 1.34] | [-1.24, 1.18] |
| PhraseBank | [-0.09, 1.56] | [-0.81, 3.29] | [-1.63, 3.08] |

## Protocol and provenance

- NVIDIA A100-SXM4-80GB, driver 610.43.02; PyTorch 2.11.0+cu130,
  Transformers 4.48.3, Python 3.13.16. FP32 eager, TF32 off, one CPU thread,
  96-token cap. Training uses FP16 AMP, AdamW, five epochs and matched settings.
- Quality uses all 1,573 SEntFiN and 970 PhraseBank test examples. Timing uses
  512 unique examples, three sessions, five repeats per mode and 20 warmups.
  Repeated measurements are not independent classification examples.
- All 288,000 timed calls match reference labels and exit depths. Teacher and
  exit-head fingerprints are unchanged. No W&B logging occurs inside timed calls.
- SEntFiN comparisons finished in job 24682539 before later preemption. All
  PhraseBank comparisons were rerun together in completed job 24688746 using
  the exact original locked student weights. Partial PhraseBank attempts are
  excluded. Both datasets use the same GPU configuration, with separate allocations.
- The tests were used in prior development; no new test results select checkpoints,
  policies or the primary seed in this run. Failed candidates remain visible.

### Files

- [SEntFiN](raw/sentfin/) and [PhraseBank](raw/phrasebank/): `quality.csv`,
  `seed_quality.csv`, all predictions, raw timing CSVs, verified session summaries,
  training histories, selection records, environment and asset/source hashes.
- [Allocation record](raw/allocations.json), [completion record](raw/complete.json),
  [raw-file checksums](checksums.json) and [independent audit](../../tools/analyze_controlled.py).
- Exact executed source snapshots: [original](raw/source-original.tar.gz) and
  [PhraseBank replay](raw/source-replay.tar.gz). The replay added locked-checkpoint
  reuse after preemption. The published runner additionally removes a redundant
  missing-checkpoint check and clarifies its module description; algorithms and
  settings are unchanged. Dataset `source_sha256.json` identifies executed files.
- [W&B summary](https://wandb.ai/wk2376-efficient-ai/finbert-controlled-a100/runs/435ttxp3):
  16 finished runs; [cloud verification](raw/wandb_cloud_verification.json) confirms
  their result artifacts are committed. W&B team access is required.
- [Report PDF](Efficient_AI_Final_Report.pdf),
  [Overleaf](https://www.overleaf.com/project/6b07a79c4b26e8098a123ad3) and
  [frozen protocol](../../CONTROLLED_PROTOCOL.md).

Weights, input bundles and duplicated tokenizer copies are excluded from Git.
Their hashes and training records remain available for provenance.

# Financial PhraseBank: A100 comparison

NYU Torch job **23431781** completed successfully on November 17, 2026.
These existing results were recovered and independently verified on November 23;
no model retraining or additional GPU benchmark was needed.

## Results

Both routes used one **NVIDIA A100-SXM4-80GB**, batch size 1, FP32 eager,
TF32 disabled, and one CPU thread. End-to-end timing includes tokenization.

| Route | Macro-F1 | F1 drop (percentage points) | Mean | Median | p95 |
|---|---:|---:|---:|---:|---:|
| Full 12 layers | 0.82327 | 0.00 | 5.843 ms | 5.829 ms | 5.947 ms |
| Conservative early exit | 0.81727 | 0.60 | 4.852 ms | 6.254 ms | 6.466 ms |
| Latency change vs. full | | | **-17.0%** | **+7.3%** | **+8.7%** |

The observed F1 drop meets the 1-percentage-point budget. Early exit finishes
44.7% of test sentences before layer 12 and executes 9.05 layers on average.

Easy sentences save enough time to improve the mean. However, 55.3% of the
test set still reaches layer 12 and pays for the extra exit checks. In the
timing subset, 51.6% reaches layer 12, so the median and p95 become worse.
The median paired time differences for exits at layers 4, 8, and 12 are
-3.236, -1.290, and +0.531 ms respectively. This supports lower average
processing time for this configuration, but not better median or tail latency.

## Protocol and verification

- Quality uses all **970 test sentences**. Timing uses 512 test sentences,
  three repeats per route and mode, warmup, randomized route order, and CUDA
  synchronization. The saved output contains **6,144 timing records**.
- The teacher, exit heads, temperatures, and confidence policy remain frozen.
  The policy checks layers 4 and 8 with threshold **0.92**, selected on validation.
  The test split was evaluated in the earlier study and reused for this comparison.
- Sentiment training starts from `yiyanghkust/finbert-pretrain`. Exact and
  near-duplicate groups stay within one split: 2,903 train, 481 validation,
  479 calibration, and 970 test sentences. Unlabeled pretraining overlap could
  not be audited because the original corpus was unavailable.
- Slurm reports `COMPLETED` and exit code `0:0`. The frozen policy matches
  the Mac policy byte for byte. Both routes match all 970 Mac predictions and
  exit depths. Every timed prediction and exit depth matches its saved test output.
- Macro-F1, median, and p95 were recomputed from the saved records. Nine benchmark
  source-file checksums match the code on this branch when the results were verified.

This is one trained model and one paired A100 run. The paired group-bootstrap
95% interval for the test F1 drop is **[-0.097, 1.525] percentage points**.
The point estimate passes the budget; the interval does not establish a
population drop below 1 point. Compare latency within each hardware setup.

## Evidence and reproduction

- [Metrics](a100-23431781/metrics.csv), [predictions](a100-23431781/predictions.csv),
  and [raw timing records](a100-23431781/timings.csv).
- [Mac/A100 comparison](a100-23431781/a100_check.json),
  [independent metric checks](a100-23431781/independent_check.json), and
  [run verification](a100-23431781/verification.json).
- [Frozen policy](a100-23431781/frozen_policy.json),
  [experiment configuration](a100-23431781/config.json),
  [asset checksums](a100-23431781/asset_manifest.json), and
  [split audit](a100-23431781/split_audit.json).
- [Environment](a100-23431781/environment.json),
  [Slurm accounting](a100-23431781/slurm-accounting.txt),
  [original GPU log](a100-23431781/phrasebank-23431781.log), and
  [source checksums](a100-23431781/source-sha256.txt).

Use the [README reproduction instructions](../../README.md#reproduce-the-comparison)
with the private PhraseBank input bundle and the unchanged frozen policy.
The run used Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.48.3, and CUDA 13.0.

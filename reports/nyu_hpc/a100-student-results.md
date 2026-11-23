# FinBERT compact student on A100: replay, retraining, and ten seeds

All runs used the original Colab assets (checksums verified with `tools/prepare_assets.py`)
on NYU Cloud Bursting, one NVIDIA A100-SXM4-40GB per job (driver 610.43.02,
PyTorch 2.11.0+cu130, Transformers 4.48.3). Timing settings match the B200 run:
FP32 eager, batch size one, one CPU thread, 512 unique headlines, three repeats.
Test quality covers all 1,573 headlines.

## Saved-model replay and retraining (job 37400)

The saved Colab teacher, layer-8 dynamic exit, and compact student matched every
Colab test label and exit depth (zero differences for all three models).

Fresh student training (seed 4142) selected epoch 4 on validation (macro-F1 0.881816):

| Model | Test macro-F1 | F1 drop (pp) | Median end-to-end | vs full model |
|---|---:|---:|---:|---:|
| Full 12 layers | 0.878955 | — | 8.934 ms | — |
| Dynamic layer-8 exit | 0.875099 | 0.386 | 9.069 ms | +1.5% |
| Fresh compact student | 0.879177 | -0.022 | 6.358 ms | **-28.8%** |

The A100 retraining is **prediction-for-prediction identical** to the B200 retraining
(job 22013353): all 1,573 test labels and exit depths match. Both differ from the
Colab-trained student on the same 42 headlines. The dynamic exit's mean latency is
9.8% lower than the full model, but its median is 1.5% higher, as on the B200.

## Ten training seeds (job 37401; reruns 38177 and 41314)

Seeds were fixed before any results (`hpc/seeds.sbatch`); every seed is reported.
Seed 4142 reproduces job 37400 exactly. Each run times the student against the full
model on its own GPU.

| Seed | Epoch | Validation F1 | Test F1 | F1 drop (pp) | Full p50 | Student p50 | Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.8881 | 0.8840 | -0.509 | 9.153 ms | 6.585 ms | 28.1% |
| 2 | 2 | 0.8772 | 0.8785 | 0.049 | 8.940 ms | 6.440 ms | 28.0% |
| 3 | 5 | 0.8780 | 0.8786 | 0.032 | 8.977 ms | 6.382 ms | 28.9% |
| 4 | 3 | 0.8783 | 0.8752 | 0.377 | 8.731 ms | 6.206 ms | 28.9% |
| 5 | 4 | 0.8856 | 0.8800 | -0.102 | 8.928 ms | 6.406 ms | 28.3% |
| 6 | 4 | 0.8791 | 0.8769 | 0.210 | 8.966 ms | 6.371 ms | 28.9% |
| 7 | 5 | 0.8867 | 0.8759 | 0.303 | 8.924 ms | 6.429 ms | 28.0% |
| 8 | 5 | 0.8854 | 0.8781 | 0.083 | 9.363 ms | 6.734 ms | 28.1% |
| 9 | 3 | 0.8752 | 0.8744 | 0.455 | 8.998 ms | 6.416 ms | 28.7% |
| 4142 | 4 | 0.8818 | 0.8792 | -0.022 | 8.876 ms | 6.359 ms | 28.4% |

- Student test macro-F1: mean 0.8781, sd 0.0028 (full model 0.8790).
- F1 drop: mean 0.088 pp, sd 0.277; worst seed 0.455 pp. **10 of 10 within the 1 pp budget.**
- Median latency reduction: mean 28.4%, sd 0.4 (28.0–28.9%).

Seed spread measures training variation only, not test-set uncertainty. The test set
was reused during development, so the budget check is not a population guarantee.

## Comparison with the A100 early-exit study

The multi-exit study (job 29383) reached a 28.2% median reduction at batch size one
with a 0.80 pp F1 drop. The student gives a similar reduction with a smaller, stable
F1 drop. The exit study used the rebuilt teacher, not the Colab teacher, so this is
a same-hardware comparison rather than an identical-model one.

## Files

- `a100-replay-37400/`, `a100-retrain-37400/`: metrics, predictions, raw timings,
  environment, verification, and (retrain) history and selection records.
- `a100-seeds-37401/`: one folder per seed plus `seed_summary.csv` and `seed_summary.txt`.
- `a100-logs/`: Slurm logs. Seeds 6 and 9 ran as job 38177 and seed 8 as job 41314,
  copied into the same seed folder before summarizing.

Student checkpoints remain on Cloud Bursting under `results/` and are not in Git.

## Cloud Bursting notes

Several A100 jobs were cancelled by the system before training began (34417, 35134,
37401_6, 37401_8, 37401_9, 38177_8), and one was requeued and admin-held with
`user env retrieval failed` (37332). Two cancellations happened about two minutes after
submission, exactly on the hour. Rerunning the affected array tasks with
`--array=<ids>` completed them; no job failed with a code error.

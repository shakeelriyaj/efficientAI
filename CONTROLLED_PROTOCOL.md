# Controlled A100 comparison

Freeze this protocol before viewing any new test predictions. The existing test
splits were used during earlier development and are not fresh holdouts.

- Use an A100-SXM4-80GB configuration throughout. Pair all comparisons within
  one uninterrupted allocation per dataset (see the preemption amendment below).
  Record hardware, driver, package versions, source hashes and asset manifests.
- Keep the existing teacher and heads for each dataset. SEntFiN uses its original
  layer-8 head and threshold 0.90; PhraseBank uses its existing layer-4/8 heads,
  calibrated temperatures and threshold 0.92. Do not transfer thresholds or
  absolute latency values between different teachers.
- Train CE-only and teacher-correct-filtered KD students for seeds 4142, 1 and 2
  on both datasets. Same eight teacher blocks, optimizer, batches, five-epoch
  budget and validation checkpoint rule. Report every seed. Seed 4142 is the
  predefined primary timing comparison; do not choose the fastest or best test seed.
- Lock every checkpoint and compute validation quality before opening the test
  split for this run. No tuning or seed exclusion based on the new test results.
- Compare full, dynamic, CE student and KD student at batch sizes 1/4/16/32.
  Use FP32 eager, TF32 off, one CPU thread and a 96-token cap throughout.
  Measure model-only and end-to-end latency separately. Randomize route order
  within paired inputs: 512 examples, five repetitions, three timing sessions,
  and 20 warmup calls. Training and W&B activity stop during timing.
- At batch size one, compare full inference with the same twelve-block manual
  loop: no intermediate head; head and confidence computation on GPU; and head
  plus GPU-to-CPU confidence transfer. All three always continue to layer 12 and
  must return the same labels as full inference. This isolates implementation,
  head-computation and host-transfer overhead without changing backbone weights.
- Validation eligibility requires at most 0.25 pp macro-F1 loss. Test eligibility
  is an observed loss no greater than 1 pp relative to the matching full teacher.
  Preserve failing candidates in results; recommend full inference when a
  candidate fails. These point checks are not a statistical population guarantee.
- Report quality on every test example. Repeat timing is evidence about runtime
  variability, not additional independent evidence about classification quality.
- Record configuration, training curves, quality, timing and raw-result artifacts
  in wk2376-efficient-ai/finbert-controlled-a100. W&B runs can be recorded offline
  and synchronized after authentication; do not represent offline runs as uploaded.

## Preemption amendment

The original two-dataset allocation was preempted after SEntFiN completed and before PhraseBank timing completed. Preserve the completed SEntFiN measurements. Discard all partial PhraseBank timings from the main comparison and replay every PhraseBank route and batch size in one new A100-SXM4-80GB allocation. Reuse the exact locked student checkpoints, recording weight-file checksums; do not retrain, change seeds, tune policies or alter timing settings. Final latency comparisons are paired within each dataset allocation. Do not pool absolute times across allocations. Submit one dataset per job to finish within the preemption grace period.

"""Paired wall-clock timing of real inference, never cached model outputs."""
import time
import numpy as np
import pandas as pd
import torch


def encode(text, tokenizer, device, max_length):
    return dict(tokenizer(text, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device))


@torch.inference_mode()
def benchmark(routes, tokenizer, frame, cfg, device, predictions):
    if device.type != "cuda":
        raise ValueError("Use a CUDA GPU for the latency benchmark.")
    rng = np.random.default_rng(cfg["timing_seed"])
    chosen = frame.iloc[rng.choice(len(frame), min(cfg["latency_samples"], len(frame)), replace=False)]
    texts, row_ids = chosen.text.tolist(), chosen.row_id.astype(int).tolist()
    tokens = [encode(text, tokenizer, device, cfg["max_length"]) for text in texts]
    names = list(routes)
    for predict in routes.values():
        for index in range(cfg["warmup"]):
            predict(tokens[index % len(tokens)])
    records = []
    for mode in ("model", "e2e"):
        for repeat in range(cfg["timing_repeats"]):
            for index in rng.permutation(len(texts)):
                for route_index in rng.permutation(len(names)):
                    name = names[route_index]
                    torch.cuda.synchronize(device)
                    started = time.perf_counter_ns()
                    inputs = tokens[index] if mode == "model" else encode(
                        texts[index], tokenizer, device, cfg["max_length"])
                    label, depth = routes[name](inputs)
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    records.append({"name": name, "mode": mode, "repeat": repeat,
                                    "row_id": row_ids[index], "ms": elapsed,
                                    "prediction": label, "exit_layer": depth})
    raw = pd.DataFrame(records)
    for name in names:
        measured = raw[raw.name == name]
        expected = predictions.loc[measured.row_id]
        if not (np.array_equal(measured.prediction, expected[name + "_prediction"]) and
                np.array_equal(measured.exit_layer, expected[name + "_exit_layer"])):
            raise AssertionError(f"Timed prediction/depth mismatch: {name}")
    summaries = []
    for name in names:
        row = {"name": name, "unique_headlines": len(texts),
               "timed_predictions_per_mode": len(texts) * cfg["timing_repeats"]}
        for mode in ("model", "e2e"):
            values = raw.loc[(raw.name == name) & (raw["mode"] == mode), "ms"]
            row[mode + "_mean_ms"] = float(values.mean())
            row[mode + "_p50_ms"] = float(values.median())
            row[mode + "_p95_ms"] = float(np.percentile(values, 95))
        summaries.append(row)
    return pd.DataFrame(summaries), raw

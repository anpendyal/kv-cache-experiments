# KV Cache Scaling Analysis

## Dataset Size Estimate

There's no fast way to get the actual number of cases from the dataset itself, so I looked at the dataset on the HuggingFace website. It says there are 6,919,240 documents, but the description says "This dataset contains 6.7 million cases" and HuggingFace estimates 5,520,526 rows, so I just estimated that there were **7 million cases**.

---

## Granite 3.3 8B

Numbers from the longest running job:

```
2026-03-24 11:50:55 | INFO | PROGRESS cases=17850 (skipped: not_reporter=1910753, bad_id=54, missing_text=0, already_exists=762) | pt_total=13.75TB | kv_time_sum=57760.8s | wall=61666.5s
```

| Metric | Per Case | Extrapolated to 7M Cases |
|---|---|---|
| Storage | ~770 MB | ~5.4 PB |
| KV compute time | ~3.24s (GPU time) | ~262 days (single GPU) |
| Wall time | ~3.45s | ~280 days |

> **Note:** 5.4 PB total storage seems ridiculously large.

---

## Granite 4 Micro

```
2026-05-08 10:34:27 | INFO | PROGRESS cases=5650 (skipped: not_reporter=461032, bad_id=0, unresolvable_id=41, missing_text=0, max_tokens=21, already_exists=0) | pt_total=867.91GB | kv_time_sum=1638.2s | wall=1791.5s
```

| Metric | Per Case | Extrapolated to 7M Cases |
|---|---|---|
| Storage | ~153.6 MB | ~1.08 PB |
| KV compute time | ~0.29s (GPU time) | ~23.5 days (single GPU) |
| Wall time | ~0.32s | ~25.7 days |

---

## Comparison

| | Granite 3.3 8B | Granite 4 Micro |
|---|---|---|
| Storage per case | ~770 MB | ~153.6 MB |
| Total storage | ~5.4 PB | ~1.08 PB |
| GPU time (7M cases) | ~262 days | ~23.5 days |

Granite 4 Micro is **~5x smaller** per case in storage and **~11x faster** — largely because it is a much smaller model.
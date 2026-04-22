import argparse
import os
import sys
import time
import logging
import io
import json

import torch
from transformers import DynamicCache

from datasets import load_dataset

from mellea.backends.model_ids import IBM_GRANITE_3_3_8B
from mellea.backends.huggingface import LocalHFBackend

# Reporters for Massachusetts
REPORTERS = {
    "davis-l-ct-cas",
    "mass",
    "mass-app-ct",
    "mass-app-dec",
    "mass-app-div",
    "mass-app-div-annual",
    "mass-l-rptr",
    "mass-supp",
    "rec-co-ct",
    "rep-cont-el",
    "rep-cont-elect-case",
    "super-ct-jud",
}

MAX_ERROR_SAMPLES = 100  # number of cases to collect quantization error stats for

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("caselaw_ma_kv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def fmt_bytes(n: int) -> str:
    """
    Takes a file size in bytes (an integer) and turns it into a human-readable string
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{n}B"

def safe_getsize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def parse_id(file_id: str):
    """
    parse ids from the database
        formatted like reporter_volume/html/page-number.html
        ex) f2d_475/html/0775-01.html

    returns tuple of (reporter, volume, page)
    """
    if "_" not in file_id:
        return (None, None, None)

    reporter, rest = file_id.split("_", 1)          # "f2d", "475/html/0775-01.html"
    parts = rest.split("/", 2)                      # ["475", "html", "0775-01.html"]
    if len(parts) < 3:
        return (None, None, None)

    volume = parts[0]
    filename = parts[2]                             # "0775-01.html"
    if not filename.endswith(".html"):
        return (None, None, None)

    page = filename[:-5]                            # "0775-01"
    return (reporter, volume, page)

MAX_TOKENS = 17000  # Truncate to avoid OOM on very long documents

def _make_dc_cache(backend: LocalHFBackend, text: str, **model_options):
    text_bytes = len(text.encode("utf-8"))
    tokens = backend._tokenizer(text, return_tensors="pt", max_length=MAX_TOKENS, truncation=True)
    n_tokens = int(tokens["input_ids"].shape[1])

    dc = DynamicCache()
    with torch.no_grad():
        out = backend._model.model(
            tokens["input_ids"].to(backend._device),
            attention_mask=tokens["attention_mask"].to(backend._device),
            past_key_values=dc,
            **model_options,
        )
        result = out.past_key_values
        del out, dc

    del tokens
    torch.cuda.empty_cache()

    return result, n_tokens, text_bytes

def tensor_bytes(x):
    return x.numel() * x.element_size()

def cache_bytes(legacy):
    total = 0
    for layer in legacy:
        if isinstance(layer, (tuple, list)):
            for t in layer:
                if torch.is_tensor(t):
                    total += tensor_bytes(t)
        elif isinstance(layer, dict):
            for t in layer.values():
                if torch.is_tensor(t):
                    total += tensor_bytes(t)
        elif torch.is_tensor(layer):
            total += tensor_bytes(layer)
    return total

def gpu_mem():
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    return f"gpu_alloc={fmt_bytes(alloc)} gpu_reserved={fmt_bytes(reserv)}"


def quantize_int4(tensor: torch.Tensor):
    """
    Quantize a FP16/FP32 tensor to INT4, packed into INT8.
    Returns (packed_int8, scale, zero_point, orig_shape) needed to reconstruct.
    """
    orig_shape = tensor.shape
    t = tensor.float().flatten()

    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / 15.0          # 15 = 2^4 - 1, range of INT4
    zero_point = (-t_min / scale).round().clamp(0, 15).to(torch.uint8)

    quantized = ((t / scale) + zero_point).round().clamp(0, 15).to(torch.uint8)

    if quantized.numel() % 2 != 0:
        quantized = torch.cat([quantized, torch.zeros(1, dtype=torch.uint8)])
    packed = (quantized[0::2] << 4) | quantized[1::2]

    return packed, scale, zero_point, orig_shape


def dequantize_int4(packed, scale, zero_point, orig_shape):
    """
    Reconstruct FP16 tensor from INT4-packed INT8.
    """
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    quantized = torch.stack([high, low], dim=1).flatten().float()

    n = 1
    for s in orig_shape:
        n *= s
    quantized = quantized[:n]

    return ((quantized - zero_point.float()) * scale).reshape(orig_shape).half()


def _save_dc_cache(
    backend: LocalHFBackend,
    text: str,
    out_path: str,
    error_samples: list | None = None,
    **model_options,
):
    result, n_tokens, text_bytes = _make_dc_cache(backend, text, **model_options)

    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    # Step 1: move to CPU (frees GPU memory)
    result.key_cache = [k.cpu() for k in result.key_cache]
    result.value_cache = [v.cpu() for v in result.value_cache]
    torch.cuda.empty_cache()

    # Step 2: quantize to INT4, and optionally collect error stats for this case
    collect_errors = error_samples is not None and len(error_samples) < MAX_ERROR_SAMPLES
    case_errors = []

    quantized_keys = []
    for layer_idx, k in enumerate(result.key_cache):
        packed, scale, zp, shape = quantize_int4(k)
        if collect_errors:
            reconstructed = dequantize_int4(packed, scale, zp, shape)
            mae = float((k.float() - reconstructed.float()).abs().mean())
            rel_err = float(mae / k.float().abs().mean())
            case_errors.append({"layer": layer_idx, "type": "key", "mae": mae, "rel_err_pct": rel_err * 100})
        quantized_keys.append((packed, scale, zp, shape))

    quantized_values = []
    for layer_idx, v in enumerate(result.value_cache):
        packed, scale, zp, shape = quantize_int4(v)
        if collect_errors:
            reconstructed = dequantize_int4(packed, scale, zp, shape)
            mae = float((v.float() - reconstructed.float()).abs().mean())
            rel_err = float(mae / v.float().abs().mean())
            case_errors.append({"layer": layer_idx, "type": "value", "mae": mae, "rel_err_pct": rel_err * 100})
        quantized_values.append((packed, scale, zp, shape))

    # Append this case's per-layer errors to the shared list
    if collect_errors and case_errors:
        error_samples.append(case_errors)

    result.key_cache = quantized_keys
    result.value_cache = quantized_values

    legacy = result.to_legacy_cache()
    kv_bytes = cache_bytes(legacy)

    buf = io.BytesIO()
    torch.save(legacy, buf)
    pt_size = buf.tell()
    del buf

    del legacy
    result.key_cache.clear()
    result.value_cache.clear()
    del result
    torch.cuda.empty_cache()

    return kv_bytes, n_tokens, text_bytes, pt_size


def write_error_summary(error_samples: list, log_path: str):
    """
    Aggregate per-layer error stats across all sampled cases and write to JSON.
    error_samples is a list of cases, each case is a list of per-layer dicts.
    """
    all_rel_errs = [e["rel_err_pct"] for case in error_samples for e in case]
    all_maes = [e["mae"] for case in error_samples for e in case]

    summary = {
        "n_cases_sampled": len(error_samples),
        "n_layer_measurements": len(all_rel_errs),
        "rel_err_pct": {
            "mean": sum(all_rel_errs) / len(all_rel_errs),
            "min": min(all_rel_errs),
            "max": max(all_rel_errs),
            "std": float(torch.tensor(all_rel_errs).std().item()),
        },
        "mae": {
            "mean": sum(all_maes) / len(all_maes),
            "min": min(all_maes),
            "max": max(all_maes),
        },
        "per_case_samples": error_samples,
    }

    error_path = os.path.join(os.path.dirname(log_path), "quantization_errors.json")
    with open(error_path, "w") as f:
        json.dump(summary, f, indent=2)


def build_kv_from_dataset(out_root: str, log_path: str, limit: int | None = None):
    logger = setup_logger(log_path)
    out_root = os.path.abspath(out_root.rstrip("/"))
    os.makedirs(out_root, exist_ok=True)

    backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B)

    dataset = load_dataset("common-pile/caselaw_access_project", split="train", streaming=True)

    total_cases = 0
    total_pt_bytes = 0
    total_kv_time = 0.0

    skipped_not_reporter = 0
    skipped_bad_id = 0
    skipped_missing_text = 0
    skipped_already_exists = 0

    # Collect quantization error samples across cases (up to MAX_ERROR_SAMPLES)
    error_samples = []

    job_id = os.environ.get("LSB_JOBID", "local")
    bad_id_path = os.path.join(os.path.dirname(log_path), f"bad_id_{job_id}.txt")

    wall_start = time.time()

    for case in dataset:
        if limit is not None and total_cases >= limit:
            break

        file_id = case.get("id")
        text = case.get("text")

        if not file_id:
            skipped_bad_id += 1
            with open(bad_id_path, "a", encoding="utf-8") as f:
                f.write("<missing_id>\n")
            continue
        if text is None:
            skipped_missing_text += 1
            with open(bad_id_path, "a", encoding="utf-8") as f:
                f.write(f"{file_id}: MISSING TEXT\n")
            continue

        reporter, volume, page = parse_id(file_id)
        if reporter is None:
            with open(bad_id_path, "a", encoding="utf-8") as f:
                f.write(f"{file_id}\n")
            skipped_bad_id += 1
            continue

        if reporter not in REPORTERS:
            skipped_not_reporter += 1
            continue

        out_dir = os.path.join(out_root, reporter, volume)
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, page + ".pt")

        if os.path.exists(pt_path):
            skipped_already_exists += 1
            continue

        t0 = time.perf_counter()

        kv_bytes, n_tokens, text_bytes, pt_size = _save_dc_cache(
            backend, text, pt_path, error_samples=error_samples
        )

        # Write error file incrementally every time a new sample is collected
        if len(error_samples) <= MAX_ERROR_SAMPLES and error_samples:
            write_error_summary(error_samples, log_path)

        dt = time.perf_counter() - t0

        total_cases += 1
        total_kv_time += dt
        total_pt_bytes += pt_size

        logger.info(
            f"case={total_cases} | id={file_id} | n_tokens={n_tokens} | text={fmt_bytes(text_bytes)} | "
            f"kv_est={fmt_bytes(kv_bytes)} | pt={fmt_bytes(pt_size)} | kv_time={dt:.3f}s | {gpu_mem()}"
        )

        if total_cases % 50 == 0:
            wall_elapsed = time.time() - wall_start
            logger.info(
                f"PROGRESS cases={total_cases} "
                f"(skipped: not_reporter={skipped_not_reporter}, bad_id={skipped_bad_id}, missing_text={skipped_missing_text}, already_exists={skipped_already_exists}) | "
                f"pt_total={fmt_bytes(total_pt_bytes)} | kv_time_sum={total_kv_time:.1f}s | wall={wall_elapsed:.1f}s"
            )


    wall_elapsed = time.time() - wall_start
    logger.info("==== FINAL TOTALS ====")
    logger.info(f"cases_processed={total_cases}")
    logger.info(f"skipped_not_reporter={skipped_not_reporter}")
    logger.info(f"skipped_bad_id={skipped_bad_id}")
    logger.info(f"skipped_missing_text={skipped_missing_text}")
    logger.info(f"skipped_already_exists={skipped_already_exists}")
    logger.info(f"total_pt_size={fmt_bytes(total_pt_bytes)}")
    logger.info(f"total_kv_time_sum={total_kv_time:.3f}s")
    logger.info(f"total_wall_time={wall_elapsed:.3f}s")
    logger.info(f"out_root={out_root}")

def parse_args():
    p = argparse.ArgumentParser(description="Build KV caches (.pt) for selected reporters from case.law dataset.")
    p.add_argument("--out_root", required=True, help="Output root for KV cache files.")
    p.add_argument("--log", required=True, help="Log file path.")
    p.add_argument("--limit", type=int, default=None, help="Optional max cases to process.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_kv_from_dataset(args.out_root, args.log, limit=args.limit)
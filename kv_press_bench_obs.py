"""
Benchmarks KV cache size before and after ObservedAttentionPress compression
for cases streamed from the Caselaw Access Project dataset.

Runs two forward passes per case (prefill only, no generation):
  1. Uncompressed — baseline KV cache size
  2. ObservedAttentionPress — compressed KV cache size

Requires attn_implementation="eager" for ObservedAttentionPress.
"""

import argparse
import logging
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/dccstor/nathan-ckpts/anooshka/kvpress")
from kvpress import ObservedAttentionPress

from sort_reporters_to_jurisdiction import load_reporters_for_jurisdiction


MODEL_ID = "ibm-granite/granite-4.0-micro"


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("kv_press_bench_obs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{n}B"


def tensor_bytes(x):
    return x.numel() * x.element_size()


def cache_bytes(cache):
    total = 0
    for layer in cache.layers:
        if torch.is_tensor(layer.keys):
            total += tensor_bytes(layer.keys)
        if torch.is_tensor(layer.values):
            total += tensor_bytes(layer.values)
    return total


def move_cache_to_cpu(cache):
    for layer in cache.layers:
        if torch.is_tensor(layer.keys):
            layer.keys = layer.keys.cpu()
        if torch.is_tensor(layer.values):
            layer.values = layer.values.cpu()


def parse_id(file_id):
    if "_" not in file_id:
        return ("malformed", None, None, None)
    reporter, rest = file_id.split("_", 1)
    parts = rest.split("/", 2)
    if len(parts) < 3:
        return ("malformed", None, None, None)
    volume = parts[0]
    filename = parts[2]
    if not filename.endswith(".html"):
        return ("unresolvable", None, None, None)
    page = filename[:-5]
    return ("ok", reporter, volume, page)


def bench_case(model, tokenizer, device, text, max_tokens, press):
    if max_tokens is not None:
        tokens = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True)
    else:
        tokens = tokenizer(text, return_tensors="pt")

    n_tokens = int(tokens["input_ids"].shape[1])
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    del tokens

    # Uncompressed pass
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.model(input_ids, attention_mask=attention_mask, past_key_values=DynamicCache(), use_cache=True)
    uncompressed_time = time.perf_counter() - t0
    move_cache_to_cpu(out.past_key_values)
    torch.cuda.empty_cache()
    uncompressed_bytes = cache_bytes(out.past_key_values)
    del out
    torch.cuda.empty_cache()

    # Compressed pass
    t0 = time.perf_counter()
    with torch.no_grad(), press(model):
        out = model.model(input_ids, attention_mask=attention_mask, past_key_values=DynamicCache(), use_cache=True)
    compressed_time = time.perf_counter() - t0
    move_cache_to_cpu(out.past_key_values)
    torch.cuda.empty_cache()
    compressed_bytes = cache_bytes(out.past_key_values)
    del out
    torch.cuda.empty_cache()

    del input_ids, attention_mask

    return n_tokens, uncompressed_bytes, compressed_bytes, uncompressed_time, compressed_time


def run(reporters, logs_dir, max_tokens, compression_ratio, limit):
    job_id = os.environ.get("LSB_JOBID", "local")
    job_dir = os.path.join(logs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    log_path = os.path.join(job_dir, f"{job_id}_obs_bench.log")
    sys.stderr = open(os.path.join(job_dir, f"{job_id}_obs_bench_error.txt"), "a", encoding="utf-8")
    logger = setup_logger(log_path)

    logger.info(f"press=ObservedAttentionPress | compression_ratio={compression_ratio} | max_tokens={max_tokens}")

    logger.info(f"Loading model {MODEL_ID!r} with attn_implementation=eager...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    logger.info("Model loaded.")

    press = ObservedAttentionPress(compression_ratio=compression_ratio)
    dataset = load_dataset("common-pile/caselaw_access_project", split="train", streaming=True)

    total = 0
    skipped_not_reporter = 0
    skipped_bad_id = 0
    skipped_missing_text = 0
    wall_start = time.time()

    for case in dataset:
        if limit is not None and total >= limit:
            break

        file_id = case.get("id", "")
        text = case.get("text")

        if not file_id:
            skipped_bad_id += 1
            continue
        if text is None:
            skipped_missing_text += 1
            continue

        status, reporter, _, _ = parse_id(file_id)
        if status != "ok":
            skipped_bad_id += 1
            continue
        if reporter not in reporters:
            skipped_not_reporter += 1
            continue

        n_tokens, unc_bytes, cmp_bytes, unc_time, cmp_time = bench_case(
            model, tokenizer, device, text, max_tokens, press
        )
        actual_ratio = 1.0 - (cmp_bytes / unc_bytes) if unc_bytes > 0 else 0.0
        total += 1

        logger.info(
            f"case={total} | id={file_id} | n_tokens={n_tokens} | "
            f"uncompressed={fmt_bytes(unc_bytes)} | compressed={fmt_bytes(cmp_bytes)} | "
            f"actual_ratio={actual_ratio:.3f} | unc_time={unc_time:.3f}s | cmp_time={cmp_time:.3f}s"
        )

        if total % 50 == 0:
            wall_elapsed = time.time() - wall_start
            logger.info(
                f"PROGRESS cases={total} | wall={wall_elapsed:.1f}s | "
                f"skipped: not_reporter={skipped_not_reporter} bad_id={skipped_bad_id} missing_text={skipped_missing_text}"
            )

    wall_elapsed = time.time() - wall_start
    logger.info("==== FINAL TOTALS ====")
    logger.info(f"cases_processed={total}")
    logger.info(f"skipped_not_reporter={skipped_not_reporter}")
    logger.info(f"skipped_bad_id={skipped_bad_id}")
    logger.info(f"skipped_missing_text={skipped_missing_text}")
    logger.info(f"total_wall_time={wall_elapsed:.3f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jurisdiction", required=True)
    p.add_argument("--reporters_metadata", default="ReportersMetadata.json")
    p.add_argument("--logs_dir", required=True)
    p.add_argument("--compression_ratio", type=float, default=0.5)
    p.add_argument("--max_tokens", type=int, default=17000)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reporters = load_reporters_for_jurisdiction(args.reporters_metadata, args.jurisdiction)
    run(reporters, args.logs_dir, args.max_tokens, args.compression_ratio, args.limit)

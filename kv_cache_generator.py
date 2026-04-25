import argparse
import os
import sys
import time
import logging
import io

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
    # human-ish sizes, but still precise enough
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        # Stop when the number is under 1024 (so it fits that unit), or when you hit the last unit (TB) so you don’t run off the end
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
    # malformed id
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

MAX_TOKENS = 8192  # Truncate to avoid OOM on very long documents

def _make_dc_cache(backend: LocalHFBackend, text: str, **model_options):
    text_bytes = len(text.encode("utf-8"))
    # tokens = backend._tokenizer(text, return_tensors="pt", max_length=MAX_TOKENS, truncation=True)
    tokens = backend._tokenizer(text, return_tensors="pt")
    n_tokens = int(tokens["input_ids"].shape[1])

    dc = DynamicCache()
    with torch.no_grad():
        out = backend._model.model( # fix: call base transfomer, skip LM head
            tokens["input_ids"].to(backend._device),
            attention_mask=tokens["attention_mask"].to(backend._device),
            past_key_values=dc,
            **model_options,
        )
        result = out.past_key_values
        del out, dc # free logits/hidden states immediately

    # explicitly delete the GPU tensors
    del tokens
    torch.cuda.empty_cache()

    return result, n_tokens, text_bytes

def tensor_bytes(x):
    return x.numel() * x.element_size()

def cache_bytes(legacy):
    total = 0
    for layer in legacy:
        # layer might be (k, v) or a dict
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


def _save_dc_cache(backend: LocalHFBackend, text: str, out_path: str, **model_options):
    result, n_tokens, text_bytes = _make_dc_cache(backend, text, **model_options)

    # Ensure parent directory exists
    out_dir = os.path.dirname(out_path)

    # Only try to create directories when there actually is a directory component
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    # Move KV tensors to CPU before legacy conversion so GPU memory is freed
    # before serialization (torch.save on GPU tensors pins extra CPU staging buffers)
    # this fixed the OOM
    result.key_cache = [k.cpu() for k in result.key_cache]
    result.value_cache = [v.cpu() for v in result.value_cache]
    torch.cuda.empty_cache()

    legacy = result.to_legacy_cache()
    kv_bytes = cache_bytes(legacy)

    with open(out_path, "wb") as ofh:
        torch.save(legacy, ofh)

    # Release GPU memory now that the cache is saved and metrics are captured
    del legacy
    # explicitly clear the DynamicCache tensors
    # manually clear DynamicCache internals instead of result.reset()
    result.key_cache.clear()
    result.value_cache.clear()
    del result
    torch.cuda.empty_cache()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    return kv_bytes, n_tokens, text_bytes


def build_kv_from_dataset(out_root: str, log_path: str, limit: int | None = None):
    logger = setup_logger(log_path)
    out_root = os.path.abspath(out_root.rstrip("/"))
    # supress error if directory already exists
    os.makedirs(out_root, exist_ok=True)

    backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B)

    # streaming=True so full dataset is not downloaded immediately
    dataset = load_dataset("common-pile/caselaw_access_project", split="train", streaming=True)

    total_cases = 0
    total_pt_bytes = 0
    total_kv_time = 0.0

    skipped_not_reporter = 0
    skipped_bad_id = 0
    skipped_missing_text = 0
    skipped_already_exists = 0

    wall_start = time.time()

    for case in dataset:
        if limit is not None and total_cases >= limit:
            break

        file_id = case.get("id")
        text = case.get("text")

        if not file_id:
            skipped_bad_id += 1
            continue
        if text is None:
            skipped_missing_text += 1
            continue

        reporter, volume, page = parse_id(file_id)
        # only need to check if reporter is None because
        # malformed ids will return (None, None, None)
        if reporter is None:
            skipped_bad_id += 1
            continue

        if reporter not in REPORTERS:
            skipped_not_reporter += 1
            continue

        # Folder per reporter and volume
        out_dir = os.path.join(out_root, reporter, volume)
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, page + ".pt")

        if os.path.exists(pt_path):
            skipped_already_exists += 1
            continue

        # Time for KV creation for this case
        t0 = time.perf_counter()

        kv_bytes, n_tokens, text_bytes = _save_dc_cache(backend, text, pt_path)
        
        dt = time.perf_counter() - t0

        pt_size = safe_getsize(pt_path)

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
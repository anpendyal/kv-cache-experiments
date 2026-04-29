"""
Generates or benchmarks KV caches for a given jurisdiction from the Caselaw
Access Project dataset.

In generate mode (default), KV caches are written to disk as .pt files.
In benchmark mode (--benchmark), serialization size is measured in memory
without writing to disk.

Pass --max_tokens N to truncate inputs and avoid OOM on very long documents.
"""

import argparse
import io
import logging
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import DynamicCache

from mellea.backends.huggingface import LocalHFBackend

from sort_reporters_to_jurisdiction import load_reporters_for_jurisdiction


def setup_log_paths(logs_dir: str, job_id: str) -> dict[str, str]:
    """
    Creates logs/{job_id}/ and returns a dict of all log file paths for this job.
    File naming convention: {job_id}_{log_type}
    """
    job_dir = os.path.join(logs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return {
        "actual":               os.path.join(job_dir, f"{job_id}.log"),
        "skipped_id":           os.path.join(job_dir, f"{job_id}_skipped_id.txt"),
        "skipped_max_tokens":   os.path.join(job_dir, f"{job_id}_skipped_max_tokens.txt"),
        "skipped_missing_text": os.path.join(job_dir, f"{job_id}_skipped_missing_text.txt"),
        "error":                os.path.join(job_dir, f"{job_id}_error.txt"),
    }


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("caselaw_kv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Log to file and stdout simultaneously
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


def parse_id(file_id: str) -> tuple[str, str | None, str | None, str | None]:
    """
    Parse a case ID formatted as reporter_volume/html/page.html.
    Example: f2d_475/html/0775-01.html -> ("ok", "f2d", "475", "0775-01")

    Returns a (status, reporter, volume, page) tuple where status is one of:
    - "ok": ID parsed successfully
    - "malformed": ID structure could not be parsed
    - "unresolvable": filename was truncated at a '?' (e.g. 000?-01.html was
      stored as 000 in the dataset because '?' is a URL query string separator)
    """
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


def _make_dc_cache(backend: LocalHFBackend, text: str, max_tokens: int | None, **model_options):
    text_bytes = len(text.encode("utf-8"))

    if max_tokens is not None:
        tokens = backend._tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True)
    else:
        tokens = backend._tokenizer(text, return_tensors="pt")

    n_tokens = int(tokens["input_ids"].shape[1])

    dc = DynamicCache()
    with torch.no_grad():
        # Call the base transformer directly to skip the LM head and avoid allocating the full vocabulary logits tensor
        out = backend._model.model(
            tokens["input_ids"].to(backend._device),
            attention_mask=tokens["attention_mask"].to(backend._device),
            past_key_values=dc,
            **model_options,
        )
        result = out.past_key_values
        del out, dc # free logits/hidden states immediately to reclaim GPU memory

    # Explicitly delete the GPU tensors
    del tokens
    torch.cuda.empty_cache()

    return result, n_tokens, text_bytes


def tensor_bytes(x):
    return x.numel() * x.element_size()


def cache_bytes(legacy):
    # legacy cache is a tuple of per-layer entries; each entry can be a
    # (key, value) tuple, a dict, or occasionally a bare tensor depending
    # on the model and transformers version
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


def _save_dc_cache(
    backend: LocalHFBackend,
    text: str,
    out_path: str,
    benchmark: bool,
    max_tokens: int | None,
    **model_options,
):
    result, n_tokens, text_bytes = _make_dc_cache(backend, text, max_tokens, **model_options)

    # Move KV tensors to CPU before legacy conversion so GPU memory is freed
    # before serialization (torch.save on GPU tensors pins extra CPU staging buffers).
    # This fixed the OOM
    result.key_cache = [k.cpu() for k in result.key_cache]
    result.value_cache = [v.cpu() for v in result.value_cache]
    torch.cuda.empty_cache()

    legacy = result.to_legacy_cache()
    kv_bytes = cache_bytes(legacy)

    if benchmark:
        # Serialize to an in-memory buffer to measure .pt size without touching disk
        buf = io.BytesIO()
        torch.save(legacy, buf)
        pt_size = buf.tell()
        del buf
    else:
        # Ensure parent directory exists
        out_dir = os.path.dirname(out_path)

        # Only try to create directories when there actually is a directory component
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as ofh:
            torch.save(legacy, ofh)
        pt_size = safe_getsize(out_path)

    # Explicitly clear DynamicCache internals to release CPU memory
    del legacy
    result.key_cache.clear()
    result.value_cache.clear()
    del result
    torch.cuda.empty_cache()

    return kv_bytes, n_tokens, text_bytes, pt_size


def build_kv_from_dataset(
    out_root: str,
    logs_dir: str,
    reporters: set,
    benchmark: bool,
    max_tokens: int | None,
    limit: int | None = None,
):
    # TODO: Check this
    # LSB_JOBID is set by LSF on CCC; fall back to "local" when running outside a job
    job_id = os.environ.get("LSB_JOBID", "local")
    log_paths = setup_log_paths(logs_dir, job_id)

    # Redirect stderr so uncaught exceptions and warnings go to the error log
    sys.stderr = open(log_paths["error"], "a", encoding="utf-8")

    logger = setup_logger(log_paths["actual"])
    logger.info(f"mode={'benchmark' if benchmark else 'generate'} | max_tokens={max_tokens} | reporters={sorted(reporters)}")

    out_root = os.path.abspath(out_root.rstrip("/"))
    os.makedirs(out_root, exist_ok=True)

    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
    # streaming=True avoids downloading the full dataset upfront
    dataset = load_dataset("common-pile/caselaw_access_project", split="train", streaming=True)

    total_cases = 0
    total_pt_bytes = 0
    total_kv_time = 0.0

    skipped_not_reporter = 0
    skipped_bad_id = 0
    skipped_unresolvable_id = 0
    skipped_missing_text = 0
    skipped_max_tokens = 0
    skipped_already_exists = 0

    wall_start = time.time()

    for case in dataset:
        if limit is not None and total_cases >= limit:
            break

        file_id = case.get("id")
        text = case.get("text")

        if not file_id:
            skipped_bad_id += 1
            with open(log_paths["skipped_id"], "a", encoding="utf-8") as f:
                f.write("<missing_id>\n")
            continue

        if text is None:
            skipped_missing_text += 1
            with open(log_paths["skipped_missing_text"], "a", encoding="utf-8") as f:
                f.write(f"{file_id}\n")
            continue

        status, reporter, volume, page = parse_id(file_id)
        if status == "malformed":
            skipped_bad_id += 1
            with open(log_paths["skipped_id"], "a", encoding="utf-8") as f:
                f.write(f"{file_id}: MALFORMED\n")
            continue
        if status == "unresolvable":
            skipped_unresolvable_id += 1
            with open(log_paths["skipped_id"], "a", encoding="utf-8") as f:
                f.write(f"{file_id}: UNRESOLVABLE\n")
            continue

        if reporter not in reporters:
            skipped_not_reporter += 1
            continue

        # Folder per reporter and volume
        out_dir = os.path.join(out_root, reporter, volume)
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, page + ".pt")

        # In benchmark mode, we never write .pt files to disk, so the file can never already exist
        if not benchmark and os.path.exists(pt_path):
            skipped_already_exists += 1
            continue

        # Time for KV creation for this case
        t0 = time.perf_counter()
        kv_bytes, n_tokens, text_bytes, pt_size = _save_dc_cache(
            backend, text, pt_path, benchmark=benchmark, max_tokens=max_tokens
        )
        dt = time.perf_counter() - t0

        # n_tokens == max_tokens means the document was longer and got truncated
        if max_tokens is not None and n_tokens == max_tokens:
            skipped_max_tokens += 1
            with open(log_paths["skipped_max_tokens"], "a", encoding="utf-8") as f:
                f.write(
                    f"id={file_id} | n_tokens={n_tokens} (truncated) | "
                    f"text={fmt_bytes(text_bytes)} | kv_est={fmt_bytes(kv_bytes)} | "
                    f"pt={fmt_bytes(pt_size)}\n"
                )

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
                f"(skipped: not_reporter={skipped_not_reporter}, bad_id={skipped_bad_id}, "
                f"unresolvable_id={skipped_unresolvable_id}, missing_text={skipped_missing_text}, "
                f"max_tokens={skipped_max_tokens}, already_exists={skipped_already_exists}) | "
                f"pt_total={fmt_bytes(total_pt_bytes)} | kv_time_sum={total_kv_time:.1f}s | wall={wall_elapsed:.1f}s"
            )

    wall_elapsed = time.time() - wall_start
    logger.info("==== FINAL TOTALS ====")
    logger.info(f"cases_processed={total_cases}")
    logger.info(f"skipped_not_reporter={skipped_not_reporter}")
    logger.info(f"skipped_bad_id={skipped_bad_id}")
    logger.info(f"skipped_unresolvable_id={skipped_unresolvable_id}")
    logger.info(f"skipped_missing_text={skipped_missing_text}")
    logger.info(f"skipped_max_tokens={skipped_max_tokens}")
    logger.info(f"skipped_already_exists={skipped_already_exists}")
    logger.info(f"total_pt_size={fmt_bytes(total_pt_bytes)}")
    logger.info(f"total_kv_time_sum={total_kv_time:.3f}s")
    logger.info(f"total_wall_time={wall_elapsed:.3f}s")
    logger.info(f"out_root={out_root}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate or benchmark KV caches for a jurisdiction from the Caselaw Access Project dataset."
    )
    p.add_argument("--jurisdiction", required=True,
                   help="Jurisdiction name as it appears in ReportersMetadata.json (e.g. 'Mass.').")
    p.add_argument("--reporters_metadata", default="ReportersMetadata.json",
                   help="Path to ReportersMetadata.json.")
    p.add_argument("--out_root", required=True,
                   help="Output root directory for KV cache files.")
    p.add_argument("--logs_dir", required=True,
                   help="Directory under which a per-job subfolder will be created for all log files.")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure serialized KV cache size without writing .pt files to disk.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help=(
                       "Truncate inputs to this many tokens to avoid OOM on long documents. "
                       "Recommended: 17000 (based on A100 80GB profiling: ~312KB/token KV cache, "
                       "peak GPU ~35.6GB at 17k tokens vs 79.25GB available; only ~1%% of cases exceed this limit)."
                   ))
    p.add_argument("--limit", type=int, default=None,
                   help="Optional max number of cases to process.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reporters = load_reporters_for_jurisdiction(args.reporters_metadata, args.jurisdiction)
    build_kv_from_dataset(
        args.out_root,
        args.logs_dir,
        reporters,
        benchmark=args.benchmark,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )
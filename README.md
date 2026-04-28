# KV Cache Generation for Caselaw Access Project

Generates or benchmarks KV caches for legal cases from the [Caselaw Access Project](https://case.law/) using IBM Granite 4.0 Micro. Cases are sourced from the `common-pile/caselaw_access_project` HuggingFace dataset, filtered to reporters for a given jurisdiction.

## Scripts

### `kv_cache.py`
Main script. Supports both generation and benchmarking via the `--benchmark` flag.

- **Generate mode** (default): writes KV caches to disk as `.pt` files under `out_root/reporter/volume/page.pt`
- **Benchmark mode** (`--benchmark`): serializes to an in-memory buffer to measure `.pt` file size without writing to disk

```
usage: kv_cache.py [-h] --jurisdiction JURISDICTION
                   [--reporters_metadata REPORTERS_METADATA]
                   --out_root OUT_ROOT --logs_dir LOGS_DIR
                   [--benchmark] [--max_tokens MAX_TOKENS] [--limit LIMIT]
```

| Flag | Description |
|------|-------------|
| `--jurisdiction` | Jurisdiction name as it appears in `ReportersMetadata.json` (e.g. `"Mass."`) |
| `--reporters_metadata` | Path to `ReportersMetadata.json` (default: `ReportersMetadata.json`) |
| `--out_root` | Output root directory for `.pt` files |
| `--logs_dir` | Directory under which a per-job subfolder is created for all log files |
| `--benchmark` | Measure serialized size only; do not write `.pt` files |
| `--max_tokens` | Truncate inputs to this many tokens to avoid OOM on long documents (recommended: `17000`) |
| `--limit` | Optional cap on the number of cases to process |

**Why `--max_tokens 17000`**: KV cache rate on A100 80GB is ~312 KB/token. At 17k tokens, peak GPU usage is ~35.6 GB — well within the 79.25 GB limit. Only ~1% of Massachusetts cases exceed this threshold.

### `sort_reporters_to_jurisdiction.py`
Groups reporter slugs by jurisdiction given a `ReportersMetadata.json`-style input. Reporters belonging to exactly one jurisdiction are filed under that jurisdiction's name; reporters spanning multiple jurisdictions go under `"Other"`. Includes unit tests — run with:

```bash
python sort_reporters_to_jurisdiction.py
```

### `get_num_cases.py`
Reads `JurisdictionsMetadata.json` and prints the case count per jurisdiction and the total.

## Log Structure

All log files are written to `{logs_dir}/{job_id}/`:

| File | Contents |
|------|----------|
| `{job_id}.log` | Main run log (one line per processed case) |
| `{job_id}_skipped_id.txt` | MALFORMED and UNRESOLVABLE case IDs |
| `{job_id}_skipped_max_tokens.txt` | Cases truncated at `--max_tokens` |
| `{job_id}_skipped_missing_text.txt` | Cases with no text field |
| `{job_id}_error.txt` | Redirected stderr (uncaught exceptions, warnings) |

`job_id` is set to `$LSB_JOBID` on CCC and `"local"` otherwise.

## Massachusetts Reporters

The following reporters are targeted:

| Slug | Description |
|------|-------------|
| `mass` | Massachusetts Reports |
| `mass-app-ct` | Massachusetts Appeals Court |
| `mass-app-dec` | Massachusetts Appellate Decisions |
| `mass-app-div` | Massachusetts Appellate Division |
| `mass-app-div-annual` | Massachusetts Appellate Division Annual |
| `mass-l-rptr` | Massachusetts Law Reporter |
| `mass-supp` | Massachusetts Supplement |
| `davis-l-ct-cas` | Davis Law Court Cases |
| `rec-co-ct` | Record of County Court |
| `rep-cont-el` | Reports of Contested Elections |
| `rep-cont-elect-case` | Reports of Contested Election Cases |
| `super-ct-jud` | Superior Court Judgments |

## Requirements

- Python 3.10+
- PyTorch
- `transformers`
- `datasets`
- `mellea` (internal — provides `LocalHFBackend` and `IBM_GRANITE_4_HYBRID_MICRO`)

## Running on CCC

Jobs are submitted to the `normal` queue on IBM's CCC cluster. Use `-gpu "num=1:gmem=79G:j_exclusive=yes"` to ensure exclusive GPU access and avoid OOM from sharing with another process.

```bash
bsub -q normal \
  -J kvcache_mass_bench \
  -gpu "num=1:gmem=79G:j_exclusive=yes" \
  -W 48:00 \
  -cwd /dccstor/nathan-ckpts/anooshka \
  -o /dccstor/nathan-ckpts/anooshka/logs/out.%J.txt \
  -e /dccstor/nathan-ckpts/anooshka/logs/err.%J.txt \
  bash -lc '
    export PATH="$HOME/.local/bin:$PATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p /dccstor/nathan-ckpts/anooshka/logs
    micromamba run -p /u/apendyal/mamba_envs/mellea_py310 \
      python -u /dccstor/nathan-ckpts/anooshka/kv_cache.py \
        --jurisdiction "Mass." \
        --reporters_metadata /dccstor/nathan-ckpts/anooshka/ReportersMetadata.json \
        --out_root /dccstor/nathan-ckpts/anooshka/caselaw_kv/massachusetts \
        --logs_dir /dccstor/nathan-ckpts/anooshka/logs \
        --max_tokens 17000 \
        --benchmark
  '
```

See the `logs` folder for per-job run records.
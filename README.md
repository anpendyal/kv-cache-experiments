# KV Cache Generation for Caselaw Access Project

Benchmarks KV cache generation and storage for Massachusetts legal cases using IBM Granite 3.3 8B. Cases are sourced from the [Caselaw Access Project](https://case.law/) via the `common-pile/caselaw_access_project` HuggingFace dataset, filtered to a set of Massachusetts reporters.

## Scripts

### `kv_cache_dataset.py`
Baseline script. Generates KV caches using IBM Granite 3.3 8B and writes them to disk as `.pt` files. Tokenization is unbounded (no truncation).

### `kv_cache_dataset_oom_fix.py`
OOM-fixed version. Key changes from baseline:
- Truncates input to 17,000 tokens to avoid OOM on long documents
- Moves KV tensors to CPU before serialization to free GPU staging buffers
- Runs in **benchmarking mode**: serializes to an in-memory buffer (`BytesIO`) to measure `.pt` file size without writing to disk

### `get_num_cases.py`
Reads `JurisdictionsMetadata.json` and prints the case count per jurisdiction and the total.

### `sort_reporters_to_jurisdiction.py`
Groups reporter slugs by jurisdiction given a `ReportersMetadata.json`-style input. Reporters belonging to exactly one jurisdiction are filed under that jurisdiction's name; reporters spanning multiple jurisdictions go under `"Other"`. Includes unit tests — run with:

```bash
python sort_reporters_to_jurisdiction.py
```

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
- `mellea` (internal — provides `LocalHFBackend` and `IBM_GRANITE_3_3_8B`)

## Running on CCC

Jobs are submitted to the `normal` queue on IBM's CCC cluster:

```bash
bsub -q normal \
  -J kvcache_mass_oom_fix \
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
      python -u /dccstor/nathan-ckpts/anooshka/kv_cache_dataset_oom_fix.py \
        --out_root /dccstor/nathan-ckpts/anooshka/caselaw_kv/massachusetts \
        --log /dccstor/nathan-ckpts/anooshka/logs/caselaw_kv.${LSB_JOBID}.log
  '
```

See the `logs` folder for per-job run records.
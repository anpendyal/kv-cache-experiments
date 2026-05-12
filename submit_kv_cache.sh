#!/bin/bash
#BSUB -J kvcache_mass_bench
#BSUB -gpu "num=1:gmem=40G"
#BSUB -W 48:00
#BSUB -cwd /dccstor/nathan-ckpts/anooshka
#BSUB -o /dccstor/nathan-ckpts/anooshka/logs/out.%J.txt
#BSUB -e /dccstor/nathan-ckpts/anooshka/logs/err.%J.txt

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
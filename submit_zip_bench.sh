#!/bin/bash
#BSUB -J zip_bench
#BSUB -gpu "num=1:gmem=40G"
#BSUB -M 65536
#BSUB -hl
#BSUB -W 48:00
#BSUB -cwd /dccstor/nathan-ckpts/anooshka
#BSUB -o /dccstor/nathan-ckpts/anooshka/logs/out.%J.txt
#BSUB -e /dccstor/nathan-ckpts/anooshka/logs/err.%J.txt

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
micromamba run -p /u/apendyal/mamba_envs/mellea_py310 \
    python -u /dccstor/nathan-ckpts/anooshka/kv_press_bench_zip.py \
        --jurisdiction "Mass." \
        --reporters_metadata /dccstor/nathan-ckpts/anooshka/ReportersMetadata.json \
        --logs_dir /dccstor/nathan-ckpts/anooshka/logs \
        --compression_ratio 0.5 \
        --max_tokens 17000

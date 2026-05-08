#!/bin/bash     
#BSUB -J kvpress_eval
#BSUB -gpu "num=1:gmem=40G"                                                                                                                                   
#BSUB -M 65536
#BSUB -hl                                                                                                                                                     
#BSUB -W 4:00   
#BSUB -cwd /dccstor/nathan-ckpts/anooshka
#BSUB -o /dccstor/nathan-ckpts/anooshka/logs/kv_press_eval/out.%J.txt                                                                                                       
#BSUB -e /dccstor/nathan-ckpts/anooshka/logs/kv_press_eval/err.%J.txt
                                                                                                                                                              
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
micromamba run -p /u/apendyal/mamba_envs/mellea_py310 python -u /dccstor/nathan-ckpts/anooshka/kv_press_eval.py 
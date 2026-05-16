#!/bin/bash

# ── CUDA device ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ── Run pipeline ──────────────────────────────────────────────
python main.py \
    --save_path         /kaggle/temp/maskgct_openslr \
    --whisper_arch      large-v3 \
    --compute_type      float16 \
    --batch_size        8 \
    --threads           4

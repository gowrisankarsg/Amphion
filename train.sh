#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0
export T2S_RUN_NAME="maskgct_t2s_v14_optimized"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# FIX: Prevents tokenizer deadlocks and multiprocessing warnings
export TOKENIZERS_PARALLELISM=false


python train_maskgct_t2s.py \
  --root-dir /teamspace/studios/this_studio/dataset \
  --output-dir /teamspace/studios/this_studio/output_maskgct_t2s \
  --config models/tts/maskgct/config/maskgct.json \
  --train-manifest /teamspace/studios/this_studio/dataset/ta_train.jsonl::ta \
  --train-manifest /teamspace/studios/this_studio/dataset/en/en/en_train.jsonl::en \
  --train-manifest /teamspace/studios/this_studio/dataset/zh/zh/zh_train.jsonl::zh \
  --val-manifest /teamspace/studios/this_studio/dataset/ta/ta/ta_val.jsonl::ta \
  --val-manifest /teamspace/studios/this_studio/dataset/en/en/en_val.jsonl::en \
  --val-manifest /teamspace/studios/this_studio/dataset/zh/zh/zh_val.jsonl::zh \
  --batch-size 40 \
  --grad-accumulation 10 \
  --epochs 2 \
  --learning-rate 2e-5 \
  --backbone-lr-scale 0.1 \
  --weight-decay 0.01 \
  --warmup-steps 200 \
  --grad-clip 1.0 \
  --num-workers 8 \
  --seed 42 \
  --lang-balance "ta:6,en:2,zh:2" \
  --log-interval 50 \
  --val-interval 500 \
  --save-every 1000 \
  --keep-last 3 \
  --ewc-lambda 5000 \
  --amp \
  --ewc-fisher-batches 10 \
  --init-ckpt /teamspace/studios/this_studio/output_maskgct_t2s/step_0002000.pth

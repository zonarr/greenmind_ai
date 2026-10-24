#!/usr/bin/env bash
# Model ID and full launch command
MODEL=CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp
PORT=8000

export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve $MODEL \
  --host 0.0.0.0 --port $PORT \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 40960 --gpu-memory-utilization 0.90 \
  2>&1 | tee serve.log

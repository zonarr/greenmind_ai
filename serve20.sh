#!/usr/bin/env bash
set -euo pipefail

MODEL="CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp"
REVISION="c44c8a80f9de5c6b252ce39a419dcf6015512afb"
PORT=8000

export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p logs
RUN_ID=$(date +%Y%m%d-%H%M%S)

vllm serve "$MODEL" \
  --revision "$REVISION" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  2>&1 | tee "logs/awq-20-${RUN_ID}.log"
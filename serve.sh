#!/usr/bin/env bash
# GreenMind: confirmed 99.2 Final configuration.
# Usage: bash serve-final.sh [run-label]
# Requires the local W8A8 model, patched vLLM environment, and record.sh.
set -eo pipefail

source "$HOME/venv/bin/activate"

MODEL="/home/researcher/models/MANGO1.5-Qwen3.5-9B-W8A8"
SERVED_MODEL="CMKL/MANGO1.5-Qwen3.5-9B"
PORT=8000
POWER_LIMIT=${POWER_LIMIT:-400}  # The 99.2 Final used 400 W.
RUN="$(date +%Y%m%d-%H%M%S)-${1:-final-w8a8}"
RUN_DIR="$HOME/greenmind/runs/$RUN"

# Stop an existing server separately before launching this one.
if [[ -n "$(ss -H -ltn "sport = :$PORT")" ]]; then
  echo "Port $PORT is busy. Stop the existing server before restarting." >&2
  exit 1
fi

# Keep the power cap constant throughout measurement, including idle and drain.
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl "$POWER_LIMIT"

export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# These controls require the custom vLLM scheduler patches.
export GREENMIND_SLO_QUEUE=1
export GREENMIND_CHUNK=1024

# Record the environment and this exact launcher before serving.
mkdir -p "$RUN_DIR"
bash "$HOME/greenmind/record.sh" "$RUN" "$(realpath "$0")"

# Preserve the winning settings; KV-cache dtype remains the vLLM default (auto).
vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 1024 \
  --max-num-seqs 64 \
  --prefix-match-unit 32 \
  --async-scheduling \
  2>&1 | tee "$RUN_DIR/serve.log"

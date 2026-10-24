#!/usr/bin/env bash
# Model ID and full launch command

# Activate the Python environment where vLLM is installed
source ~/venv/bin/activate

# Which model to serve: the 4-bit (AWQ) version that includes the MTP draft head.
MODEL=CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp
# Port the server listens on. Grader endpoint: http://<vm-ip>:8000/v1
PORT=8000
# GPU power cap in watts, held for the whole run (A100 default 400, min 100).
# Override per run: POWER_LIMIT=200 bash serve.sh mtp2-async-pl200
POWER_LIMIT=${POWER_LIMIT:-400}

# Refuse to start if an old server still holds the port (avoids "Address already in use" runs)
if ss -ltn | grep -q ":$PORT "; then
  echo "Port $PORT is busy - stop the old server first (pkill -f 'vllm serve')"; exit 1
fi

# Set the power cap BEFORE recording, so env.txt / gpu-settings.txt capture it.
# Persistence mode keeps the driver loaded so the cap is not reset while idle.
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl "$POWER_LIMIT"

# Name this run: bash serve.sh mtp2-async-pl250  ->  runs/20261024-2130-mtp2-async-pl250
RUN="$(date +%Y%m%d-%H%M)-${1:-unnamed}"
# Snapshot THIS script (not ~/greenmind/serve.sh), so the record matches what launched
bash ~/greenmind/record.sh "$RUN" "$(realpath "$0")"

# HF_HUB_OFFLINE=1      use only the already-downloaded model, never contact Hugging Face
# VLLM_NO_USAGE_STATS=1 / DO_NOT_TRACK=1   turn off vLLM's anonymous telemetry
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Start the server. Each line ending in \ continues the same command.
#   --host 0.0.0.0                 accept connections from outside the VM (the grader)
#   --port $PORT                   listen on port 8000
#   --enable-auto-tool-choice      let the model decide when to call tools (required)
#   --tool-call-parser qwen3_coder turn the model's tool-call text into OpenAI tool_calls
#   --reasoning-parser qwen3       REQUIRED: without it the grader's thinking_token_budget requests fail (400)
#   --max-model-len 40960          longest conversation: 32K prompt + 8K answer
#   --gpu-memory-utilization 0.95  Final-track peak (~110 running) needs the cache room
#   MTP 2 tokens                   MTP3 fills the cache at the Final peak; MTP2 was the best submission
#   --max-num-batched-tokens 8192  no queueing in the burst (4096 left 4-12 waiting)
#   --kv-cache-dtype fp8           as in the best submission; A/B test removing it later
#   --async-scheduling             as in the best submission
#   2>&1 | tee serve.log           show output on screen AND save it to runs/$RUN/serve.log

BT=${BT:-2048}        # max batched tokens per step; override with: BT=1024 bash serve.sh name
MTP=${MTP:-2}         # speculative tokens

vllm serve $MODEL \
  --revision c44c8a80f9de5c6b252ce39a419dcf6015512afb \
  --host 0.0.0.0 --port $PORT \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 40960 --gpu-memory-utilization 0.95 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}" \
  --max-num-batched-tokens $BT \
  --long-prefill-token-threshold $((BT/2)) \
  --max-num-seqs 160 \
  --kv-cache-dtype fp8 \
  --async-scheduling \
  2>&1 | tee ~/greenmind/runs/$RUN/serve.log

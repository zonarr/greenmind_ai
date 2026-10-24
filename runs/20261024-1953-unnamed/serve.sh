#!/usr/bin/env bash
# Model ID and full launch command

# Activate the Python environment where vLLM is installed
source ~/venv/bin/activate

# Name this run: bash serve.sh mtp1  ->  runs/20261024-1405-mtp1
  RUN="$(date +%Y%m%d-%H%M)-${1:-unnamed}"
  bash ~/greenmind/record.sh "$RUN"



# --- Auto-save everything in ~/greenmind to GitHub ---
# Runs before the server starts and after it stops, never during a graded run.
push_to_github() {
  ( cd ~/greenmind || exit
    git add -A                                   # stage every new/changed file
    git commit -q -m "auto: $RUN ($1)" || true   # save a snapshot ("|| true": fine if nothing changed)
    timeout 60 git push -q \
      || echo "WARNING: git push failed (server not affected)" )
}
push_to_github "before start"                    # saves the previous run's log
trap 'exit 130' INT                              # Ctrl+C -> exit normally, so the next line runs
trap 'push_to_github "after stop"' EXIT          # saves this run's log when the server stops


# Which model to serve: the 4-bit (AWQ) version that includes the MTP draft head.
MODEL=CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp
# Port the server listens on. Grader endpoint: http://<vm-ip>:8000/v1
PORT=8000

# HF_HUB_OFFLINE=1      use only the already-downloaded model, never contact Hugging Face
# VLLM_NO_USAGE_STATS=1 / DO_NOT_TRACK=1   turn off vLLM's anonymous telemetry
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Start the server. Each line ending in \ continues the same command.
#   --host 0.0.0.0                 accept connections from outside the VM (the grader)
#   --port $PORT                   listen on port 8000
#   --enable-auto-tool-choice      let the model decide when to call tools (required)
#   --tool-call-parser qwen3_coder turn the model's tool-call text into OpenAI tool_calls
#   --max-model-len 40960          longest conversation: 32K prompt + 8K answer
#   --gpu-memory-utilization 0.90  use 90% of GPU memory for model + request memory
#   2>&1 | tee serve.log           show output on screen AND save it to serve.log

vllm serve $MODEL \
  --revision c44c8a80f9de5c6b252ce39a419dcf6015512afb \
  --host 0.0.0.0 --port $PORT \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 40960 --gpu-memory-utilization 0.95 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --max-num-seqs 160 \
  --kv-cache-dtype fp8 \
  --async-scheduling \
  2>&1 | tee ~/greenmind/runs/$RUN/serve.log

# vllm serve $MODEL \
#   --revision c44c8a80f9de5c6b252ce39a419dcf6015512afb \
#   --host 0.0.0.0 --port $PORT \
#   --language-model-only \
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder \
#   --max-model-len 40960 --gpu-memory-utilization 0.95 \
#   --max-num-batched-tokens 8192 \
#   --long-prefill-token-threshold 2048 \
#   --kv-cache-dtype fp8 \
#   2>&1 | tee ~/greenmind/runs/$RUN/serve.log

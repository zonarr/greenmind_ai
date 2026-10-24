#!/usr/bin/env bash
# Usage: bash record.sh <run-name>   e.g. bash record.sh 01-awq-baseline
RUN=${1:-$(date +%Y%m%d-%H%M)}
DIR=~/greenmind/runs/$RUN
mkdir -p $DIR

cp "${2:-$HOME/greenmind/serve.sh}" "$DIR/serve.sh"

{
  echo "date:   $(date -Is)"
  echo "vllm:   $(python -c 'import vllm; print(vllm.__version__)')"
  echo "torch:  $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
  echo "model snapshot: $(ls ~/.cache/huggingface/hub/models--CMKL--MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp/snapshots/)"
  echo
  echo "## GPU power / clocks"
  nvidia-smi --query-gpu=name,driver_version,persistence_mode,power.limit,power.default_limit,clocks.max.sm --format=csv
} > $DIR/env.txt

nvidia-smi -q -d POWER,CLOCK > $DIR/gpu-settings.txt
pip freeze > $DIR/pip-freeze.txt

echo "Saved to $DIR"

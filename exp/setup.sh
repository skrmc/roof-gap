#!/bin/bash
# Installs the GPU-side dependencies and downloads the model (ungated mirror with the same weights as
# meta-llama/Llama-3.1-8B-Instruct; the Llama 3.1 Community License applies).
set -eu
cd "$(dirname "$0")"
pip install vllm==0.32.0
hf download unsloth/Meta-Llama-3.1-8B-Instruct --include "*.safetensors" --include "*.json" \
  --local-dir "${MODEL_DIR:-$PWD/../models/llama31-8b}"

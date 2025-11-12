#!/bin/bash
# Run TinyLlama CodeGen in an isolated environment

# Unset problematic environment variables
unset TF_CPP_MIN_LOG_LEVEL
unset TRANSFORMERS_VERBOSITY

# Set environment to use PyTorch only
export DISABLE_TENSORFLOW=1
export USE_TORCH=1
export USE_TF=0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Disable any GPU usage to ensure CPU-only
export CUDA_VISIBLE_DEVICES=""
export ROCM_VISIBLE_DEVICES=""

# Run the codegen version
echo "🚀 Starting TinyLlama CodeGen..."
python3 /home/jason/tinyllama/tinyllama_codegen.py "$@"

#!/bin/bash
# Run TinyLlama in an isolated environment to avoid TensorFlow conflicts

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

# Run the fixed version
echo "🚀 Starting TinyLlama Chat (Fixed Version)..."
python3 /home/jason/tinyllama/tinyllama_chat_fixed.py "$@"

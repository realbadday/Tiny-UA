#!/bin/bash
# Run TinyLlama in chat mode with TTS enabled

# Unset problematic environment variables
unset TF_CPP_MIN_LOG_LEVEL
unset TRANSFORMERS_VERBOSITY

# Set environment for PyTorch only
export DISABLE_TENSORFLOW=1
export USE_TORCH=1
export USE_TF=0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""
export ROCM_VISIBLE_DEVICES=""

echo "🚀 Starting TinyLlama Chat with TTS..."
echo "💬 Voice-enabled chat assistant"
echo ""

# Run unified assistant in chat mode with TTS
python3 /home/jason/tinyllama/tinyllama_unified_tts.py --mode chat "$@"

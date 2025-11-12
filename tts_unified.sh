#!/bin/bash
# Run TinyLlama Unified Assistant with TTS enabled

# Check for required dependencies
if ! command -v espeak &> /dev/null; then
    echo "⚠️  Warning: espeak not found. TTS may not work properly."
    echo "Install with: sudo apt-get install espeak libespeak-dev"
fi

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

# Parse arguments
SHOW_HELP=false
for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        SHOW_HELP=true
        break
    fi
done

if $SHOW_HELP; then
    echo "🚀 TinyLlama Unified Assistant with TTS"
    echo ""
    echo "Usage: $0 [options] [query]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE      Initial mode (chat, code, function, etc.)"
    echo "  --no-tts             Disable text-to-speech"
    echo "  --tts-rate RATE      Speech rate in WPM (50-300, default: 175)"
    echo "  --tts-volume VOL     Volume (0.0-1.0, default: 0.9)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                          # Interactive mode with TTS"
    echo "  $0 --mode function          # Start in function generation mode"
    echo "  $0 'explain decorators'     # Single query with TTS response"
    echo "  $0 --no-tts                 # Interactive mode without TTS"
    echo ""
    echo "Voice Commands (in interactive mode):"
    echo "  /tts         Toggle voice output"
    echo "  /voices      List available voices"
    echo "  /rate 150    Set speech rate"
    echo "  /volume 0.8  Set volume"
    echo ""
    exit 0
fi

echo "🚀 Starting TinyLlama Unified Assistant with TTS..."
echo "💡 Tip: Use --help for usage information"
echo ""

# Run the unified assistant
python3 /home/jason/tinyllama/tinyllama_unified_tts.py "$@"

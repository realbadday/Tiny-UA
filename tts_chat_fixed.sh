#!/bin/bash
# TinyLlama Chat with TTS - Fixed Version
# Wrapper script for easy access to TTS features

# Check if --help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "🎤 TinyLlama Chat with TTS (Fixed Version)"
    echo ""
    echo "Usage: $0 [options] [query]"
    echo ""
    echo "Options:"
    echo "  --no-tts         Run without text-to-speech"
    echo "  --rate RATE      Speech rate in WPM (50-300, default: 175)"
    echo "  --volume VOL     Volume (0.0-1.0, default: 0.9)"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Interactive mode with TTS"
    echo "  $0 'explain decorators'      # Single query with TTS"
    echo "  $0 --no-tts                  # Interactive without TTS"
    echo "  $0 --rate 150                # Slower speech"
    echo ""
    echo "Voice Commands (in interactive mode):"
    echo "  /tts         Toggle voice output"
    echo "  /voices      List available voices"
    echo "  /rate 150    Set speech rate"
    echo "  /volume 0.8  Set volume"
    echo ""
    exit 0
fi

# Default to TTS enabled
TTS_FLAG="--tts"

# Parse arguments
ARGS=""
for arg in "$@"; do
    if [[ "$arg" == "--no-tts" ]]; then
        TTS_FLAG=""
    else
        ARGS="$ARGS $arg"
    fi
done

# Run the fixed chat with TTS enabled by default
exec python3 "$(dirname "$0")/tinyllama_chat_fixed.py" $TTS_FLAG $ARGS

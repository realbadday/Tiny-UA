#!/bin/bash
# TinyLlama Unified Assistant - Simple Start Script

# Activate TinyLlama environment if it exists
if [ -f "$HOME/.tinyllama/bin/activate" ]; then
    source "$HOME/.tinyllama/bin/activate"
fi

# Run the unified assistant
exec python3 "$(dirname "$0")/tinyllama_unified_tts.py" "$@"

#!/bin/bash
# TinyLlama Verbose Launch Script (shows all output including ALSA messages)
# Usage: ./launch-verbose.sh [options]

cd /home/jason/projects/tinyllama

# Check if virtual environment exists
if [ ! -d "tinyllama_env" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    echo "Run: ./activate_training.sh"
    exit 1
fi

# Activate environment and launch with full output
echo "🚀 Starting TinyLlama (verbose mode)..."
source tinyllama_env/bin/activate

# Set up audio environment
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0

# If no arguments, start interactive mode
if [ $# -eq 0 ]; then
    python3 main.py
else
    # Pass all arguments to main.py
    python3 main.py "$@"
fi

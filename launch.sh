#!/bin/bash
# TinyLlama Quick Launch Script
# Usage: ./launch.sh [options]

cd /home/jason/projects/tinyllama

# Check if virtual environment exists
if [ ! -d "tinyllama_env" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    echo "Run: ./activate_training.sh"
    exit 1
fi

# Activate environment and launch
echo "🚀 Starting TinyLlama..."
source tinyllama_env/bin/activate

# Ensure correct TTS voice configuration
if [ ! -f ~/.tinyllama/tts_config.json ] || ! grep -q '"english+f2"' ~/.tinyllama/tts_config.json 2>/dev/null; then
    echo "🔧 Setting up TTS voice..."
    mkdir -p ~/.tinyllama
    cat > ~/.tinyllama/tts_config.json << 'EOF'
{
  "rate": 160,
  "volume": 0.9,
  "voice": "english+f2",
  "silent_status": false,
  "speak_summaries": true
}
EOF
fi

# Set up audio environment to minimize ALSA warnings
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0

# If no arguments, start interactive mode
if [ $# -eq 0 ]; then
    python3 main.py 2>/dev/null
else
    # Pass all arguments to main.py
    python3 main.py "$@" 2>/dev/null
fi

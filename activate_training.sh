#!/bin/bash
# TinyLlama Training Environment Activation Script
# This script activates the virtual environment with all training dependencies

echo "🚀 Activating TinyLlama Training Environment..."

# Change to the project directory
cd /home/jason/projects/tinyllama

# Check if virtual environment exists
if [ ! -d "tinyllama_env" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    exit 1
fi

# Activate the virtual environment
source tinyllama_env/bin/activate

echo "✅ Environment activated!"
echo "📚 Available commands:"
echo "   # Training"
echo "   python train_tinyllama.py --help    # Show training options"
echo "   python train_tinyllama.py --dataset data/sample_dataset.csv --epochs 3 --test-prompts"
echo ""
echo "   # Main Interface"
echo "   python main.py                      # Start unified assistant with TTS"
echo "   python main.py --no-tts            # Start without TTS"
echo "   python main.py 'your question'     # Direct query"
echo ""
echo "   # Alternative Interfaces"
echo "   python tinyllama_chat.py            # Basic chat interface"
echo "   python programming_assistant.py     # Enhanced programming assistant"
echo ""
echo "💡 To exit the environment, type: deactivate"

# Start a new bash session with the environment activated
exec bash

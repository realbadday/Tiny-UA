#!/bin/bash
# TinyLlama Programming Assistant Setup Script

echo "🚀 TinyLlama Programming Assistant Setup"
echo "========================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv tinyllama_env
    source tinyllama_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
echo "This may take a few minutes..."

# Check if peft is needed
read -p "Include fine-tuning support? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    pip install -r requirements.txt
else
    # Install without peft
    pip install torch transformers accelerate tqdm
fi

# Install TTS support
echo ""
read -p "Install text-to-speech support? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    pip install pyttsx4
    # Install espeak on Debian/Ubuntu
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y espeak
    fi
    echo "✅ TTS support installed"
fi

# Create convenient launcher
echo ""
echo "🔧 Creating launcher script..."
cat > tinyllama << 'EOF'
#!/bin/bash
# TinyLlama launcher

# Activate virtual environment if it exists
if [ -d "tinyllama_env" ]; then
    source tinyllama_env/bin/activate
fi

# Default to programming assistant
if [ $# -eq 0 ]; then
    python3 ~/tinyllama/programming_assistant.py
else
    # Pass all arguments
    python3 ~/tinyllama/programming_assistant.py "$@"
fi
EOF

chmod +x tinyllama

# Test the installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import torch, transformers, accelerate; print('✅ Core packages installed successfully')"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🎯 Quick Start:"
    echo "   ./tinyllama              # Start interactive assistant"
    echo "   ./tinyllama --help       # Show all options"
    echo "   ./tinyllama --tts        # Enable voice output"
    echo ""
    echo "📚 Or use directly:"
    echo "   python3 tinyllama_chat.py         # Basic chat"
    echo "   python3 programming_assistant.py  # Full features"
    echo "   python3 train_tinyllama.py        # Fine-tune on your data"
    echo ""
    echo "💡 The first run will download the model (~1.1GB)"
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi

# Offer to download model now
echo ""
read -p "Download TinyLlama model now? (~1.1GB) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading TinyLlama model..."
    python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('Downloading...'); AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); print('✅ Model downloaded successfully!')"
fi

echo ""
echo "🎉 Setup complete! Enjoy your propaganda-free programming assistant!"

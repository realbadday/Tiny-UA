# 🤖 TinyLlama Programming Assistant

An efficient, CPU-optimized AI assistant for programming tasks, based on TinyLlama-1.1B with enhanced features and no propaganda.

## 🌟 Features

- **Efficient CPU Inference**: Optimized for 4-core CPUs with INT8 quantization
- **Programming-Focused**: Specialized modes for code generation, debugging, and planning
- **Enhanced Communication**: Better conversational abilities than CodeGen
- **No Propaganda**: Clean, practical responses focused on programming
- **Response Caching**: Fast repeated queries with intelligent caching
- **TTS Support**: Optional text-to-speech for responses
- **Fine-tuning Ready**: Easy training on your own datasets

## 📋 Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0  # For LoRA fine-tuning

# Optional
pyttsx4  # For text-to-speech
```

## 🚀 Quick Start

### Installation (First Time Only)

```bash
# Extract the tiny 87KB archive
tar -xzf tinyllama-v1.0-source.tar.gz
cd tinyllama

# Run setup (downloads ~6GB of dependencies)
./setup.sh
```

### 1. Basic Usage

```bash
# Interactive chat
./launch.sh

# Direct query
./launch.sh "how to read a CSV file in Python"

# Different modes
./launch.sh -m code "merge two sorted lists"
./launch.sh -m explain "decorators"
```

### 2. Enhanced Features

```bash
# With text-to-speech
./launch.sh --tts

# Planning mode
./launch.sh -m plan "build a REST API"

# Debug mode
./launch.sh -m debug "paste your code here"
```

### 3. Access from Chat Menu

```bash
cd ~/chat
python chat.py
# Select option 5: TinyLlama Programming Assistant
```

## 🎯 Usage Modes

### Standard Modes
- **chat**: General programming discussion
- **code**: Generate Python code
- **explain**: Explain concepts
- **debug**: Debug code issues

### Enhanced Modes
- **plan**: Break down complex tasks
- **review**: Code quality feedback
- **test**: Generate unit tests
- **optimize**: Performance improvements
- **doc**: Add documentation

## 🔧 Fine-tuning

### Train on Your Data

1. Prepare your dataset (CSV with question/answer columns)
2. Run training:

```bash
# Activate the environment first
source tinyllama_env/bin/activate

# Run training
python train_tinyllama.py \
  --dataset data/dataset_clean.csv \
  --epochs 3 \
  --merge-weights \
  --test-prompts
```

### Training Tips
- Keep batch size at 1 for CPU training
- Use LoRA for efficient fine-tuning (only trains ~3% of parameters)
- Enable `--merge-weights` for easier deployment
- Test with `--test-prompts` to verify quality

## 💡 Examples

### Code Generation
```
[CODE] Query: binary search function
→ Generates optimized binary search implementation
```

### Task Planning
```
[PLAN] Query: create a web scraper
→ Step-by-step implementation plan with best practices
```

### Code Review
```
[REVIEW] Query: [paste your function]
→ Identifies issues, suggests improvements
```

### Multiline Input
```
/multiline
def factorial(n):
    if n <= 1:
        return n
    return factorial(n-1) * n
EOF
→ Reviews and fixes the code
```

## ⚡ Performance Optimization

### CPU Settings
- Uses 4 threads by default (modify in `tinyllama_chat.py`)
- INT8 quantization reduces memory by ~75%
- Response caching for instant repeated queries

### Memory Usage
- Base model: ~1.1GB
- Quantized: ~300-400MB
- Fine-tuned LoRA: +16MB

## 🔍 Troubleshooting

### Model Download Issues
- First run downloads ~1.1GB from HuggingFace
- Ensure stable internet connection
- Models cached in `~/.cache/huggingface`

### Slow Generation
- Normal: 2-5 seconds per response on CPU
- Enable caching for repeated queries
- Reduce `max_new_tokens` for faster responses

### Import Errors
```bash
pip install torch transformers peft
```

### TTS Not Working
```bash
pip install pyttsx4
sudo apt-get install espeak  # On Debian/Ubuntu
```

### Improve Mode Bug (IMPORTANT)
The `/mode improve` command has a bug where it gets stuck in an input loop. **Use this workaround instead:**
```bash
# DON'T use: /mode improve
# USE: chat mode
/mode chat
improve this code: def hello(): print("hello")
```
This works perfectly and gives the same results. See KNOWN_ISSUES.md for details.

## 🛠️ Advanced Configuration

### Custom Model Path
```bash
python tinyllama_chat.py --model-path /path/to/model
```

### Disable Quantization
```bash
python tinyllama_chat.py --no-quantization
```

### Adjust Generation
Edit generation parameters in `tinyllama_chat.py`:
- `temperature`: 0.7 (creativity)
- `top_p`: 0.9 (nucleus sampling)
- `max_new_tokens`: 256 (response length)

## 📂 Project Structure

```
~/tinyllama/
├── tinyllama_chat.py          # Basic chat interface
├── programming_assistant.py    # Enhanced assistant
├── train_tinyllama.py         # Fine-tuning script
├── models/                    # Saved models
│   └── tinyllama-python-qa/   # Fine-tuned model
├── data/                      # Datasets
│   └── dataset_clean.csv      # Python Q&A data
└── response_cache.json        # Cached responses
```

## 🤝 Comparison with Alternatives

| Feature | TinyLlama | GPT-2 | CodeGen |
|---------|-----------|--------|----------|
| Size | 1.1B | 124M-1.5B | 350M-16B |
| Programming Knowledge | Good | Limited | Excellent |
| Conversation | Excellent | Poor | Limited |
| CPU Speed | Fast | Very Fast | Slow |
| Memory | ~400MB | ~500MB | ~700MB+ |

## 🚧 Future Enhancements

- [ ] Support for more programming languages
- [ ] Integration with code execution
- [ ] RAG for documentation lookup
- [ ] Quantization to 4-bit for even smaller size
- [ ] Web UI interface

## 📄 License

This project uses:
- TinyLlama: Apache 2.0 License
- Your code: MIT License

## 🙏 Acknowledgments

- TinyLlama team for the efficient base model
- Hugging Face for transformers library
- Your existing chat infrastructure for inspiration

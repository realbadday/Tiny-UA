# TinyLlama Scripts Guide

## Overview
This directory contains various TinyLlama scripts optimized for CPU-only inference with different capabilities.

## Available Scripts

### 1. **tinyllama_chat_fixed.py** - General Chat (TensorFlow Conflict Fixed)
The fixed version of the basic chat interface that avoids TensorFlow/PyTorch conflicts.

```bash
# Interactive mode
./run_tinyllama.sh

# Single query
./run_tinyllama.sh "What is a Python decorator?"
```

**Features:**
- General programming Q&A
- Multiple modes: chat, code, explain, debug
- Response caching for speed
- CPU-optimized with INT8 quantization

### 2. **tinyllama_codegen.py** - Code Generation Specialist
Enhanced version specifically for generating Python code with multiple generation modes.

```bash
# Interactive mode
./run_codegen.sh

# Generate a function
./run_codegen.sh function "merge two sorted lists"

# Generate a class
./run_codegen.sh class "binary search tree with insert and search"

# Generate a complete script
./run_codegen.sh script "web scraper for news articles"
```

**Features:**
- **Code Generation Modes:**
  - `/function` - Generate Python functions
  - `/class` - Generate Python classes  
  - `/script` - Generate complete scripts
  - `/algorithm` - Implement specific algorithms
  
- **Code Improvement Modes:**
  - `/fix` - Fix buggy code
  - `/improve` - Optimize code for efficiency
  - `/explain` - Get detailed explanations
  - `/test` - Generate unit tests
  - `/convert` - Convert from other languages

- **Advanced Features:**
  - Custom stopping criteria for complete code
  - Code block extraction
  - Multiline input support
  - Pattern templates (API, CLI, async, etc.)

### 3. **programming_assistant.py** - Full-Featured Assistant
The most comprehensive version with TTS support and all features.

```bash
python3 programming_assistant.py
```

**Features:**
- All chat and code generation features
- Text-to-speech support
- Planning and review modes
- Code documentation generation

## Quick Start Examples

### Generate a Function
```bash
./run_codegen.sh
>>> /function calculate fibonacci numbers recursively
```

### Fix Buggy Code
```bash
./run_codegen.sh
>>> /fix
📝 Enter code to fix (type 'EOF' or '```' when done):
... def factorial(n):
...     return factorial(n-1) * n
... EOF
```

### Generate Unit Tests
```bash
./run_codegen.sh
>>> /test
📝 Enter code to test (type 'EOF' or '```' when done):
... def is_prime(n):
...     if n < 2:
...         return False
...     for i in range(2, int(n**0.5) + 1):
...         if n % i == 0:
...             return False
...     return True
... EOF
```

### Convert JavaScript to Python
```bash
./run_codegen.sh
>>> /convert JavaScript
📝 Enter code to convert (type 'EOF' or '```' when done):
... const sum = (a, b) => a + b;
... const numbers = [1, 2, 3, 4, 5];
... const total = numbers.reduce(sum, 0);
... EOF
```

## Troubleshooting

### TensorFlow Conflict Error
If you see "computation placer already registered" error:
- Use the fixed scripts: `tinyllama_chat_fixed.py` or `tinyllama_codegen.py`
- Use the wrapper scripts: `./run_tinyllama.sh` or `./run_codegen.sh`

### Memory Issues
- The scripts use INT8 quantization by default to reduce memory usage
- Disable quantization if you have issues: `--no-quantization`

### Slow Generation
- First run downloads the model (~2GB)
- Subsequent runs use cached responses for common queries
- Clear cache with `/clear` command

## Performance Tips

1. **Use specific modes** for better results:
   - `/function` for function generation
   - `/class` for class generation
   - `/explain` for explanations

2. **Keep queries concise** - shorter prompts generate faster

3. **Use caching** - common queries are cached for instant responses

4. **CPU Optimization** - Scripts are optimized for 4-core CPUs

## Model Information

- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Size**: 1.1 billion parameters
- **Optimized for**: Python programming assistance
- **Inference**: CPU-only with INT8 quantization
- **Memory Usage**: ~2-3GB RAM

## Additional Commands

All scripts support these commands in interactive mode:
- `/help` - Show detailed help
- `/clear` - Clear response cache
- `/nocache` - Toggle caching on/off
- `quit` - Exit the program

## Examples Directory

Check the `scripts/` directory for example use cases and batch processing scripts.

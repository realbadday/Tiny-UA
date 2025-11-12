#!/usr/bin/env python3
"""Diagnose TensorFlow/PyTorch conflicts"""

import sys
import os

print("🔍 Diagnosing Python environment...")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\n📦 Checking installed packages...")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"   PyTorch location: {torch.__file__}")
except ImportError as e:
    print(f"❌ PyTorch not found: {e}")

try:
    import tensorflow as tf
    print(f"⚠️  TensorFlow version: {tf.__version__}")
    print(f"   TensorFlow location: {tf.__file__}")
    print("   ⚠️  TensorFlow is installed and may cause conflicts!")
except ImportError:
    print("✅ TensorFlow not imported (good for avoiding conflicts)")

try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
    print(f"   Transformers location: {transformers.__file__}")
except ImportError as e:
    print(f"❌ Transformers not found: {e}")

print("\n🔧 Environment variables:")
for key in ['TF_CPP_MIN_LOG_LEVEL', 'USE_TORCH', 'USE_TF', 'CUDA_VISIBLE_DEVICES']:
    value = os.environ.get(key, '<not set>')
    print(f"   {key}: {value}")

print("\n💡 Recommendations:")
print("1. Use the fixed script: python3 tinyllama_chat_fixed.py")
print("2. Or use the wrapper: ./run_tinyllama.sh")
print("3. To test: ./run_tinyllama.sh 'What is a Python decorator?'")

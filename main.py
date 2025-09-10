#!/usr/bin/env python3
"""
TinyLlama - Simple Main Entry Point
Just run: python3 main.py
"""

import os
import sys

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DISABLE_TENSORFLOW'] = '1'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

# Import and run the unified assistant
from tinyllama_unified_tts import main

if __name__ == "__main__":
    # Default to TTS enabled for simplicity
    if len(sys.argv) == 1:
        sys.argv.append("--tts")
    main()

#!/usr/bin/env python3
"""
Simple script to improve code without the confusing mode switching
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from tinyllama_unified_tts import TinyLlamaUnifiedTTS

print("TinyLlama Code Improver")
print("=" * 50)
print("Paste your code below, then type END on a new line:")
print()

# Collect code
code_lines = []
while True:
    line = input()
    if line.strip() == "END":
        break
    code_lines.append(line)

code = "\n".join(code_lines)

# Load model and generate improvement
print("\nLoading model...")
assistant = TinyLlamaUnifiedTTS()
assistant.current_mode = "improve"

print("Generating improvements...")
result = assistant.generate_response(code, use_cache=False)

print("\n" + "=" * 50)
print("IMPROVED CODE:")
print("=" * 50)

if isinstance(result, dict):
    print(result.get("response", "No improvement generated"))
else:
    print(result)

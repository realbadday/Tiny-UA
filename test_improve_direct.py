#!/usr/bin/env python3
"""Direct test of improve functionality"""

import os
import sys

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Testing improve mode directly...")

# Import the unified assistant
from tinyllama_unified_tts import TinyLlamaUnifiedTTS

# Create instance
print("Loading model...")
assistant = TinyLlamaUnifiedTTS()

# Test code to improve
test_code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
"""

print("\nOriginal code:")
print(test_code)

print("\n" + "="*60)
print("Generating improvement...")

# Switch to improve mode
assistant.current_mode = "improve"

# Generate improvement (no cache)
result = assistant.generate_response(test_code, use_cache=False)

print("="*60)
print("\nResult:")
if isinstance(result, dict):
    print("Response:", result.get("response", "No response"))
    if result.get("code_blocks"):
        print("\nCode blocks:")
        for block in result["code_blocks"]:
            print("```python")
            print(block)
            print("```")
else:
    print(result)

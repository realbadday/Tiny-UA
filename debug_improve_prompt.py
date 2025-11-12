#!/usr/bin/env python3
"""Debug the improve mode prompt"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from tinyllama_unified_tts import TinyLlamaUnifiedTTS

# Create instance
assistant = TinyLlamaUnifiedTTS()

# Test code
test_code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
"""

# Set to improve mode
assistant.current_mode = "improve"

# Format the prompt
formatted_prompt = assistant.format_prompt(test_code, mode="improve")

print("Mode:", assistant.current_mode)
print("\nFormatted prompt:")
print("-" * 60)
print(formatted_prompt)
print("-" * 60)

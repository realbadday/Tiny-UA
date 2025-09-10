#!/usr/bin/env python3
"""Test the improve mode functionality"""

# First, let's test if we're using the right script
import sys
import os

print("Testing improve mode functionality...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check which scripts have improve functionality
scripts = [
    "tinyllama.py",
    "tinyllama_codegen.py", 
    "tinyllama_unified_tts.py",
    "programming_assistant.py"
]

print("\nChecking scripts with improve mode:")
for script in scripts:
    if os.path.exists(script):
        with open(script, 'r') as f:
            content = f.read()
            if 'improve' in content and ('/mode' in content or '/improve' in content):
                print(f"✓ {script} - has improve mode")
            else:
                print(f"✗ {script} - no improve mode found")
    else:
        print(f"✗ {script} - file not found")

print("\nTo use improve mode correctly:")
print("1. If using tinyllama_codegen.py:")
print("   - Type: /improve")
print("   - Paste your code")
print("   - Type EOF on a new line (just 'EOF', no quotes, no spaces)")
print("\n2. If using mode-based scripts:")
print("   - Type: /mode improve")
print("   - Then paste your code")
print("   - Type EOF on a new line")

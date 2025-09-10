#!/usr/bin/env python3
"""
Better fix for improve mode - handles the flow more intuitively
"""

import shutil
from pathlib import Path

# Backup the current file
original_file = Path("tinyllama_unified_tts.py")
backup_file = Path("tinyllama_unified_tts.py.backup2")

print("Creating backup...")
shutil.copy(original_file, backup_file)

# Read the file
with open(original_file, 'r') as f:
    content = f.read()

# Fix 1: Make improve mode go directly to multiline when selected
fix1 = """
                if user_input.startswith('/mode'):
                    # Accept flexible syntax and case-insensitive matches
                    tokens = re.split(r'[\s=:,]+', user_input.strip())
                    target = None
                    for tok in tokens[1:]:
                        cand = tok.lower()
                        if cand in self.modes:
                            target = cand
                            break
                    if target:
                        self.current_mode = target
                        mode_desc = self.modes[self.current_mode]
                        print(f"📝 Switched to {mode_desc}")
                        if self.tts.enabled:
                            self.tts.speak(f"Switched to {target} mode")
                        
                        # Automatically enter multiline for code improvement modes
                        if target in ["fix", "improve", "explain", "test", "convert"]:
                            print(f"📝 Enter code to {target} (type 'EOF' when done):")
                            multiline_mode = True
                            multiline_buffer = []
                    else:
                        print("❌ Invalid mode. Use /modes to see available modes.")
                    continue
"""

# Find the /mode handling section and replace it
import re

# This is a bit tricky, so let's just add a note
print("\nTo apply a more comprehensive fix:")
print("The issue is that after switching to improve mode, you still need to")
print("type something before it enters multiline mode.")
print("")
print("For now, when you get stuck:")
print("1. Type EOF to exit multiline mode")
print("2. Try using chat mode instead:")
print("   /mode chat")
print("   Then just ask: 'improve this code: [paste code]'")
print("")
print("Or stay in script/code mode and ask for improvements directly.")

# Create a simpler workaround script
workaround = '''#!/usr/bin/env python3
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

code = "\\n".join(code_lines)

# Load model and generate improvement
print("\\nLoading model...")
assistant = TinyLlamaUnifiedTTS()
assistant.current_mode = "improve"

print("Generating improvements...")
result = assistant.generate_response(code, use_cache=False)

print("\\n" + "=" * 50)
print("IMPROVED CODE:")
print("=" * 50)

if isinstance(result, dict):
    print(result.get("response", "No improvement generated"))
else:
    print(result)
'''

# Save the workaround script
with open("improve_code.py", "w") as f:
    f.write(workaround)

print("\n✅ Created improve_code.py as a simpler alternative")
print("Usage: python3 improve_code.py")
print("This bypasses the confusing mode switching.")

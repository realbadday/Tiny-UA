#!/usr/bin/env python3
"""
Patch to fix the improve mode in tinyllama_unified_tts.py
"""

import shutil
from pathlib import Path

# Backup the original file
original_file = Path("tinyllama_unified_tts.py")
backup_file = Path("tinyllama_unified_tts.py.backup")

print("Creating backup...")
shutil.copy(original_file, backup_file)

# Read the file
with open(original_file, 'r') as f:
    lines = f.readlines()

# Find and fix the format_prompt method
fixed = False
for i, line in enumerate(lines):
    # Look for the problematic section around line 195-197
    if "elif mode in [\"fix\", \"improve\", \"explain\", \"test\", \"convert\"]:" in line:
        # We need to fix the next few lines
        # Replace the simple return with proper template formatting
        print(f"Found issue at line {i+1}")
        
        # Replace the next lines with proper formatting
        lines[i+1] = "            # Format with template for code-based modes\n"
        lines[i+2] = "            if mode == \"improve\":\n"
        lines.insert(i+3, "                return self.code_templates[mode].format(code=query, aspect=\"efficiency\")\n")
        lines.insert(i+4, "            elif mode in [\"fix\", \"explain\", \"test\"]:\n")
        lines.insert(i+5, "                return self.code_templates[mode].format(code=query)\n")
        lines.insert(i+6, "            elif mode == \"convert\":\n")
        lines.insert(i+7, "                return self.code_templates[mode].format(code=query, source_lang=\"JavaScript\")\n")
        lines.insert(i+8, "            else:\n")
        lines.insert(i+9, "                return query\n")
        
        # Remove the old "return query" line
        if "return query" in lines[i+10]:
            del lines[i+10]
        
        fixed = True
        break

if fixed:
    print("Writing fixed file...")
    with open(original_file, 'w') as f:
        f.writelines(lines)
    print("✅ Fix applied successfully!")
    print("\nThe improve mode should now work correctly.")
    print("Usage: /mode improve -> 'improve' -> paste code -> EOF")
else:
    print("❌ Could not find the section to fix. The file may have already been patched.")

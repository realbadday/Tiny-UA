#!/usr/bin/env python3
"""Test TTS functionality"""
import pyttsx4

try:
    engine = pyttsx4.init('espeak')
    engine.say("TinyLlama TTS test successful!")
    engine.runAndWait()
    print("✅ TTS is working!")
except Exception as e:
    print(f"❌ TTS Error: {e}")
    print("You may need to install espeak: sudo apt-get install espeak libespeak-dev")

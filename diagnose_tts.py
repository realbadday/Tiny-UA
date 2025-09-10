#!/usr/bin/env python3
"""
TTS Diagnostic Script - Test TTS functionality step by step
"""

import os
import sys
import time

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("🔍 TTS Diagnostic Tool")
print("=" * 40)

# Test 1: Basic pyttsx4 availability
print("\n1. Testing pyttsx4 availability...")
try:
    import pyttsx4
    print("✅ pyttsx4 imported successfully")
except ImportError as e:
    print(f"❌ pyttsx4 import failed: {e}")
    exit(1)

# Test 2: Engine initialization
print("\n2. Testing engine initialization...")
try:
    engine = pyttsx4.init('espeak')
    print("✅ Engine initialized successfully")
except Exception as e:
    print(f"❌ Engine initialization failed: {e}")
    exit(1)

# Test 3: Voice configuration
print("\n3. Testing voice configuration...")
try:
    voices = engine.getProperty('voices')
    print(f"Available voices: {len(voices)} found")
    
    # Look for english+f2 or similar
    english_voices = [v for v in voices if 'english' in v.id.lower()]
    print(f"English voices found: {len(english_voices)}")
    
    if english_voices:
        for voice in english_voices[:3]:
            print(f"  - {voice.id}: {voice.name}")
    
    # Try setting the voice
    engine.setProperty('voice', 'english+f2')
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 0.9)
    print("✅ Voice properties set")
    
except Exception as e:
    print(f"❌ Voice configuration failed: {e}")

# Test 4: Direct TTS test
print("\n4. Testing direct TTS output...")
try:
    print("Speaking test message... (listen for audio)")
    engine.say("TTS diagnostic test. If you hear this, the basic TTS is working.")
    engine.runAndWait()
    print("✅ Direct TTS test completed")
except Exception as e:
    print(f"❌ Direct TTS test failed: {e}")

# Test 5: TTSManager class
print("\n5. Testing TTSManager class...")
try:
    from tts_utils import TTSManager
    
    tts = TTSManager(auto_init=True)
    print(f"TTSManager enabled: {tts.enabled}")
    
    if tts.enabled:
        print("Speaking through TTSManager... (listen for audio)")
        tts.speak("This is a test through the TTS Manager class.", interrupt=True)
        time.sleep(3)  # Give time for threaded speech to complete
        print("✅ TTSManager test completed")
    else:
        print("❌ TTSManager not enabled")
        
except Exception as e:
    print(f"❌ TTSManager test failed: {e}")

# Test 6: Check audio system
print("\n6. Testing audio system...")
os.system("aplay -l 2>/dev/null | head -5")
print("If no audio devices listed above, that might explain TTS issues")

# Test 7: Check if audio is muted
print("\n7. Testing system audio...")
os.system("amixer get Master 2>/dev/null | grep -E '(Playback|%)'")

print("\n" + "=" * 40)
print("🏁 Diagnostic completed!")
print("\nIf you heard audio in tests 4 and 5, TTS is working.")
print("If not, the issue might be:")
print("- Audio system configuration")
print("- Volume settings") 
print("- Audio device conflicts")
print("- PipeWire/ALSA conflicts")

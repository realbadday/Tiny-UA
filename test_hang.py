#!/usr/bin/env python3
"""
Test script to find where the application hangs
"""

import os
import sys
import signal

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def timeout_handler(signum, frame):
    print("\n🚨 TIMEOUT: Application appears to be hanging!")
    print("Last operation was likely the TTS or model generation.")
    sys.exit(1)

# Set up timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    print("🔍 Testing where application hangs...")
    
    print("1. Importing modules...")
    from tinyllama_unified_tts import TinyLlamaUnifiedTTS
    print("✅ Import successful")
    
    print("2. Initializing assistant (without TTS)...")
    assistant = TinyLlamaUnifiedTTS(enable_tts=False, use_quantization=False)
    print("✅ Assistant initialized")
    
    print("3. Testing simple generation...")
    result = assistant.generate_response("Hello")
    print("✅ Generation successful")
    print(f"Response: {result.get('response', 'No response')[:50]}...")
    
    print("4. Testing with TTS enabled...")
    assistant_tts = TinyLlamaUnifiedTTS(enable_tts=True, use_quantization=False)
    print("✅ TTS Assistant initialized")
    
    print("5. Testing TTS generation...")
    result = assistant_tts.generate_response("Test TTS")
    print("✅ TTS Generation successful")
    
    signal.alarm(0)  # Cancel timeout
    print("🎉 All tests passed - no hanging detected!")
    
except Exception as e:
    signal.alarm(0)
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
Test script to verify the response extraction fix
"""

import os
import sys

# Add the tinyllama directory to the path
sys.path.insert(0, '/home/jason/projects/tinyllama')

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import and test
from tinyllama_unified_tts import TinyLlamaUnifiedTTS

def test_response_extraction():
    """Test that responses are extracted correctly without missing characters"""
    print("🧪 Testing response extraction fix...")
    
    # Initialize assistant without TTS to focus on the text issue
    assistant = TinyLlamaUnifiedTTS(enable_tts=False, use_quantization=False)
    
    test_queries = [
        "What is Python?",
        "How do I create a list?",
        "Explain variables",
        "Write a hello world program"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        
        result = assistant.generate_response(query, use_cache=False)
        response = result.get("response", "")
        
        print(f"Response length: {len(response)} characters")
        print(f"First 50 characters: '{response[:50]}'")
        print(f"Starts with letter/word: {response[0].isalpha() if response else 'Empty'}")
        
        # Check if response seems complete
        if response and response[0].isalpha():
            print("✅ Response appears to start correctly")
        elif response:
            print(f"⚠️  Response starts with: '{response[0]}' (char code: {ord(response[0])})")
        else:
            print("❌ Empty response")
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_response_extraction()

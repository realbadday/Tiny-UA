#!/usr/bin/env python3
"""
Stability test for TinyLlama
Tests all major features to ensure nothing breaks
"""

import subprocess
import time
import sys

def test_feature(name, command_input):
    """Test a feature by sending commands to the assistant"""
    print(f"\n🧪 Testing: {name}")
    try:
        process = subprocess.Popen(
            ['python3', 'main.py', '--no-tts'],  # No TTS for faster testing
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send commands
        output, error = process.communicate(input=command_input, timeout=30)
        
        if process.returncode == 0:
            print(f"✅ {name} - PASSED")
            return True
        else:
            print(f"❌ {name} - FAILED")
            print(f"Error: {error}")
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"⏰ {name} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {name} - ERROR: {e}")
        return False

def main():
    print("🔍 TinyLlama Stability Test")
    print("=" * 50)
    
    tests = [
        # Test basic query
        ("Basic Query", "What is Python?\nexit\n"),
        
        # Test mode switching
        ("Mode Switch", "/mode code\nwrite hello world\nexit\n"),
        
        # Test TTS toggle (won't speak, just toggle)
        ("TTS Toggle", "/tts\n/tts\nexit\n"),
        
        # Test help
        ("Help Command", "/help\nexit\n"),
        
        # Test modes listing
        ("List Modes", "/modes\nexit\n"),
        
        # Test multiple mode switches
        ("Multiple Mode Switches", "/mode code\n/mode chat\n/mode function\nexit\n"),
        
        # Test cache clear
        ("Clear Cache", "/clear\nexit\n"),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_input in tests:
        if test_feature(test_name, test_input):
            passed += 1
        else:
            failed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! TinyLlama is stable.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

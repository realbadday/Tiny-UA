#!/usr/bin/env python3
"""
Test script for TinyLlama Programming Assistant
Verifies installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test required imports"""
    print("🔍 Testing imports...")
    try:
        import torch
        print("✅ torch imported successfully")
        
        import transformers
        print("✅ transformers imported successfully")
        
        try:
            import peft
            print("✅ peft imported (fine-tuning available)")
        except ImportError:
            print("⚠️  peft not installed (fine-tuning disabled)")
        
        try:
            import pyttsx4
            print("✅ pyttsx4 imported (TTS available)")
        except ImportError:
            print("⚠️  pyttsx4 not installed (TTS disabled)")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test basic model loading"""
    print("\n🔍 Testing model loading (this may download ~1.1GB on first run)...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model size: {model.num_parameters():,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_generation():
    """Test basic text generation"""
    print("\n🔍 Testing text generation...")
    try:
        from tinyllama_chat import TinyLlamaChat
        
        chat = TinyLlamaChat(use_quantization=False)  # Skip quantization for test
        
        test_queries = [
            ("What is a Python list?", "explain"),
            ("function to reverse a string", "code"),
        ]
        
        for query, mode in test_queries:
            print(f"\nTest: {query} (mode: {mode})")
            response = chat.generate_response(query, mode=mode, use_cache=False)
            print(f"Response preview: {response[:100]}...")
            
            if response and len(response) > 10:
                print("✅ Generation successful")
            else:
                print("❌ Generation failed or too short")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_enhanced_features():
    """Test enhanced assistant features"""
    print("\n🔍 Testing enhanced features...")
    try:
        from programming_assistant import ProgrammingAssistant
        
        assistant = ProgrammingAssistant(use_quantization=False)
        
        # Test planning
        plan = assistant.plan_task("create a todo app")
        if "1." in plan:
            print("✅ Planning feature works")
        else:
            print("❌ Planning feature failed")
        
        # Test code review
        code = "def add(a,b): return a+b"
        review = assistant.review_code(code)
        if review and len(review) > 20:
            print("✅ Code review feature works")
        else:
            print("❌ Code review feature failed")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TinyLlama Programming Assistant Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Generation", test_generation),
        ("Enhanced Features", test_enhanced_features),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Running {name} Test...")
        print("-" * 40)
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! TinyLlama is ready to use.")
        print("\n🚀 Quick start:")
        print("   python ~/tinyllama/tinyllama_chat.py")
        print("   python ~/tinyllama/programming_assistant.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Run setup.sh to install missing dependencies.")

if __name__ == "__main__":
    # Add tinyllama directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

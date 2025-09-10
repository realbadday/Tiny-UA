#!/usr/bin/env python3
"""
Minimal test to isolate hanging issue
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_direct_generation():
    """Test direct model generation without the wrapper"""
    print("🔍 Testing direct model generation...")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    print("2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    print("✅ Model loaded")
    
    print("3. Testing generation...")
    prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello</s>\n<|assistant|>\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    print(f"✅ Generation successful!")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_direct_generation()

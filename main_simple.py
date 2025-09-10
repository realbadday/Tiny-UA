#!/usr/bin/env python3
"""
TinyLlama Minimal - Maximum Stability Version
Fewer features, but rock solid
"""

import os
import sys

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DISABLE_TENSORFLOW'] = '1'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)


class SimpleAssistant:
    """Minimal assistant with just core features"""
    
    def __init__(self):
        print("🤖 Loading TinyLlama...")
        
        # Load model
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load with quantization for efficiency
        print("📊 Applying quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()
        
        print("✅ Ready!\n")
    
    def generate(self, query):
        """Generate a response"""
        # Simple prompt format
        prompt = f"<|system|>\nYou are a helpful programming assistant.</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        print("🔄 Generating...")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    def run(self):
        """Run interactive mode"""
        print("💬 TinyLlama Simple Mode")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                query = input(">>> ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                print("\n" + "=" * 50)
                response = self.generate(query)
                print(response)
                print("=" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    assistant = SimpleAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

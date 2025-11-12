#!/usr/bin/env python3
"""
TinyLlama CodeGen - Enhanced Programming Assistant with Code Generation
Combines TinyLlama's efficiency with specialized code generation features
"""

import os
import sys

# CRITICAL: Disable TensorFlow before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DISABLE_TENSORFLOW'] = '1'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
import json
import time
from pathlib import Path
import re

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)


class CodeStoppingCriteria(StoppingCriteria):
    """Stop generation when we have complete code"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Check if we have a complete function/class
        if "```" in text and text.count("```") >= 2:
            return True
        
        # Check for natural code endings
        lines = text.split('\n')
        if len(lines) > 10:  # Reasonable code length
            last_line = lines[-1].strip()
            if last_line == "" and lines[-2].strip() in ["return", "pass", "}", "```"]:
                return True
                
        return False


class TinyLlamaCodeGen:
    """Enhanced TinyLlama for code generation and programming assistance"""
    
    def __init__(self, model_path=None, use_quantization=True):
        """
        Initialize TinyLlama CodeGen
        Args:
            model_path: Path to model or HF model name
            use_quantization: Use INT8 quantization for CPU efficiency
        """
        # Check for fine-tuned model
        if model_path is None:
            fine_tuned_path = Path.home() / "tinyllama" / "models" / "tinyllama-python-codegen"
            if fine_tuned_path.exists():
                print(f"🎯 Loading fine-tuned CodeGen model from {fine_tuned_path}")
                model_path = str(fine_tuned_path)
                self.is_finetuned = True
            else:
                print("🤖 Loading TinyLlama-1.1B-Chat-v1.0...")
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.is_finetuned = False
        else:
            self.is_finetuned = "codegen" in str(model_path).lower()
        
        print("⚙️  Optimizing for code generation...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CPU optimization
        if use_quantization:
            try:
                print("📊 Applying INT8 quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("✅ Quantization successful")
            except Exception as e:
                print(f"⚠️  Quantization failed, using float16: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
        
        self.model.eval()
        
        # Code templates for different generation tasks
        self.code_templates = {
            "function": "Write a Python function that {description}:\n```python\n",
            "class": "Write a Python class for {description}:\n```python\n",
            "script": "Write a complete Python script that {description}:\n```python\n",
            "fix": "Fix this Python code:\n```python\n{code}\n```\nFixed version:\n```python\n",
            "improve": "Improve this Python code for better {aspect}:\n```python\n{code}\n```\nImproved version:\n```python\n",
            "explain": "Explain this Python code step by step:\n```python\n{code}\n```\nExplanation:\n",
            "test": "Write pytest unit tests for this function:\n```python\n{code}\n```\nTests:\n```python\n",
            "convert": "Convert this {source_lang} code to Python:\n```{source_lang}\n{code}\n```\nPython version:\n```python\n",
            "algorithm": "Implement the {algorithm_name} algorithm in Python:\n```python\n"
        }
        
        # Common programming patterns
        self.patterns = {
            "api": "REST API endpoint",
            "cli": "command-line interface",
            "gui": "graphical user interface",
            "web": "web application",
            "ml": "machine learning model",
            "data": "data processing pipeline",
            "async": "asynchronous function",
            "decorator": "decorator function",
            "generator": "generator function",
            "context": "context manager"
        }
        
        # Response cache
        self.cache_file = Path.home() / "tinyllama" / "codegen_cache.json"
        self.response_cache = self._load_cache()
        
        print("✅ CodeGen model ready!")
        print("-" * 50)
    
    def _load_cache(self):
        """Load response cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """Save response cache"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except:
            pass
    
    def format_codegen_prompt(self, task_type, **kwargs):
        """Format prompt for specific code generation task"""
        if task_type in self.code_templates:
            return self.code_templates[task_type].format(**kwargs)
        else:
            # Default format
            return f"<|system|>\nYou are an expert Python programmer. Generate clean, efficient, and well-documented code.</s>\n<|user|>\n{kwargs.get('query', '')}</s>\n<|assistant|>\n"
    
    def extract_code_blocks(self, text):
        """Extract code blocks from response"""
        code_blocks = []
        
        # Find code blocks with ``` markers
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            code_blocks.extend(matches)
        else:
            # Try to find indented code
            lines = text.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.startswith('    ') or line.startswith('\t'):
                    in_code = True
                    code_lines.append(line)
                elif in_code and line.strip() == '':
                    code_lines.append(line)
                elif in_code and not line.startswith(' '):
                    if code_lines:
                        code_blocks.append('\n'.join(code_lines))
                        code_lines = []
                    in_code = False
            
            if code_lines:
                code_blocks.append('\n'.join(code_lines))
        
        return code_blocks
    
    def generate_code(self, task_type, **kwargs):
        """Generate code for specific task"""
        # Check cache
        cache_key = f"{task_type}:{json.dumps(kwargs, sort_keys=True)}"
        if cache_key in self.response_cache:
            print("💾 Using cached response...")
            return self.response_cache[cache_key]
        
        # Format prompt
        prompt = self.format_codegen_prompt(task_type, **kwargs)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(inputs)
        
        # Generation config optimized for code
        generation_config = {
            "max_new_tokens": 512,  # Longer for code
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "use_cache": True,
        }
        
        # Add stopping criteria for code
        stopping_criteria = StoppingCriteriaList([CodeStoppingCriteria(self.tokenizer)])
        
        # Generate
        print("🔄 Generating code...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Extract code blocks
        code_blocks = self.extract_code_blocks(response)
        
        # Format result
        result = {
            "response": response,
            "code_blocks": code_blocks,
            "generation_time": generation_time
        }
        
        # Cache result
        if len(self.response_cache) < 500:
            self.response_cache[cache_key] = result
            self._save_cache()
        
        if generation_time > 2.0:
            print(f"⏱️  Generated in {generation_time:.1f}s")
        
        return result
    
    def generate_function(self, description):
        """Generate a Python function"""
        return self.generate_code("function", description=description)
    
    def generate_class(self, description):
        """Generate a Python class"""
        return self.generate_code("class", description=description)
    
    def generate_script(self, description):
        """Generate a complete Python script"""
        return self.generate_code("script", description=description)
    
    def fix_code(self, code):
        """Fix buggy Python code"""
        return self.generate_code("fix", code=code)
    
    def improve_code(self, code, aspect="efficiency"):
        """Improve code for specific aspect"""
        return self.generate_code("improve", code=code, aspect=aspect)
    
    def explain_code(self, code):
        """Explain Python code"""
        return self.generate_code("explain", code=code)
    
    def generate_tests(self, code):
        """Generate unit tests for code"""
        return self.generate_code("test", code=code)
    
    def convert_code(self, code, source_lang="JavaScript"):
        """Convert code from another language to Python"""
        return self.generate_code("convert", code=code, source_lang=source_lang)
    
    def generate_algorithm(self, algorithm_name):
        """Generate implementation of specific algorithm"""
        return self.generate_code("algorithm", algorithm_name=algorithm_name)
    
    def interactive_codegen(self):
        """Interactive code generation interface"""
        print("\n💻 TinyLlama CodeGen - Python Code Generator")
        print("=" * 50)
        print("Commands:")
        print("  /function <description> - Generate a function")
        print("  /class <description>    - Generate a class")
        print("  /script <description>   - Generate complete script")
        print("  /fix                    - Fix code (multiline input)")
        print("  /improve <aspect>       - Improve code")
        print("  /explain               - Explain code")
        print("  /test                  - Generate tests")
        print("  /convert <lang>        - Convert from another language")
        print("  /algorithm <name>      - Implement an algorithm")
        print("  /patterns              - Show available patterns")
        print("  /clear                 - Clear cache")
        print("  /help                  - Show this help")
        print("  quit                   - Exit")
        print("-" * 50 + "\n")
        
        multiline_mode = False
        multiline_buffer = []
        current_command = None
        
        while True:
            try:
                if multiline_mode:
                    prompt = "... "
                else:
                    prompt = ">>> "
                
                user_input = input(prompt)
                
                # Handle multiline mode
                if multiline_mode:
                    # Don't strip in multiline mode to preserve formatting
                    if user_input.strip() == "" or user_input.strip() == "```":
                        # Process if we have any content in buffer
                        if multiline_buffer:
                            multiline_mode = False
                            code = '\n'.join(multiline_buffer)
                            multiline_buffer = []
                        else:
                            # Empty buffer - exit multiline mode without processing
                            multiline_mode = False
                            continue
                    else:
                        multiline_buffer.append(user_input)
                        continue
                
                # Strip input for command processing
                user_input = user_input.strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input == '/help':
                    self.interactive_codegen()  # Show help again
                    continue
                
                if user_input == '/clear':
                    self.response_cache = {}
                    self._save_cache()
                    print("🗑️  Cache cleared")
                    continue
                
                if user_input == '/patterns':
                    print("\n📚 Available patterns:")
                    for key, desc in self.patterns.items():
                        print(f"  - {key}: {desc}")
                    print()
                    continue
                
                # Parse commands
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0][1:]  # Remove /
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command == "function" and args:
                        result = self.generate_function(args)
                        self._display_result(result)
                    
                    elif command == "class" and args:
                        result = self.generate_class(args)
                        self._display_result(result)
                    
                    elif command == "script" and args:
                        result = self.generate_script(args)
                        self._display_result(result)
                    
                    elif command == "algorithm" and args:
                        result = self.generate_algorithm(args)
                        self._display_result(result)
                    
                    elif command in ["fix", "improve", "explain", "test", "convert"]:
                        print(f"📝 Enter code to {command} (press Enter on empty line or type '```' when done):")
                        multiline_mode = True
                        current_command = command
                        multiline_buffer = []
                    
                    else:
                        print(f"❌ Unknown command: /{command}")
                
                elif user_input:
                    # Direct query - treat as function generation
                    result = self.generate_function(user_input)
                    self._display_result(result)
                        if current_command == "fix":
                            result = self.fix_code(code)
                        elif current_command == "improve":
                            aspect = "efficiency"  # default
                            result = self.improve_code(code, aspect)
                        elif current_command == "explain":
                            result = self.explain_code(code)
                        elif current_command == "test":
                            result = self.generate_tests(code)
                        elif current_command == "convert":
                            lang = "JavaScript"  # default
                            result = self.convert_code(code, lang)
                        
                        self._display_result(result)
                        current_command = None
                    else:
                        multiline_buffer.append(user_input)
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input == '/help':
                    self.interactive_codegen()  # Show help again
                    continue
                
                if user_input == '/clear':
                    self.response_cache = {}
                    self._save_cache()
                    print("🗑️  Cache cleared")
                    continue
                
                if user_input == '/patterns':
                    print("\n📚 Available patterns:")
                    for key, desc in self.patterns.items():
                        print(f"  - {key}: {desc}")
                    print()
                    continue
                
                # Parse commands
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0][1:]  # Remove /
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command == "function" and args:
                        result = self.generate_function(args)
                        self._display_result(result)
                    
                    elif command == "class" and args:
                        result = self.generate_class(args)
                        self._display_result(result)
                    
                    elif command == "script" and args:
                        result = self.generate_script(args)
                        self._display_result(result)
                    
                    elif command == "algorithm" and args:
                        result = self.generate_algorithm(args)
                        self._display_result(result)
                    
                    elif command in ["fix", "improve", "explain", "test", "convert"]:
                        print(f"📝 Enter code to {command} (press Enter on empty line or type '```' when done):")
                        multiline_mode = True
                        current_command = command
                        multiline_buffer = []
                    
                    else:
                        print(f"❌ Unknown command: /{command}")
                
                elif user_input:
                    # Direct query - treat as function generation
                    result = self.generate_function(user_input)
                    self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _display_result(self, result):
        """Display generation result"""
        print("\n" + "=" * 50)
        
        if isinstance(result, dict):
            if result.get("code_blocks"):
                print("📄 Generated Code:")
                for i, code in enumerate(result["code_blocks"]):
                    if len(result["code_blocks"]) > 1:
                        print(f"\n--- Code Block {i+1} ---")
                    print("```python")
                    print(code)
                    print("```")
            else:
                print(result.get("response", "No code generated"))
            
            if result.get("generation_time", 0) > 2.0:
                print(f"\n⏱️  Generation time: {result['generation_time']:.1f}s")
        else:
            print(result)
        
        print("=" * 50 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyLlama CodeGen - Python Code Generator")
    parser.add_argument('command', nargs='?', help='Command: function, class, script, etc.')
    parser.add_argument('description', nargs='*', help='Description of what to generate')
    parser.add_argument('--model-path', help='Path to model')
    parser.add_argument('--no-quantization', action='store_true', help='Disable quantization')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    print("\n🚀 TinyLlama CodeGen")
    print("=" * 50)
    
    # Initialize
    codegen = TinyLlamaCodeGen(
        model_path=args.model_path,
        use_quantization=not args.no_quantization
    )
    
    # Process command line arguments
    if args.command:
        if args.command == "function" and args.description:
            result = codegen.generate_function(" ".join(args.description))
            codegen._display_result(result)
        elif args.command == "class" and args.description:
            result = codegen.generate_class(" ".join(args.description))
            codegen._display_result(result)
        elif args.command == "script" and args.description:
            result = codegen.generate_script(" ".join(args.description))
            codegen._display_result(result)
        elif args.command == "algorithm" and args.description:
            result = codegen.generate_algorithm(" ".join(args.description))
            codegen._display_result(result)
        else:
            # Treat as function description
            result = codegen.generate_function(" ".join([args.command] + args.description))
            codegen._display_result(result)
    else:
        # Interactive mode
        codegen.interactive_codegen()


if __name__ == "__main__":
    main()

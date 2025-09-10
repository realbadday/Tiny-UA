#!/usr/bin/env python3
"""
TinyLlama Unified TTS - Complete Programming Assistant with Voice Support
Combines chat, code generation, and text-to-speech capabilities
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
import re
from pathlib import Path

# Import TTS utilities
from tts_utils import TTSManager, TTSCommands

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)


class CodeStoppingCriteria(StoppingCriteria):
    """Stop generation when we have complete code"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if "```" in text and text.count("```") >= 2:
            return True
        lines = text.split('\n')
        if len(lines) > 10:
            last_line = lines[-1].strip()
            if last_line == "" and lines[-2].strip() in ["return", "pass", "}", "```"]:
                return True
        return False


class TinyLlamaUnifiedTTS:
    """Unified TinyLlama with chat, code generation, and TTS support"""
    
    def __init__(self, model_path=None, use_quantization=True, enable_tts=True):
        """
        Initialize Unified TinyLlama with TTS
        Args:
            model_path: Path to model or HF model name
            use_quantization: Use INT8 quantization
            enable_tts: Enable TTS on startup
        """
        # Initialize TTS first
        self.tts = TTSManager(auto_init=enable_tts)
        self.tts_commands = TTSCommands(self.tts)
        
        # Announce startup
        if self.tts.enabled:
            self.tts.speak("Starting TinyLlama unified assistant", interrupt=True, is_status=True)
        
        # Model selection
        if model_path is None:
            fine_tuned_path = Path.home() / "tinyllama" / "models" / "tinyllama-unified"
            if fine_tuned_path.exists():
                print(f"🎯 Loading fine-tuned unified model from {fine_tuned_path}")
                model_path = str(fine_tuned_path)
                self.is_finetuned = True
            else:
                print("🤖 Loading TinyLlama-1.1B-Chat-v1.0...")
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.is_finetuned = False
        
        print("⚙️  Initializing unified assistant...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
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
                print(f"⚠️  Quantization failed: {e}")
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
        
        # Initialize modes and templates
        self.current_mode = "chat"
        self.modes = {
            "chat": "💬 General chat",
            "code": "💻 Code generation", 
            "explain": "📖 Explanations",
            "debug": "🐛 Debug code",
            "function": "🔧 Generate functions",
            "class": "📦 Generate classes",
            "script": "📄 Generate scripts",
            "test": "🧪 Generate tests",
            "improve": "⚡ Improve code",
            "convert": "🔄 Convert code"
        }
        
        # Code generation templates
        self.code_templates = {
            "function": "Write a Python function that {description}:\n```python\n",
            "class": "Write a Python class for {description}:\n```python\n",
            "script": "Write a complete Python script that {description}:\n```python\n",
            "fix": "Fix this Python code:\n```python\n{code}\n```\nFixed version:\n```python\n",
            "improve": "Improve this Python code for better {aspect}:\n```python\n{code}\n```\nImproved version:\n```python\n",
            "explain": "Explain this Python code step by step:\n```python\n{code}\n```\nExplanation:\n",
            "test": "Write pytest unit tests for this function:\n```python\n{code}\n```\nTests:\n```python\n",
            "convert": "Convert this {source_lang} code to Python:\n```{source_lang}\n{code}\n```\nPython version:\n```python\n"
        }
        
        # Response cache
        self.cache_file = Path.home() / "tinyllama" / "unified_cache.json"
        self.response_cache = self._load_cache()
        
        print("✅ Unified assistant ready!")
        
        # Announce ready
        if self.tts.enabled:
            self.tts.speak("TinyLlama unified assistant is ready", is_status=True)
        
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
    
    def format_prompt(self, query, mode=None):
        """Format prompt based on mode"""
        if mode is None:
            mode = self.current_mode
        
        if mode in ["function", "class", "script"]:
            return self.code_templates[mode].format(description=query)
        elif mode in ["fix", "improve", "explain", "test", "convert"]:
            # Format with template for code-based modes
            if mode == "improve":
                return self.code_templates[mode].format(code=query, aspect="efficiency")
            elif mode in ["fix", "explain", "test"]:
                return self.code_templates[mode].format(code=query)
            elif mode == "convert":
                return self.code_templates[mode].format(code=query, source_lang="JavaScript")
            else:
                return query
        else:
            # Chat format
            system_prompt = "You are a helpful programming assistant. Provide clear, concise answers."
            
            if mode == "code":
                user_prompt = f"Write Python code for: {query}"
            elif mode == "explain":
                user_prompt = f"Explain this concept: {query}"
            elif mode == "debug":
                user_prompt = f"Debug this issue: {query}"
            else:
                user_prompt = query
            
            return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    
    def extract_code_blocks(self, text):
        """Extract code blocks from response"""
        code_blocks = []
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code_blocks.extend(matches)
        return code_blocks
    
    def generate_response(self, query, mode=None, use_cache=True):
        """Generate response with optional TTS"""
        if mode is None:
            mode = self.current_mode
        
        # Check cache
        cache_key = f"{mode}:{query}"
        if use_cache and cache_key in self.response_cache:
            print("💾 Using cached response...")
            result = self.response_cache[cache_key]
            if self.tts.enabled:
                self._speak_result(result, mode)
            return result
        
        # Announce generation (silenced)
        
        # Format prompt
        prompt = self.format_prompt(query, mode)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(inputs)
        
        # Generation config
        is_code_mode = mode in ["function", "class", "script", "code", "test", "improve"]
        generation_config = {
            "max_new_tokens": 512 if is_code_mode else 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }
        
        # Add stopping criteria for code
        if is_code_mode:
            stopping_criteria = StoppingCriteriaList([CodeStoppingCriteria(self.tokenizer)])
            generation_config["stopping_criteria"] = stopping_criteria
        
        # Generate
        print("🔄 Generating...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode - use a more robust method to extract the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Find the assistant's response more reliably
        if "<|assistant|>" in full_response:
            # Split at the assistant marker and take everything after
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            # Fallback: try to find where the prompt ends more carefully
            # Decode just the input to see what it actually looks like
            input_decoded = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            if input_decoded in full_response:
                response = full_response[len(input_decoded):].strip()
            else:
                # Final fallback: use original method but be more conservative
                response = full_response[len(prompt):].strip()
        
        # Debug: Check if response seems truncated (optional debug output)
        if os.environ.get('TINYLLAMA_DEBUG') == '1':
            print(f"DEBUG: Full response length: {len(full_response)}")
            print(f"DEBUG: Prompt length: {len(prompt)}")
            print(f"DEBUG: Extracted response length: {len(response)}")
            print(f"DEBUG: Response starts with: '{response[:20]}...'" if response else "DEBUG: Empty response")
        
        # Extract code blocks if applicable
        code_blocks = self.extract_code_blocks(response) if is_code_mode else []
        
        # Format result
        result = {
            "response": response,
            "code_blocks": code_blocks,
            "generation_time": generation_time,
            "mode": mode
        }
        
        # Cache result
        if use_cache and len(self.response_cache) < 500:
            self.response_cache[cache_key] = result
            self._save_cache()
        
        # Speak result
        if self.tts.enabled:
            self._speak_result(result, mode)
        
        return result
    
    def _speak_result(self, result, mode):
        """Speak the result based on mode"""
        if isinstance(result, dict):
            if result.get("code_blocks"):
                # For code, speak a summary
                num_blocks = len(result["code_blocks"])
                total_lines = sum(len(block.split('\n')) for block in result["code_blocks"])
                self.tts.speak(f"Generated {num_blocks} code blocks with {total_lines} total lines")
            else:
                # For text responses, speak the content
                text = result.get("response", "")
                if len(text) > 200:
                    # Summarize long responses (no status suffix)
                    sentences = text.split('. ')
                    summary = '. '.join(sentences[:2])
                    self.tts.speak(summary)
                else:
                    self.tts.speak(text)
        else:
            self.tts.speak(str(result))
    
    def interactive_unified(self):
        """Unified interactive interface with TTS"""
        print("\n🚀 TinyLlama Unified Assistant with Voice Support")
        print("=" * 60)
        print("Available modes:", ", ".join(f"{k}" for k in self.modes.keys()))
        print("\n📋 Commands:")
        print("  /mode <name>    - Switch mode")
        print("  /modes          - List all modes")
        print("  /tts            - Toggle voice output")
        print("  /voices         - List available voices")
        print("  /rate <wpm>     - Set speech rate")
        print("  /volume <0-1>   - Set volume")
        print("  /multiline      - Enter multiline input")
        print("  /clear          - Clear cache")
        print("  /train          - Launch training interface")
        print("  /help           - Show help")
        print("  quit            - Exit")
        
        if self.tts.enabled:
            print("\n🔊 Voice output is ENABLED")
            self.tts.speak("Voice output is enabled. Say 'slash tts' to toggle.")
        else:
            print("\n🔇 Voice output is DISABLED")
        
        print(f"\nCurrent mode: {self.current_mode}")
        print("-" * 60 + "\n")
        
        multiline_mode = False
        multiline_buffer = []
        
        while True:
            try:
                # Prompt
                if multiline_mode:
                    prompt = "... "
                else:
                    icon = self.modes[self.current_mode].split()[0]
                    prompt = f"[{icon} {self.current_mode.upper()}] >>> "
                
                user_input = input(prompt).strip()
                
                # Handle multiline
                if multiline_mode:
                    if user_input in ["EOF", "```"]:
                        multiline_mode = False
                        user_input = '\n'.join(multiline_buffer)
                        multiline_buffer = []
                    else:
                        multiline_buffer.append(user_input)
                        continue
                
                # Exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    if self.tts.enabled:
                        self.tts.speak("Goodbye!")
                    print("\n👋 Goodbye!")
                    break
                
                # TTS commands
                if self.tts_commands.handle_command(user_input):
                    continue
                
                # Other commands
                if user_input == '/help':
                    self._show_help()
                    continue
                
                if user_input == '/modes':
                    self._list_modes()
                    continue
                
                if user_input.startswith('/mode'):
                    # Accept flexible syntax and case-insensitive matches
                    tokens = re.split(r'[\s=:,]+', user_input.strip())
                    target = None
                    for tok in tokens[1:]:
                        cand = tok.lower()
                        if cand in self.modes:
                            target = cand
                            break
                    if target:
                        self.current_mode = target
                        mode_desc = self.modes[self.current_mode]
                        print(f"📝 Switched to {mode_desc}")
                        if self.tts.enabled:
                            self.tts.speak(f"Switched to {target} mode")
                    else:
                        print("❌ Invalid mode. Use /modes to see available modes.")
                    continue
                
                if user_input == '/multiline':
                    print("📝 Multiline mode - Enter 'EOF' or '```' to finish")
                    multiline_mode = True
                    continue
                
                if user_input == '/clear':
                    self.response_cache = {}
                    self._save_cache()
                    print("🗑️  Cache cleared")
                    if self.tts.enabled:
                        self.tts.speak("Cache cleared")
                    continue
                
                if user_input == '/train':
                    print("🎓 Launching training interface...")
                    if self.tts.enabled:
                        self.tts.speak("Launching training interface")
                    self._launch_training()
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("\n" + "=" * 60)
                
                # Handle special modes that need code input
                if self.current_mode in ["fix", "improve", "explain", "test", "convert"]:
                    print(f"📝 Enter code to {self.current_mode} (type 'EOF' when done):")
                    multiline_mode = True
                    multiline_buffer = [user_input]  # Save the context
                    continue
                
                # Generate response
                result = self.generate_response(user_input)
                
                # Display result
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
                        print(result.get("response", "No response generated"))
                    
                    if result.get("generation_time", 0) > 2.0:
                        print(f"\n⏱️  Generation time: {result['generation_time']:.1f}s")
                else:
                    print(result)
                
                print("=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                if self.tts.enabled:
                    self.tts.speak("Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                if self.tts.enabled:
                    self.tts.speak(f"Error: {str(e)[:50]}")
    
    def _show_help(self):
        """Show comprehensive help"""
        help_text = """
📚 TinyLlama Unified Assistant Help

🎯 Modes:
- chat      : General conversation
- code      : Generate code snippets
- explain   : Explain concepts
- debug     : Debug issues
- function  : Generate functions
- class     : Generate classes
- script    : Generate complete scripts
- test      : Generate unit tests
- improve   : Improve existing code
- convert   : Convert code between languages

🗣️ Voice Commands:
- /tts         : Toggle voice output
- /voices      : List available voices
- /voice <name>: Change voice
- /rate <wpm>  : Set speech rate (50-300)
- /volume <0-1>: Set volume

🎓 Training Commands:
- /train       : Launch training interface

💡 Usage Examples:
[CHAT mode]
>>> What's the difference between list and tuple?

[FUNCTION mode]
>>> binary search in a sorted list

[CODE mode]  
>>> REST API endpoint for user authentication

[TEST mode]
>>> def add(a, b): return a + b
... EOF

🚀 Tips:
- Use specific modes for better results
- Responses are cached for speed
- Voice output summarizes long code
- Use /multiline for complex input
        """
        print(help_text)
        if self.tts.enabled:
            self.tts.speak("Help displayed. Check the screen for details.")
    
    def _list_modes(self):
        """List all available modes"""
        print("\n📋 Available Modes:")
        for mode, desc in self.modes.items():
            marker = "→" if mode == self.current_mode else " "
            print(f" {marker} {mode:10} - {desc}")
        print()
        
        if self.tts.enabled:
            self.tts.speak(f"Currently in {self.current_mode} mode. {len(self.modes)} modes available.")
    
    def _launch_training(self):
        """Launch the training interface"""
        import os
        
        print("\n🎓 Training Interface")
        print("For training options, please run:")
        print("💉 ./launch_training.py")
        print("or")
        print("💉 python3 launch_training.py")
        print()
        print("This keeps the training separate from the chat interface.")
        print("You can also run training directly with:")
        print("💉 python3 train_tinyllama.py --help")
        print()
    
    def cleanup(self):
        """Clean up resources"""
        if self.tts:
            self.tts.shutdown()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyLlama Unified Assistant with TTS")
    parser.add_argument('query', nargs='*', help='Direct query')
    parser.add_argument('-m', '--mode', 
                        choices=['chat', 'code', 'explain', 'debug', 'function', 'class', 'script', 'test', 'improve', 'convert'],
                        default='chat', help='Initial mode')
    parser.add_argument('--tts', action='store_true', help='Enable text-to-speech output')
    parser.add_argument('--no-tts', action='store_true', help='Explicitly disable text-to-speech')
    parser.add_argument('--tts-rate', type=int, default=175, help='TTS speech rate in WPM (50-300)')
    parser.add_argument('--tts-volume', type=float, default=0.9, help='TTS volume level (0.0-1.0)')
    parser.add_argument('--model-path', help='Path to model directory or HuggingFace model name')
    parser.add_argument('--no-quantization', action='store_true', help='Disable INT8 quantization (uses more memory)')
    
    args = parser.parse_args()
    
    print("\n🚀 TinyLlama Unified Assistant")
    print("=" * 50)
    
    # Determine TTS setting
    enable_tts = args.tts and not args.no_tts
    
    # Initialize
    assistant = TinyLlamaUnifiedTTS(
        model_path=args.model_path,
        use_quantization=not args.no_quantization,
        enable_tts=enable_tts
    )
    
    # Set TTS parameters
    if assistant.tts.enabled:
        assistant.tts.set_rate(args.tts_rate)
        assistant.tts.set_volume(args.tts_volume)
    
    # Set initial mode
    assistant.current_mode = args.mode
    
    # Process query if provided
    if args.query:
        query = " ".join(args.query)
        result = assistant.generate_response(query)
        
        # Display result
        print("\n" + "=" * 50)
        if isinstance(result, dict) and result.get("code_blocks"):
            for code in result["code_blocks"]:
                print("```python")
                print(code)
                print("```")
        else:
            print(result.get("response", result))
        print("=" * 50)
        
        # Wait for speech to complete
        if assistant.tts.enabled:
            assistant.tts.wait_for_speech()
    else:
        # Interactive mode
        try:
            assistant.interactive_unified()
        finally:
            assistant.cleanup()


if __name__ == "__main__":
    main()

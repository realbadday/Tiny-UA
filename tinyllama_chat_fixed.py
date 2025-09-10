#!/usr/bin/env python3
"""
TinyLlama Chat Interface - Efficient Programming Assistant
Designed for CPU-only inference with minimal resource usage
Fixed version that avoids TensorFlow conflicts
"""

import os
import sys

# CRITICAL: Disable TensorFlow before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DISABLE_TENSORFLOW'] = '1'

# Force transformers to use PyTorch only
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import warnings
warnings.filterwarnings('ignore')

# Now safe to import PyTorch
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
import json
import time
import re
from pathlib import Path

# Import TTS utilities
try:
    from tts_utils import TTSManager, TTSCommands
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  TTS module not found. Voice features disabled.")

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)  # Optimize for 4-core CPU


class TinyLlamaChat:
    """Efficient TinyLlama chat interface for programming assistance with TTS support"""
    
    def __init__(self, model_path=None, use_quantization=True, enable_tts=False, tts_rate=175, tts_volume=0.9):
        """
        Initialize TinyLlama chat with TTS support
        Args:
            model_path: Path to model or HF model name
            use_quantization: Use INT8 quantization for CPU efficiency
            enable_tts: Enable TTS on startup
            tts_rate: Speech rate in words per minute
            tts_volume: Volume level (0.0 to 1.0)
        """
        # Initialize TTS manager and command handler (allow toggling even if not started with --tts)
        self.tts = TTSManager(auto_init=enable_tts) if TTS_AVAILABLE else None
        self.tts_commands = TTSCommands(self.tts) if TTS_AVAILABLE else None
        if self.tts:
            if tts_rate:
                try:
                    self.tts.set_rate(tts_rate)
                except Exception:
                    pass
            if tts_volume is not None:
                try:
                    self.tts.set_volume(tts_volume)
                except Exception:
                    pass
            if self.tts.enabled:
                self.tts.speak("Initializing TinyLlama chat", interrupt=True)
        # Model selection
        if model_path is None:
            # Check for fine-tuned model first
            fine_tuned_path = Path.home() / "tinyllama" / "models" / "tinyllama-python-qa"
            if fine_tuned_path.exists():
                print(f"🎯 Loading fine-tuned model from {fine_tuned_path}")
                model_path = str(fine_tuned_path)
                self.is_finetuned = True
            else:
                print("🤖 Loading TinyLlama-1.1B-Chat-v1.0 from HuggingFace...")
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.is_finetuned = False
        else:
            self.is_finetuned = "python-qa" in str(model_path).lower()
        
        print("⚙️  Optimizing for CPU inference...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # CPU-optimized loading
        if use_quantization:
            try:
                print("📊 Attempting INT8 quantization for efficiency...")
                # For CPU, we'll use torch.qint8 dynamic quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Start with float32
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                # Apply dynamic quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("✅ INT8 quantization applied successfully")
            except Exception as e:
                print(f"⚠️  Quantization failed: {e}")
                print("📱 Loading with float16 precision...")
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
        
        # Response cache for common queries
        self.cache_file = Path.home() / "tinyllama" / "response_cache.json"
        self.response_cache = self._load_cache()
        
        print(f"✅ Model loaded successfully!")
        print(f"📚 {'Fine-tuned for Python Q&A' if self.is_finetuned else 'Base chat model'}")
        
        if self.tts and self.tts.enabled:
            print("🔊 Voice output enabled")
            self.tts.speak("TinyLlama chat is ready")
        
        print("-" * 50)
    
    def _load_cache(self):
        """Load response cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """Save response cache to disk"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except:
            pass
    
    def format_prompt(self, user_input, mode="chat"):
        """
        Format prompt for TinyLlama with appropriate template
        """
        if self.is_finetuned:
            # Fine-tuned on Q&A format
            if mode == "code":
                return f"Question: Write Python code for: {user_input}\nAnswer:"
            elif mode == "explain":
                return f"Question: Explain this Python concept: {user_input}\nAnswer:"
            elif mode == "debug":
                return f"Question: Debug this Python code:\n{user_input}\nAnswer:"
            else:
                return f"Question: {user_input}\nAnswer:"
        else:
            # Use TinyLlama chat template
            system_prompt = "You are a helpful programming assistant specializing in Python. Provide clear, concise answers focused on practical programming solutions."
            
            if mode == "code":
                user_prompt = f"Write Python code for: {user_input}"
            elif mode == "explain":
                user_prompt = f"Explain this Python concept: {user_input}"
            elif mode == "debug":
                user_prompt = f"Debug this Python code:\n{user_input}"
            else:
                user_prompt = user_input
            
            # TinyLlama chat format
            return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    
    def generate_response(self, user_input, mode="chat", use_cache=True):
        """
        Generate response with CPU-optimized settings and TTS support
        """
        # Check cache first
        cache_key = f"{mode}:{user_input}"
        if use_cache and cache_key in self.response_cache:
            print("💾 Using cached response...")
            response = self.response_cache[cache_key]
            if self.tts and self.tts.enabled:
                self._speak_response(response, mode)
            return response
        
        # Format prompt
        prompt = self.format_prompt(user_input, mode)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(inputs)
        
        # Generation settings optimized for CPU
        generation_config = {
            "max_new_tokens": 256,
            "min_new_tokens": 30,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "use_cache": True,
            "num_beams": 1,  # Greedy for speed on CPU
        }
        
        # Generate with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = full_response[len(prompt):].strip()
        
        # For chat template, extract assistant response
        if not self.is_finetuned and "<|assistant|>" in full_response:
            parts = full_response.split("<|assistant|>")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        # Cache the response
        if use_cache and len(self.response_cache) < 1000:  # Limit cache size
            self.response_cache[cache_key] = response
            self._save_cache()
        
        if generation_time > 2.0:
            print(f"⏱️  Generated in {generation_time:.1f}s")
        
        # Speak response if TTS enabled
        if self.tts and self.tts.enabled:
            self._speak_response(response, mode)
        
        return response
    
    def _clean_response(self, response):
        """Clean up generated response"""
        if not response:
            return "I need more information to provide a helpful answer."
        
        # Remove any remaining template tokens
        for token in ["<|system|>", "<|user|>", "<|assistant|>", "</s>", "<s>"]:
            response = response.replace(token, "")
        
        # Ensure proper ending
        if response and not response[-1] in ".!?:;'\"":
            # Find last complete sentence
            sentences = response.split(". ")
            if len(sentences) > 1:
                response = ". ".join(sentences[:-1]) + "."
            else:
                response += "."
        
        return response.strip()
    
    def _speak_response(self, response, mode):
        """Speak the response using TTS"""
        if not self.tts or not self.tts.enabled:
            return
        
        # Check if response contains code
        if "```" in response or mode == "code":
            # For code responses, speak a summary
            lines = response.split('\n')
            code_lines = len([l for l in lines if l.strip().startswith(('def ', 'class ', 'import ', 'from '))])
            if code_lines > 0:
                self.tts.speak(f"Generated code with approximately {len(lines)} lines")
            else:
                # Speak first part of response
                first_sentence = response.split('.')[0]
                if len(first_sentence) > 100:
                    first_sentence = first_sentence[:100] + "..."
                self.tts.speak(first_sentence)
        else:
            # For text responses, speak the content
            if len(response) > 200:
                # Summarize long responses (no status suffix)
                sentences = response.split('. ')
                summary = '. '.join(sentences[:2])
                self.tts.speak(summary)
            else:
                self.tts.speak(response)
    
    def stream_response(self, user_input, mode="chat"):
        """
        Stream response token by token for better UX
        """
        prompt = self.format_prompt(user_input, mode)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Create streamer
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_config = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        with torch.no_grad():
            self.model.generate(inputs, **generation_config)
    
    def interactive_chat(self):
        """Main interactive chat loop with TTS support"""
        print("\n💬 TinyLlama Programming Assistant")
        print("=" * 50)
        print("Commands:")
        print("  /mode [chat|code|explain|debug] - Switch modes")
        print("  /nocache - Disable response caching")
        print("  /clear - Clear response cache")
        if TTS_AVAILABLE:
            print("  /tts or /speak - Toggle voice output")
            print("  /mute - Temporarily disable voice")
            print("  /voices - List available voices")
            print("  /voice <name> - Change voice")
            print("  /rate <wpm> - Set speech rate (50-300)")
            print("  /volume <0-1> - Set volume level")
            print("  /tts-help - Voice commands help")
        print("  /help - Show help")
        print("  quit - Exit")
        
        if self.tts and self.tts.enabled:
            print("\n🔊 Voice output is ENABLED")
        else:
            print("\n🔇 Voice output is DISABLED")
        
        print("\nCurrent mode: chat")
        print("-" * 50 + "\n")
        
        mode = "chat"
        use_cache = True
        
        while True:
            try:
                # Mode indicator in prompt
                mode_emoji = {"chat": "💬", "code": "💻", "explain": "📖", "debug": "🐛"}
                prompt = f"[{mode_emoji.get(mode, '?')} {mode.upper()}] Query: "
                
                user_input = input(prompt).strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    if self.tts and self.tts.enabled:
                        self.tts.speak("Goodbye!")
                    print("\n👋 Goodbye!")
                    break
                
                # Handle TTS commands if available
                if self.tts_commands and self.tts_commands.handle_command(user_input):
                    continue
                
                if user_input.lower() == '/help':
                    self._show_help()
                    continue
                
                if user_input.lower() == '/clear':
                    self.response_cache = {}
                    self._save_cache()
                    print("🗑️  Response cache cleared")
                    continue
                
                if user_input.lower() == '/nocache':
                    use_cache = not use_cache
                    print(f"💾 Cache {'enabled' if use_cache else 'disabled'}")
                    continue
                
                if user_input.lower().startswith('/mode'):
                    valid_modes = ['chat', 'code', 'explain', 'debug']
                    # Accept flexible syntax: '/mode chat', '/mode: chat', '/mode=chat', '/mode to chat'
                    tokens = re.split(r'[\s=:,]+', user_input.strip())
                    target = None
                    for tok in tokens[1:]:
                        if tok.lower() in valid_modes:
                            target = tok.lower()
                            break
                    if target:
                        mode = target
                        print(f"📝 Switched to {mode} mode\n")
                        if self.tts and self.tts.enabled:
                            self.tts.speak(f"Switched to {mode} mode")
                    else:
                        print("❌ Valid modes: chat, code, explain, debug\n")
                    continue
                
                if not user_input:
                    continue
                
                # Generate and display response
                print("\n" + "=" * 50)
                response = self.generate_response(user_input, mode, use_cache)
                print(response)
                print("=" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _show_help(self):
        """Show detailed help"""
        help_text = """
📖 TinyLlama Programming Assistant Help

🎯 Modes:
- chat    : General programming discussion
- code    : Generate Python code
- explain : Explain Python concepts
- debug   : Debug Python code

💡 Example queries:
[CHAT mode]
- "What's the difference between list and tuple?"
- "When should I use async/await?"

[CODE mode]
- "function to merge two sorted lists"
- "class for a binary search tree"

[EXPLAIN mode]
- "decorators"
- "list comprehensions"

[DEBUG mode]
- "def factorial(n): return factorial(n-1) * n"

🚀 Tips:
- Responses are cached for speed
- Use /nocache to always generate fresh responses
- The model is optimized for Python programming
- Keep queries concise for faster responses
        """
        print(help_text)


def main():
    """Main entry point with TTS support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyLlama Programming Assistant with TTS")
    parser.add_argument('query', nargs='?', help='Direct query (optional)')
    parser.add_argument('-m', '--mode', choices=['chat', 'code', 'explain', 'debug'],
                        default='chat', help='Query mode')
    parser.add_argument('--model-path', help='Path to model (uses default if not specified)')
    parser.add_argument('--no-quantization', action='store_true',
                        help='Disable INT8 quantization')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable response caching')
    
    # TTS arguments
    parser.add_argument('--tts', action='store_true',
                        help='Enable text-to-speech output')
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable text-to-speech output (default)')
    parser.add_argument('--tts-rate', type=int, default=175,
                        help='TTS speech rate in words per minute (50-300)')
    parser.add_argument('--tts-volume', type=float, default=0.9,
                        help='TTS volume level (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    print("\n🚀 TinyLlama Programming Assistant")
    print("=" * 50)
    print("Efficient, CPU-optimized AI for programming")
    print("No propaganda, just practical programming help!\n")
    
    # Determine TTS setting
    enable_tts = args.tts and not args.no_tts
    
    # Initialize chat with TTS support
    chat = TinyLlamaChat(
        model_path=args.model_path,
        use_quantization=not args.no_quantization,
        enable_tts=enable_tts,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume
    )
    
    # Single query mode
    if args.query:
        response = chat.generate_response(
            args.query,
            mode=args.mode,
            use_cache=not args.no_cache
        )
        print("\n" + "=" * 50)
        print(response)
        print("=" * 50)
    else:
        # Interactive mode
        chat.interactive_chat()


if __name__ == "__main__":
    main()

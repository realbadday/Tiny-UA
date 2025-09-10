#!/usr/bin/env python3
"""
TinyLlama - Unified Programming Assistant
A single entry point for all TinyLlama functionality with consistent CLI, modes, and TTS support.
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
from typing import Optional, Dict, List, Any

# Import shared modules
from cli_args import build_parser, apply_tts_settings, determine_tts_enabled, get_standard_epilog
from mode_manager import ModeManager
from command_router import CommandRouter
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


class TinyLlama:
    """Unified TinyLlama Assistant with all features"""
    
    # Define all supported modes
    ALL_MODES = [
        "chat", "code", "explain", "debug", "function", 
        "class", "script", "test", "improve", "convert"
    ]
    
    def __init__(self, args):
        """
        Initialize TinyLlama with command-line arguments
        Args:
            args: Parsed command-line arguments
        """
        # Initialize TTS
        enable_tts = determine_tts_enabled(args)
        self.tts = TTSManager(auto_init=enable_tts)
        self.tts_commands = TTSCommands(self.tts)
        
        # Apply TTS settings from CLI
        if enable_tts:
            apply_tts_settings(args, self.tts)
        
        # Initialize mode manager
        self.mode_manager = ModeManager(
            modes=self.ALL_MODES,
            default_mode=args.mode
        )
        
        # Initialize command router with all handlers
        self.command_router = CommandRouter()
        self._setup_command_handlers()
        
        # Announce startup
        if self.tts.enabled:
            self.tts.speak("Starting TinyLlama assistant", interrupt=True, is_status=True)
        
        # Model configuration
        self.model_path = self._resolve_model_path(args.model_path)
        self.use_quantization = not args.no_quantization
        self.use_cache = not args.no_cache
        
        print("⚙️  Initializing TinyLlama...")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize templates and cache
        self._init_templates()
        self._init_cache()
        
        print("✅ TinyLlama ready!")
        
        # Announce ready
        if self.tts.enabled:
            self.tts.speak("TinyLlama is ready", is_status=True)
        
        print("-" * 50)
    
    def _setup_command_handlers(self):
        """Set up all command handlers"""
        # Mode commands
        @self.command_router.register("/mode")
        def handle_mode_change(args: str):
            """Change the current mode"""
            result = self.mode_manager.change_mode(args)
            if result:
                print(result)
                if self.tts.enabled:
                    self.tts.speak(f"Switched to {self.mode_manager.current_mode} mode")
            
        @self.command_router.register("/modes")
        def handle_list_modes(args: str):
            """List available modes"""
            self.mode_manager.list_modes()
        
        # TTS commands - register all from TTSCommands
        for cmd, handler in self.tts_commands.commands.items():
            self.command_router.register(cmd)(handler)
        
        # System commands
        @self.command_router.register("/help")
        def handle_help(args: str):
            """Show help"""
            self.show_help()
            
        @self.command_router.register("/clear")
        def handle_clear(args: str):
            """Clear screen"""
            os.system('clear' if os.name == 'posix' else 'cls')
            
        @self.command_router.register("/exit")
        def handle_exit(args: str):
            """Exit the assistant"""
            print("\n👋 Goodbye!")
            if self.tts.enabled:
                self.tts.speak("Goodbye!")
                self.tts.wait_for_speech(timeout=2.0)
            self.tts.shutdown()
            sys.exit(0)
            
        @self.command_router.register("/quit")
        def handle_quit(args: str):
            """Exit the assistant"""
            handle_exit(args)
    
    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve the model path"""
        if model_path:
            return model_path
            
        # Check for fine-tuned model
        fine_tuned_path = Path.home() / "tinyllama" / "models" / "tinyllama-unified"
        if fine_tuned_path.exists():
            print(f"🎯 Found fine-tuned model at {fine_tuned_path}")
            self.is_finetuned = True
            return str(fine_tuned_path)
        
        # Default to base model
        print("🤖 Using TinyLlama-1.1B-Chat-v1.0")
        self.is_finetuned = False
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    def _load_model(self):
        """Load the model and tokenizer"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if requested
        if self.use_quantization:
            try:
                print("📊 Applying INT8 quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
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
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
        
        self.model.eval()
    
    def _init_templates(self):
        """Initialize code generation templates"""
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
    
    def _init_cache(self):
        """Initialize response cache"""
        self.cache_file = Path.home() / ".tinyllama" / "response_cache.json"
        self.response_cache = {}
        if self.use_cache:
            self.response_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
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
        if not self.use_cache:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except:
            pass
    
    def format_prompt(self, query: str) -> str:
        """Format prompt based on current mode"""
        mode = self.mode_manager.current_mode
        
        if mode in ["function", "class", "script"]:
            return self.code_templates[mode].format(description=query)
        elif mode in ["fix", "improve", "explain", "test", "convert"]:
            # These need code input
            return query
        else:
            # Chat format
            system_prompt = "You are a helpful programming assistant. Provide clear, concise answers."
            if mode == "code":
                system_prompt += " Focus on generating clean, working code."
            elif mode == "explain":
                system_prompt += " Explain concepts clearly with examples."
            elif mode == "debug":
                system_prompt += " Help debug code and fix issues."
            
            return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
    
    def generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate response from model"""
        # Check cache first
        cache_key = f"{self.mode_manager.current_mode}:{prompt}"
        if self.use_cache and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            # Use cache if less than 1 hour old
            if time.time() - cached['timestamp'] < 3600:
                print("📎 Using cached response...")
                return cached['response']
        
        # Announce generation
        if self.tts.enabled:
            self.tts.speak("Generating response", interrupt=True, is_status=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Create streamer for live output
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Set up stopping criteria
        stopping_criteria = StoppingCriteriaList([])
        if self.mode_manager.current_mode in ["code", "function", "class", "script", "fix", "improve", "test", "convert"]:
            stopping_criteria.append(CodeStoppingCriteria(self.tokenizer))
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                stopping_criteria=stopping_criteria
            )
        
        # Get the generated text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Cache the response
        if self.use_cache:
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            self._save_cache()
        
        return response
    
    def extract_code(self, response: str) -> Optional[str]:
        """Extract code from response"""
        code_match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return None
    
    def show_help(self):
        """Show interactive help"""
        help_text = f"""
🤖 TinyLlama Commands:
  
📝 General:
  /help          Show this help
  /clear         Clear screen
  /exit, /quit   Exit assistant

🎭 Modes:
  /mode MODE     Switch mode (current: {self.mode_manager.current_mode})
  /modes         List all modes

🔊 Text-to-Speech:
  /tts           Toggle voice output
  /voices        List available voices
  /voice NAME    Set voice
  /rate WPM      Set speech rate
  /volume 0-1    Set volume level

💡 Current Mode: {self.mode_manager.get_mode_info()}
"""
        print(help_text)
    
    def process_query(self, query: str) -> bool:
        """
        Process a user query
        Returns: True if should continue, False to exit
        """
        # Check for commands first
        if self.command_router.is_command(query):
            try:
                return self.command_router.route(query)
            except SystemExit:
                return False
            except Exception as e:
                print(f"❌ Command error: {e}")
            return True
        
        # Regular query processing
        try:
            # Format prompt
            prompt = self.format_prompt(query)
            
            # Generate response
            print("\n" + "="*50)
            response = self.generate_response(prompt)
            print("="*50 + "\n")
            
            # Handle TTS for response
            if self.tts.enabled:
                if self.mode_manager.current_mode in ["code", "function", "class", "script", "fix", "improve", "test", "convert"]:
                    # For code modes, speak a summary
                    code = self.extract_code(response)
                    if code:
                        self.tts.speak_code(code, summary=True)
                    else:
                        # Just speak the response if no code block found
                        self.tts.speak(response[:200])  # First 200 chars
                else:
                    # For chat/explain modes, speak the full response
                    self.tts.speak(response)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n💡 Tip: Use /exit to quit properly")
            return True
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return True
    
    def run_interactive(self):
        """Run interactive mode"""
        print(f"\n💬 Interactive mode - {self.mode_manager.get_mode_info()}")
        print("Type /help for commands, /exit to quit\n")
        
        while True:
            try:
                # Get input with mode indicator
                mode_emoji = {
                    "chat": "💬", "code": "💻", "explain": "📖",
                    "debug": "🐛", "function": "🔧", "class": "📦",
                    "script": "📄", "test": "🧪", "improve": "⚡",
                    "convert": "🔄"
                }.get(self.mode_manager.current_mode, "🤖")
                
                query = input(f"{mode_emoji} > ").strip()
                
                if not query:
                    continue
                
                if not self.process_query(query):
                    break
                    
            except KeyboardInterrupt:
                print("\n💡 Use /exit to quit")
            except EOFError:
                break
    
    def run_single_query(self, query: str):
        """Run a single query and exit"""
        self.process_query(query)
        if self.tts.enabled:
            self.tts.wait_for_speech(timeout=30.0)
        self.tts.shutdown()


def main():
    """Main entry point"""
    # Build parser with all modes
    parser = build_parser(
        app_name="tinyllama",
        description="TinyLlama - Unified Programming Assistant with Voice Support",
        allowed_modes=TinyLlama.ALL_MODES,
        default_mode="chat",
        supports_query=True,
        epilog=get_standard_epilog()
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize TinyLlama
    assistant = TinyLlama(args)
    
    # Run based on whether query was provided
    if args.query:
        query = ' '.join(args.query)
        assistant.run_single_query(query)
    else:
        assistant.run_interactive()


if __name__ == "__main__":
    main()

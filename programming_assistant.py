#!/usr/bin/env python3
"""
TinyLlama Programming Assistant
Unified interface with TTS support and enhanced features
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "chat"))

from tinyllama_chat import TinyLlamaChat
import warnings
import time

warnings.filterwarnings('ignore')

# Try to import TTS components
TTS_AVAILABLE = False
try:
    import pyttsx4
    TTS_AVAILABLE = True
except ImportError:
    pass


class ProgrammingAssistant(TinyLlamaChat):
    """Enhanced TinyLlama with TTS and programming-specific features"""
    
    def __init__(self, model_path=None, use_quantization=True, enable_tts=False):
        """Initialize with TTS support"""
        super().__init__(model_path, use_quantization)
        
        # TTS setup
        self.tts_enabled = False
        self.tts = None
        
        if enable_tts and TTS_AVAILABLE:
            try:
                self.tts = pyttsx4.init('espeak')
                self.tts.setProperty('rate', 175)
                self.tts.setProperty('volume', 0.9)
                self.tts_enabled = True
                print("🔊 TTS enabled")
            except Exception as e:
                print(f"⚠️  TTS initialization failed: {e}")
        
        # Programming-specific templates
        self.templates = {
            "plan": "Create a step-by-step plan to: {query}",
            "review": "Review this Python code and suggest improvements:\n{code}",
            "convert": "Convert this code to Python:\n{code}",
            "test": "Write unit tests for this function:\n{code}",
            "optimize": "Optimize this Python code for performance:\n{code}",
            "document": "Add docstrings and comments to this code:\n{code}"
        }
        
        # Code execution safety
        self.safe_mode = True
        
    def speak(self, text):
        """Speak text using TTS if enabled"""
        if self.tts_enabled and self.tts:
            try:
                # Clean up code for speech
                clean_text = ' '.join(text.split())
                if len(clean_text) > 500:
                    clean_text = clean_text[:500] + "... (truncated for speech)"
                
                self.tts.say(clean_text)
                self.tts.runAndWait()
            except Exception as e:
                print(f"⚠️  TTS Error: {e}")
    
    def toggle_tts(self):
        """Toggle TTS on/off (initializes engine on-demand)"""
        if not TTS_AVAILABLE:
            print("❌ TTS not available. Install with: pip install pyttsx4")
            return
        
        if not self.tts_enabled:
            # Enabling: initialize engine if needed
            if self.tts is None:
                try:
                    self.tts = pyttsx4.init('espeak')
                    self.tts.setProperty('rate', 175)
                    self.tts.setProperty('volume', 0.9)
                except Exception as e:
                    print(f"⚠️  TTS initialization failed: {e}")
                    return
            self.tts_enabled = True
            print("🔊 TTS enabled")
            self.speak("Text to speech enabled")
        else:
            # Disabling
            self.tts_enabled = False
            print("🔇 TTS disabled")
    
    def plan_task(self, query):
        """Create a step-by-step plan for a programming task"""
        prompt = self.templates["plan"].format(query=query)
        response = self.generate_response(prompt, mode="chat")
        
        # Format as numbered list
        lines = response.split('\n')
        formatted = []
        step_num = 1
        
        for line in lines:
            line = line.strip()
            if line and not line[0].isdigit():
                formatted.append(f"{step_num}. {line}")
                step_num += 1
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)
    
    def review_code(self, code):
        """Review code and suggest improvements"""
        prompt = self.templates["review"].format(code=code)
        return self.generate_response(prompt, mode="explain")
    
    def generate_tests(self, code):
        """Generate unit tests for code"""
        prompt = self.templates["test"].format(code=code)
        return self.generate_response(prompt, mode="code")
    
    def optimize_code(self, code):
        """Suggest optimizations for code"""
        prompt = self.templates["optimize"].format(code=code)
        return self.generate_response(prompt, mode="code")
    
    def document_code(self, code):
        """Add documentation to code"""
        prompt = self.templates["document"].format(code=code)
        return self.generate_response(prompt, mode="code")
    
    def interactive_assistant(self):
        """Enhanced interactive loop with all features"""
        print("\n🚀 TinyLlama Programming Assistant")
        print("=" * 60)
        print("Enhanced features for programming tasks")
        print("=" * 60)
        print("\n📋 Commands:")
        print("  /mode [chat|code|explain|debug|plan|review|test|optimize|doc]")
        print("  /tts              - Toggle text-to-speech")
        print("  /safe             - Toggle safe mode (prevents code execution)")
        print("  /multiline        - Enter multiline mode for code input")
        print("  /example [topic]  - Show example code for a topic")
        print("  /nocache          - Disable response caching")
        print("  /clear            - Clear response cache")
        print("  /help             - Show detailed help")
        print("  quit              - Exit")
        print("\nCurrent mode: chat")
        print("-" * 60 + "\n")
        
        mode = "chat"
        use_cache = True
        multiline_mode = False
        multiline_buffer = []
        
        while True:
            try:
                # Mode indicator
                mode_icons = {
                    "chat": "💬", "code": "💻", "explain": "📖", 
                    "debug": "🐛", "plan": "📋", "review": "🔍",
                    "test": "🧪", "optimize": "⚡", "doc": "📝"
                }
                
                if multiline_mode:
                    prompt = "... "
                else:
                    prompt = f"[{mode_icons.get(mode, '?')} {mode.upper()}] Query: "
                
                user_input = input(prompt)
                
                # Handle multiline mode
                if multiline_mode:
                    # Don't strip in multiline mode to preserve formatting
                    if user_input.strip() == "" or user_input.strip() == "```":
                        # Process if we have any content in buffer
                        if multiline_buffer:
                            multiline_mode = False
                            user_input = '\n'.join(multiline_buffer)
                            multiline_buffer = []
                            print("📤 Processing multiline input...\n")
                        else:
                            # Empty buffer - exit multiline mode without processing
                            multiline_mode = False
                            continue
                    else:
                        multiline_buffer.append(user_input)
                        continue
                
                # Strip input for command processing
                user_input = user_input.strip()
                
                # Commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    if self.tts_enabled:
                        self.speak("Goodbye!")
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == '/help':
                    self._show_enhanced_help()
                    continue
                
                if user_input.lower() == '/tts':
                    self.toggle_tts()
                    continue
                
                if user_input.lower() == '/safe':
                    self.safe_mode = not self.safe_mode
                    print(f"🔒 Safe mode: {'ON' if self.safe_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == '/multiline':
                    multiline_mode = True
                    print("📝 Multiline mode - Press Enter on empty line or type '```' to finish")
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
                
                if user_input.lower().startswith('/example'):
                    parts = user_input.split(maxsplit=1)
                    topic = parts[1] if len(parts) > 1 else "list"
                    self._show_example(topic)
                    continue
                
                if user_input.lower().startswith('/mode'):
                    parts = user_input.split()
                    valid_modes = ['chat', 'code', 'explain', 'debug', 'plan', 
                                   'review', 'test', 'optimize', 'doc']
                    if len(parts) > 1 and parts[1] in valid_modes:
                        mode = parts[1]
                        print(f"📝 Switched to {mode} mode")
                        
                        # Auto-enter multiline mode for code-based modes
                        if mode in ['review', 'test', 'optimize', 'doc']:
                            print(f"📝 Paste your code, then press Enter on empty line to process")
                            multiline_mode = True
                            multiline_buffer = []
                    else:
                        print(f"❌ Valid modes: {', '.join(valid_modes)}\n")
                    continue
                
                if not user_input:
                    continue
                
                # Generate response based on mode
                print("\n" + "=" * 60)
                
                if mode == "plan":
                    response = self.plan_task(user_input)
                elif mode == "review":
                    response = self.review_code(user_input)
                elif mode == "test":
                    response = self.generate_tests(user_input)
                elif mode == "optimize":
                    response = self.optimize_code(user_input)
                elif mode == "doc":
                    response = self.document_code(user_input)
                else:
                    response = self.generate_response(user_input, mode, use_cache)
                
                print(response)
                print("=" * 60 + "\n")
                
                # Auto-switch back to chat mode after processing code in special modes
                if mode in ['review', 'test', 'optimize', 'doc']:
                    mode = 'chat'
                    print("ℹ️  Switched back to chat mode. Use /mode <name> to switch again.\n")
                
                # Speak response if TTS is enabled
                if self.tts_enabled:
                    self.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _show_enhanced_help(self):
        """Show enhanced help with all features"""
        help_text = """
📚 TinyLlama Programming Assistant - Complete Guide

🎯 Standard Modes:
- chat     : General programming discussion
- code     : Generate Python code
- explain  : Explain Python concepts
- debug    : Debug Python code

🔧 Enhanced Modes:
- plan     : Create step-by-step plans for tasks
- review   : Review code and suggest improvements
- test     : Generate unit tests for functions
- optimize : Suggest performance optimizations
- doc      : Add documentation to code

🛠️ Special Features:
- /tts       : Enable voice output
- /multiline : Input multiple lines of code
- /safe      : Toggle safe mode (prevents risky operations)
- /example   : Show example code (e.g., /example decorator)

💡 Usage Examples:

[PLAN mode] - Break down complex tasks
Query: "Build a REST API with Flask"
→ Generates step-by-step implementation plan

[REVIEW mode] - Get code feedback
Query: [paste your function]
→ Analyzes code quality, suggests improvements

[TEST mode] - Generate unit tests
Query: [paste your function]
→ Creates pytest-compatible test cases

[OPTIMIZE mode] - Improve performance
Query: [paste slow code]
→ Suggests faster alternatives

[DOC mode] - Add documentation
Query: [paste undocumented code]
→ Adds docstrings and comments

🎤 Voice Features:
- Enable TTS with /tts command
- All responses will be spoken
- Code is intelligently truncated for speech

📝 Multiline Input:
1. Type /multiline
2. Enter code line by line
3. Press Enter on empty line or type ``` to process

🚀 Pro Tips:
- Use specific modes for better results
- Cache speeds up repeated queries
- Keep code snippets focused
- Use multiline for complex code input
        """
        print(help_text)
    
    def _show_example(self, topic):
        """Show example code for common topics"""
        examples = {
            "decorator": """
# Python Decorator Example
def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    time.sleep(1)
    return "Done!"

slow_function()  # Prints: slow_function took 1.00 seconds
""",
            "async": """
# Async/Await Example
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(2)  # Simulate network delay
    return f"Data from {url}"

async def main():
    # Run multiple async tasks concurrently
    urls = ["api.com/1", "api.com/2", "api.com/3"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

asyncio.run(main())
""",
            "context": """
# Context Manager Example
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        self.file.close()

# Usage
with FileManager('test.txt', 'w') as f:
    f.write("Hello, World!")
""",
            "list": """
Available examples:
- decorator  : Python decorators
- async      : Async/await pattern
- context    : Context managers
- generator  : Generator functions
- class      : Class with properties
- exception  : Custom exceptions
- iterator   : Custom iterator
- dataclass  : Python dataclasses

Use: /example [topic]
"""
        }
        
        example = examples.get(topic, examples["list"])
        print("\n" + "=" * 60)
        print(f"📌 Example: {topic.capitalize()}")
        print("=" * 60)
        print(example)
        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TinyLlama Programming Assistant - Enhanced Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  programming_assistant.py                           # Interactive mode
  programming_assistant.py "merge two sorted lists"  # Single query
  programming_assistant.py -m plan "build a web app" # Planning mode
  programming_assistant.py --tts                     # With voice output
        """
    )
    
    parser.add_argument('query', nargs='?', help='Direct query (optional)')
    parser.add_argument('-m', '--mode', 
                        choices=['chat', 'code', 'explain', 'debug', 'plan', 
                                'review', 'test', 'optimize', 'doc'],
                        default='chat', help='Query mode')
    parser.add_argument('--model-path', help='Path to model')
    parser.add_argument('--no-quantization', action='store_true',
                        help='Disable INT8 quantization')
    parser.add_argument('--tts', action='store_true',
                        help='Enable text-to-speech')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable response caching')
    
    args = parser.parse_args()
    
    print("\n🚀 TinyLlama Programming Assistant - Enhanced Edition")
    print("=" * 60)
    print("Complete programming companion with voice support")
    print("No propaganda, just practical programming help!")
    print("=" * 60 + "\n")
    
    # Initialize assistant
    assistant = ProgrammingAssistant(
        model_path=args.model_path,
        use_quantization=not args.no_quantization,
        enable_tts=args.tts
    )
    
    # Single query mode
    if args.query:
        print(f"Mode: {args.mode}")
        print("-" * 60)
        
        if args.mode == "plan":
            response = assistant.plan_task(args.query)
        elif args.mode == "review":
            response = assistant.review_code(args.query)
        elif args.mode == "test":
            response = assistant.generate_tests(args.query)
        elif args.mode == "optimize":
            response = assistant.optimize_code(args.query)
        elif args.mode == "doc":
            response = assistant.document_code(args.query)
        else:
            response = assistant.generate_response(
                args.query,
                mode=args.mode,
                use_cache=not args.no_cache
            )
        
        print("\n" + "=" * 60)
        print(response)
        print("=" * 60)
        
        if assistant.tts_enabled:
            assistant.speak(response)
    else:
        # Interactive mode
        assistant.interactive_assistant()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TTS Utility Module for TinyLlama
Provides text-to-speech functionality with error handling and queue management
"""

import threading
import queue
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import pyttsx4
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  TTS not available. Install with: pip install pyttsx4")


class TTSManager:
    """Manages text-to-speech functionality with queuing and error handling"""
    
    def __init__(self, engine_name: str = 'espeak', auto_init: bool = True):
        """
        Initialize TTS Manager
        Args:
            engine_name: TTS engine to use ('espeak' on Linux)
            auto_init: Automatically initialize TTS on creation
        """
        self.engine_name = engine_name
        self.engine = None
        self.enabled = False
        self.speaking = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Default settings
        self.settings = {
            'rate': 175,  # Words per minute
            'volume': 0.9,  # 0.0 to 1.0
            'voice': 'english+f2',  # Default to female voice
        }
        
        # Control flags
        self.silent_status = True  # Don't speak status messages like "Generating..."
        self.speak_summaries = True  # Speak code summaries
        
        # Load saved settings
        self.config_file = Path.home() / ".tinyllama" / "tts_config.json"
        self._load_settings()
        
        if auto_init and TTS_AVAILABLE:
            self.initialize()
    
    def initialize(self) -> bool:
        """Initialize TTS engine"""
        if not TTS_AVAILABLE:
            return False
        
        try:
            self.engine = pyttsx4.init(self.engine_name)
            self._apply_settings()
            
            # Start worker thread
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            
            self.enabled = True
            print("🔊 TTS initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            print("Try: sudo apt-get install espeak libespeak-dev")
            self.engine = None
            self.enabled = False
            return False
    
    def _apply_settings(self):
        """Apply current settings to engine"""
        if not self.engine:
            return
            
        try:
            self.engine.setProperty('rate', self.settings['rate'])
            self.engine.setProperty('volume', self.settings['volume'])
            
            if self.settings['voice']:
                # For espeak, voice format can be like "en+f2"
                # Try setting directly first
                try:
                    self.engine.setProperty('voice', self.settings['voice'])
                except:
                    # If that fails, try to find matching voice in list
                    voices = self.engine.getProperty('voices')
                    for voice in voices:
                        if self.settings['voice'] in voice.id:
                            self.engine.setProperty('voice', voice.id)
                            break
        except Exception as e:
            print(f"⚠️  Error applying TTS settings: {e}")
    
    def _speech_worker(self):
        """Worker thread for processing speech queue"""
        while not self.stop_event.is_set():
            try:
                # Get text from queue with timeout
                text = self.speech_queue.get(timeout=0.1)
                
                if text is None:  # Poison pill to stop thread
                    # Mark the poison pill as processed to avoid join() hanging
                    try:
                        self.speech_queue.task_done()
                    except Exception:
                        pass
                    break
                
                # Speak the text
                self.speaking = True
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                finally:
                    self.speaking = False
                    # Mark this item as done so join() can return
                    try:
                        self.speech_queue.task_done()
                    except Exception:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  TTS Error: {e}")
                self.speaking = False
    
    def speak(self, text: str, interrupt: bool = False, is_status: bool = False):
        """
        Speak text using TTS
        Args:
            text: Text to speak
            interrupt: Stop current speech and speak immediately
            is_status: Whether this is a status message (e.g. "Generating...")
        """
        if not self.enabled or not self.engine:
            return
        
        # Skip status messages if silent_status is True
        if is_status and self.silent_status:
            return
        
        # Clean text for speech
        text = self._clean_for_speech(text)
        
        if not text:
            return
        
        if interrupt:
            self.clear_queue()
        
        self.speech_queue.put(text)
    
    def speak_code(self, code: str, summary: bool = True):
        """
        Speak code with appropriate formatting
        Args:
            code: Code to speak
            summary: If True, only speak a summary
        """
        if not self.enabled:
            return
        
        if summary:
            # Count lines and extract function/class names
            lines = code.strip().split('\n')
            num_lines = len(lines)
            
            # Try to extract main elements
            elements = []
            for line in lines:
                if line.strip().startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '')
                    elements.append(f"function {func_name}")
                elif line.strip().startswith('class '):
                    class_name = line.split(':')[0].replace('class ', '').split('(')[0]
                    elements.append(f"class {class_name}")
            
            if elements:
                summary_text = f"Generated {num_lines} lines of code with " + ", ".join(elements[:3])
                if len(elements) > 3:
                    summary_text += f" and {len(elements) - 3} more elements"
            else:
                summary_text = f"Generated {num_lines} lines of code"
            
            self.speak(summary_text)
        else:
            # Speak full code with pauses
            self.speak("Here's the generated code:")
            time.sleep(0.5)
            self.speak(code)
    
    def _clean_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length for speech
        if len(text) > 500:
            text = text[:500] + "... (truncated for speech)"
        
        # Replace code-specific symbols for better speech
        replacements = {
            "```": "code block",
            ">>>": "prompt",
            "...": "continued",
            "__": "double underscore",
            "->": "returns",
            "=>": "arrow",
            "!=": "not equal",
            "==": "equals",
            "<=": "less than or equal",
            ">=": "greater than or equal",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def toggle(self) -> bool:
        """Toggle TTS on/off"""
        if not TTS_AVAILABLE:
            print("❌ TTS not available. Install with: pip install pyttsx4")
            return False
        
        if self.enabled:
            self.enabled = False
            self.clear_queue()
            print("🔇 TTS disabled")
        else:
            if not self.engine:
                self.initialize()
            else:
                self.enabled = True
                print("🔊 TTS enabled")
        
        return self.enabled
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.settings['rate'] = max(50, min(300, rate))
        self._apply_settings()
        self._save_settings()
        self.speak(f"Speech rate set to {rate}")
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        self.settings['volume'] = max(0.0, min(1.0, volume))
        self._apply_settings()
        self._save_settings()
    
    def list_voices(self) -> List[str]:
        """List available voices"""
        if not self.engine:
            return []
        
        voices = []
        try:
            for voice in self.engine.getProperty('voices'):
                voices.append(f"{voice.name} ({voice.id})")
        except Exception as e:
            print(f"⚠️  Error listing voices: {e}")
        
        return voices
    
    def set_voice(self, voice_id: str):
        """Set voice by ID or partial name"""
        if not self.engine:
            return
        
        # Voice aliases for convenience
        voice_aliases = {
            'default': 'english+f2',
            'female': 'english+f2',
            'woman': 'english+f2',
            'male': 'english+m1',
            'man': 'english+m1',
            'female2': 'english+f3',
            'female3': 'english+f4',
            'male2': 'english+m2',
            'male3': 'english+m3'
        }
        
        # Apply aliases
        voice_id_lower = voice_id.lower() if voice_id else 'default'
        if voice_id_lower in voice_aliases:
            voice_id = voice_aliases[voice_id_lower]
            print(f"🔊 Using voice alias: {voice_id}")
        
        try:
            # For espeak, try setting the voice directly first (e.g., "en+f2")
            try:
                self.engine.setProperty('voice', voice_id)
                self.settings['voice'] = voice_id
                self._save_settings()
                self.speak(f"Voice changed to {voice_id}")
                return
            except:
                # If direct setting fails, search in available voices
                pass
            
            # Search in available voices
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if voice_id.lower() in voice.id.lower() or voice_id.lower() in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    self.settings['voice'] = voice.id
                    self._save_settings()
                    self.speak(f"Voice changed to {voice.name}")
                    return
            
            print(f"❌ Voice '{voice_id}' not found")
            print("💡 Valid voices:")
            print("  Aliases: default, female, male, woman, man")
            print("  Direct: english+f2, english+f3, english+m1, etc.")
        except Exception as e:
            print(f"⚠️  Error setting voice: {e}")
    
    def clear_queue(self):
        """Clear speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                # Mark the removed item as done to keep the counter in sync
                try:
                    self.speech_queue.task_done()
                except Exception:
                    pass
            except queue.Empty:
                break
    
    def wait_for_speech(self, timeout: Optional[float] = None):
        """Wait for all queued speech to complete"""
        if timeout:
            # Use a simple timeout mechanism
            import time
            start = time.time()
            while not self.speech_queue.empty() or self.speaking:
                if time.time() - start > timeout:
                    break
                time.sleep(0.1)
        else:
            self.speech_queue.join()
    
    def shutdown(self):
        """Shutdown TTS manager"""
        self.stop_event.set()
        if self.worker_thread:
            self.speech_queue.put(None)  # Poison pill
            self.worker_thread.join(timeout=1.0)
    
    def _save_settings(self):
        """Save TTS settings to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving TTS settings: {e}")
    
    def _load_settings(self):
        """Load TTS settings from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                    self.settings.update(saved)
            except Exception:
                pass


class TTSCommands:
    """Command handler for TTS-related commands"""
    
    def __init__(self, tts_manager: TTSManager):
        self.tts = tts_manager
        
        self.commands = {
            '/tts': self.toggle_tts,
            '/speak': self.toggle_tts,
            '/mute': self.mute,
            '/rate': self.set_rate,
            '/volume': self.set_volume,
            '/voice': self.set_voice,
            '/voices': self.list_voices,
            '/tts-help': self.show_help,
        }
    
    def handle_command(self, command: str) -> bool:
        """
        Handle TTS command
        Returns: True if command was handled, False otherwise
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in self.commands:
            self.commands[cmd](args)
            return True
        
        return False
    
    def toggle_tts(self, args: str = ""):
        """Toggle TTS on/off"""
        self.tts.toggle()
    
    def mute(self, args: str = ""):
        """Mute TTS"""
        if self.tts.enabled:
            self.tts.enabled = False
            print("🔇 TTS muted")
    
    def set_rate(self, args: str):
        """Set speech rate"""
        try:
            rate = int(args)
            self.tts.set_rate(rate)
            print(f"🎚️  Speech rate set to {rate} words per minute")
        except ValueError:
            print("❌ Usage: /rate <number>  (e.g., /rate 150)")
    
    def set_volume(self, args: str):
        """Set speech volume"""
        try:
            volume = float(args)
            self.tts.set_volume(volume)
            print(f"🔊 Volume set to {int(volume * 100)}%")
        except ValueError:
            print("❌ Usage: /volume <0.0-1.0>  (e.g., /volume 0.8)")
    
    def set_voice(self, args: str):
        """Set voice"""
        if args:
            self.tts.set_voice(args)
        else:
            print("❌ Usage: /voice <name>  (e.g., /voice english)")
    
    def list_voices(self, args: str = ""):
        """List available voices"""
        voices = self.tts.list_voices()
        if voices:
            print("\n📢 Available voices:")
            for i, voice in enumerate(voices):
                print(f"  {i+1}. {voice}")
        else:
            print("❌ No voices available")
    
    def show_help(self, args: str = ""):
        """Show TTS help"""
        help_text = """
🔊 TTS Commands:
  /tts or /speak  - Toggle text-to-speech
  /mute           - Disable TTS temporarily  
  /rate <wpm>     - Set speech rate (50-300)
  /volume <0-1>   - Set volume (0.0 to 1.0)
  /voice <name>   - Change voice
  /voices         - List available voices
  /tts-help       - Show this help
        """
        print(help_text)


# Convenience function for quick TTS setup
def create_tts_manager() -> TTSManager:
    """Create and return a TTS manager instance"""
    return TTSManager(auto_init=True)


if __name__ == "__main__":
    # Test TTS functionality
    print("🧪 Testing TTS Manager...")
    
    tts = create_tts_manager()
    
    if tts.enabled:
        print("\n✅ TTS is working!")
        
        # Test basic speech
        tts.speak("Hello! This is TinyLlama with text-to-speech support.")
        tts.wait_for_speech()
        
        # Test code speech
        sample_code = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        
        tts.speak_code(sample_code)
        tts.wait_for_speech()
        
        # List voices
        voices = tts.list_voices()
        print(f"\nFound {len(voices)} voices")
        
        # Test commands
        cmd_handler = TTSCommands(tts)
        cmd_handler.show_help("")
        
        tts.shutdown()
    else:
        print("\n❌ TTS not available")

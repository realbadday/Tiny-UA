#!/usr/bin/env python3
"""
Simple TTS implementation without threading to avoid blocking issues
"""

import os
import json
from pathlib import Path

try:
    import pyttsx4
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class SimpleTTS:
    """Simple TTS without threading - direct execution"""
    
    def __init__(self, auto_init=True):
        self.engine = None
        self.enabled = False
        
        # Default settings
        self.settings = {
            'rate': 175,
            'volume': 0.9,
            'voice': 'english+f2'
        }
        
        # Load saved settings
        self.config_file = Path.home() / ".tinyllama" / "tts_config.json"
        self._load_settings()
        
        if auto_init and TTS_AVAILABLE:
            self.initialize()
    
    def initialize(self):
        """Initialize TTS engine"""
        if not TTS_AVAILABLE:
            return False
        
        try:
            self.engine = pyttsx4.init('espeak')
            self._apply_settings()
            self.enabled = True
            print("🔊 Simple TTS initialized successfully")
            return True
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            self.enabled = False
            return False
    
    def _apply_settings(self):
        """Apply settings to engine"""
        if not self.engine:
            return
        
        try:
            self.engine.setProperty('rate', self.settings['rate'])
            self.engine.setProperty('volume', self.settings['volume'])
            if self.settings['voice']:
                self.engine.setProperty('voice', self.settings['voice'])
        except Exception as e:
            print(f"⚠️  Error applying TTS settings: {e}")
    
    def speak(self, text, interrupt=False, is_status=False):
        """Speak text immediately (non-threaded)"""
        if not self.enabled or not self.engine or not text:
            return
        
        try:
            # Clean text for speech
            text = self._clean_for_speech(text)
            if len(text) > 300:  # Limit length to avoid very long speech
                text = text[:300] + "... continuing in chat"
            
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️  TTS Error: {e}")
    
    def _clean_for_speech(self, text):
        """Clean text for better speech synthesis"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Replace code-specific symbols
        replacements = {
            "```": "code block",
            ">>": "prompt",
            "__": "underscore",
            "!=": "not equal",
            "==": "equals",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _load_settings(self):
        """Load TTS settings from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                    self.settings.update(saved)
            except Exception:
                pass
    
    def toggle(self):
        """Toggle TTS on/off"""
        if not TTS_AVAILABLE:
            print("❌ TTS not available")
            return False
        
        if self.enabled:
            self.enabled = False
            print("🔇 TTS disabled")
        else:
            if not self.engine:
                self.initialize()
            else:
                self.enabled = True
                print("🔊 TTS enabled")
        return self.enabled
    
    def shutdown(self):
        """Shutdown TTS"""
        self.enabled = False


# Create a simple wrapper that matches the expected interface
class TTSManager:
    """Wrapper to match the existing TTSManager interface"""
    
    def __init__(self, auto_init=True, **kwargs):
        self.tts = SimpleTTS(auto_init=auto_init)
        self.enabled = self.tts.enabled
    
    def speak(self, text, interrupt=False, is_status=False):
        self.tts.speak(text, interrupt, is_status)
    
    def toggle(self):
        result = self.tts.toggle()
        self.enabled = self.tts.enabled
        return result
    
    def shutdown(self):
        self.tts.shutdown()
        self.enabled = False
    
    def wait_for_speech(self, timeout=None):
        # Since we're not using threading, this is a no-op
        pass


class TTSCommands:
    """Simple command handler"""
    
    def __init__(self, tts_manager):
        self.tts = tts_manager
    
    def handle_command(self, command):
        if command.strip() == '/tts':
            self.tts.toggle()
            return True
        return False

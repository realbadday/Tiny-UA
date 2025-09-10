# TinyLlama TTS (Text-to-Speech) Guide

## Overview
TinyLlama now includes comprehensive text-to-speech support, allowing you to hear responses from the AI assistant. This feature works offline using the pyttsx4 library with the espeak engine on Linux.

## Prerequisites

### Install Required Dependencies
```bash
# Install espeak (TTS engine for Linux)
sudo apt-get update
sudo apt-get install espeak libespeak-dev

# Install Python TTS library (if not already installed)
pip3 install pyttsx4
```

## Quick Start

### 1. Test TTS Functionality
```bash
# Test if TTS is working
python3 test_tts.py
```

### 2. Launch with Voice Support
```bash
# Unified assistant with all features and TTS
./tts_unified.sh

# Chat mode with voice
./tts_chat.sh

# Fixed version with voice (recommended if having TensorFlow conflicts)
./tts_chat_fixed.sh

# Code generation with voice feedback
./tts_codegen.sh
```

## Available Scripts

### tinyllama_unified_tts.py
The main unified assistant with complete TTS integration:
- Combines chat and code generation
- Voice announcements for all actions
- TTS commands integrated into the interface
- Automatic voice summaries for code

### tinyllama_chat_fixed.py
The TensorFlow conflict-free version with TTS support:
- All TTS features from tinyllama_chat_tts.py
- Avoids TensorFlow import conflicts
- Use with --tts flag to enable voice
- Recommended if experiencing import errors

### tts_utils.py
The TTS utility module providing:
- Thread-safe speech queue
- Voice management
- Settings persistence
- Error handling

## Voice Commands

While in any TTS-enabled mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `/tts` or `/speak` | Toggle voice output on/off |
| `/mute` | Temporarily disable voice (keeps settings) |
| `/rate <wpm>` | Set speech rate (50-300 words per minute) |
| `/volume <0-1>` | Set volume (0.0 to 1.0) |
| `/voice <name>` | Change voice (use `/voices` to list) |
| `/voices` | List all available voices |
| `/tts-help` | Show TTS help |

## Usage Examples

### Basic Usage
```bash
# Interactive mode with voice
./tts_unified.sh

# Single query with voice response
./tts_unified.sh "explain Python decorators"

# Start in specific mode
./tts_unified.sh --mode function
```

### Adjusting Voice Settings
```bash
# Set custom speech rate and volume
./tts_unified.sh --tts-rate 150 --tts-volume 0.8

# Disable TTS from command line
./tts_unified.sh --no-tts
```

### In Interactive Mode
```
[💬 CHAT] >>> /tts              # Toggle voice on/off
[💬 CHAT] >>> /rate 200         # Speak faster
[💬 CHAT] >>> /volume 0.5       # Quieter volume
[💬 CHAT] >>> /voices           # List voices
[💬 CHAT] >>> /voice english+f3 # Change voice
```

## Features

### 1. **Intelligent Speech**
- Long responses are automatically summarized
- Code is announced with line count and main elements
- Errors are spoken clearly

### 2. **Non-Blocking Speech**
- Speech runs in background thread
- Can continue typing while assistant speaks
- Speech queue ensures order is maintained

### 3. **Code Awareness**
- Code blocks are summarized (e.g., "Generated 15 lines with function calculate_pi")
- Option to read full code or just summary
- Special handling for programming symbols

### 4. **Persistent Settings**
- TTS settings saved to `~/.tinyllama/tts_config.json`
- Remembers your preferred voice, rate, and volume
- Settings persist between sessions

## Troubleshooting

### No Sound / TTS Not Working
1. Check espeak is installed:
   ```bash
   espeak "test"
   ```

2. Install missing dependencies:
   ```bash
   sudo apt-get install espeak libespeak-dev python3-espeak
   ```

3. Test Python TTS:
   ```bash
   python3 -c "import pyttsx4; engine = pyttsx4.init('espeak'); engine.say('test'); engine.runAndWait()"
   ```

### Voice Sounds Robotic
This is normal for espeak. For better voices:
1. List available voices: `/voices`
2. Try different voices: `/voice english+f3`
3. Adjust rate: `/rate 150`

### TTS Crashes or Hangs
1. Disable and re-enable: `/tts` twice
2. Restart the assistant
3. Check system audio is working

### Performance Issues
1. Disable TTS if not needed: `--no-tts`
2. Use shorter responses
3. Clear speech queue by toggling TTS

## Advanced Usage

### Custom Voice Profiles
Create different startup scripts for different use cases:

```bash
# Fast speech for quick reviews
./tts_unified.sh --tts-rate 250 --tts-volume 0.7

# Slow and clear for learning
./tts_unified.sh --tts-rate 120 --tts-volume 1.0
```

### Integration with Screen Readers
TTS works alongside screen readers. To avoid conflicts:
1. Mute TinyLlama TTS: `/mute`
2. Or run without TTS: `--no-tts`

## Tips

1. **For Code Generation**: TTS will announce when code is generated and provide a summary
2. **For Learning**: Slow down speech rate to better understand complex explanations
3. **For Quick Use**: Speed up rate and lower volume for rapid feedback
4. **Keyboard Shortcuts**: Currently not implemented, use slash commands

## Platform Notes

### Linux (Primary Support)
- Uses espeak engine
- Best compatibility
- Multiple voice options

### Windows
- Uses SAPI5 (if available)
- Limited testing
- May need additional setup

### macOS
- Uses NSSpeechSynthesizer
- Not tested
- May work with modifications

## Future Enhancements
- [ ] Voice input support
- [ ] Multiple language support
- [ ] Custom voice training
- [ ] Hotkey support
- [ ] Speech interruption
- [ ] Voice profiles

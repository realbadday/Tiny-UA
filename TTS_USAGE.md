# 🔊 TinyLlama TTS Usage Guide

## ✅ **TTS is Working!**
Your TTS is functioning correctly with the **English+f2 voice**. The ALSA error messages are just harmless warnings from the audio subsystem.

## 🚀 **Launch Options**

### **Clean Launch (Recommended)**
```bash
./launch.sh                    # No ALSA warnings, clean output
./launch.sh "your question"    # Direct query without warnings
```

### **Verbose Launch (Debug)**
```bash
./launch-verbose.sh            # Shows all messages including ALSA
```

### **Manual Launch**
```bash
source tinyllama_env/bin/activate
python3 main.py                # Full output with ALSA warnings
python3 main.py --no-tts       # Disable TTS completely
```

## 🔧 **TTS Configuration**
Your TTS settings are saved in `~/.tinyllama/tts_config.json`:
- **Voice**: English+f2 (female voice)
- **Rate**: 175 WPM
- **Volume**: 0.9 (90%)
- **Status Messages**: Enabled
- **Code Summaries**: Enabled

## 🎯 **What Those ALSA Errors Mean**
```
ALSA lib pcm_dmix.c:1000:(snd_pcm_dmix_open) unable to open slave
aplay: main:850: audio open error: Device or resource busy
```

These are **harmless warnings** that occur because:
- Your system uses PipeWire for audio management
- ALSA libraries occasionally try to access audio devices directly  
- The TTS still works perfectly despite these messages
- The `launch.sh` script now suppresses them for a cleaner experience

## 🎤 **Voice Commands in Chat**
While in interactive mode, you can use:
- `/voice rate 200` - Change speech rate
- `/voice volume 0.8` - Change volume
- `/voice off` - Disable TTS temporarily  
- `/voice on` - Re-enable TTS

## 💡 **Tips**
- Use `./launch.sh` for everyday use (clean, no warnings)
- Use `./launch-verbose.sh` if you need to debug audio issues
- The TTS automatically speaks responses and code summaries
- Your preferred voice settings are persistent across sessions

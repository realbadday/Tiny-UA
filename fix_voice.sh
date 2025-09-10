#!/bin/bash
# Quick script to restore the correct TTS voice

echo "🔊 Restoring TTS voice to english+f2..."

# Ensure the .tinyllama directory exists
mkdir -p ~/.tinyllama

# Write the correct configuration
cat > ~/.tinyllama/tts_config.json << 'EOF'
{
  "rate": 175,
  "volume": 0.9,
  "voice": "english+f2",
  "silent_status": false,
  "speak_summaries": true
}
EOF

echo "✅ Voice configuration restored!"
echo "Your TTS should now use the female english+f2 voice."

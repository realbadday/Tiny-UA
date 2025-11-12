#!/bin/bash
# Quick test to verify the response extraction fix

cd /home/jason/projects/tinyllama
source tinyllama_env/bin/activate

echo "🧪 Testing response extraction fix..."
echo "======================================"

test_queries=(
    "What is Python?"
    "How do I use loops?"
    "Explain variables"
    "What is a dictionary?"
)

for i in "${!test_queries[@]}"; do
    query="${test_queries[$i]}"
    echo ""
    echo "Test $((i+1)): '$query'"
    echo "----------------------------------------"
    
    response=$(python3 main.py --no-tts "$query" 2>/dev/null | grep -A 20 "====" | tail -n +2 | head -n 3)
    
    first_char=$(echo "$response" | head -c 1)
    echo "First character: '$first_char'"
    echo "Response starts with: '$(echo "$response" | head -c 50)...'"
    
    if [[ "$first_char" =~ [A-Za-z] ]]; then
        echo "✅ Response starts correctly with a letter"
    else
        echo "⚠️  Response may be truncated (starts with: '$first_char')"
    fi
done

echo ""
echo "🏁 Test completed! If all responses start with letters, the fix is working!"

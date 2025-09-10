# How to Use Improve Mode Correctly

## Method 1: Using /mode improve (in tinyllama_unified_tts.py)

```
[💬 CHAT] >>> /mode improve
📝 Switched to ⚡ improve - Improve code

[⚡ IMPROVE] >>> improve this code
📝 Enter code to improve (type 'EOF' when done):
... def add(a,b):
... return a+b
... EOF

[Response will appear here]
```

## Method 2: Direct approach

Instead of using `/mode improve`, you can stay in chat mode and ask directly:

```
[💬 CHAT] >>> Please improve this code for efficiency: def add(a,b): return a+b
```

## Method 3: Using multiline mode

```
[💬 CHAT] >>> /multiline
📝 Multiline mode - Enter 'EOF' or '```' to finish
... Please improve this code:
... def add(a,b):
...     return a+b
... EOF
```

## The Issue

When you use `/mode improve`, you still need to provide a query/prompt first before entering the code. The system expects:
1. Switch to improve mode
2. Type a query like "improve this code" or just "improve"  
3. THEN it enters multiline mode for you to paste the code
4. Type EOF to process

This is a bit confusing because it requires two steps after switching modes.

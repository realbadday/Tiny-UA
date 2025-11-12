# Known Issues and Workarounds

## 1. Improve Mode Gets Stuck in Input Loop

### Issue Description
The `/mode improve` command has a critical bug where:
- After switching to improve mode and typing anything, it enters multiline mode
- Typing `EOF` does NOT exit the multiline mode as expected
- The program gets stuck in an infinite input loop
- Only Ctrl+C can break out of the loop

### Symptoms
```
[⚡ IMPROVE] >>> improve this
📝 Enter code to improve (type 'EOF' when done):
... def hello():
... print("hello")
... EOF
... [STUCK - EOF doesn't work]
... [Only Ctrl+C exits]
```

### Root Cause
The multiline handling logic for improve mode is broken. When in improve mode, any input triggers multiline mode, but the EOF handler doesn't properly exit and process the code.

### Workaround (WORKS PERFECTLY)

**Use chat mode instead:**
```
/mode chat
improve this code: def hello(): print("hello")
```

This gives the same functionality without the bug.

### Alternative Workarounds

1. **Multiline in chat mode:**
   ```
   /mode chat
   /multiline
   Please improve this code:
   def hello():
       print("hello")
   EOF
   ```

2. **Use the standalone improve script:**
   ```bash
   python3 improve_code.py
   # Paste code
   # Type END
   ```

### Status
- **Severity**: High - mode is unusable
- **Workaround**: Available and works well
- **Fix Priority**: Low (since workaround is actually better UX)

### Recommendation
Avoid using `/mode improve` entirely. The chat mode approach is simpler and more reliable.

---

## 2. Other Modes with Similar Issues

The following modes may have similar multiline handling issues:
- `/mode fix`
- `/mode explain` 
- `/mode test`
- `/mode convert`

**Recommendation**: Use chat mode for all of these tasks:
- "fix this code: [code]"
- "explain this code: [code]"
- "write tests for: [code]"
- "convert this to Python: [code]"

---

Last updated: 2024-01-10

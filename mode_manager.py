#!/usr/bin/env python3
"""
Unified mode management for TinyLlama applications.
Provides consistent mode resolution, listing, and prompt formatting.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class ModeInfo:
    """Information about a mode."""
    name: str
    description: str
    group: str
    synonyms: List[str]
    visible_by_default: bool = True
    prompt_template: Optional[str] = None


# Complete mode registry
MODE_REGISTRY = {
    "chat": ModeInfo(
        name="chat",
        description="💬 General conversation",
        group="basic",
        synonyms=["conversation", "talk"],
        prompt_template=None  # Uses default
    ),
    "code": ModeInfo(
        name="code",
        description="💻 Code generation",
        group="basic",
        synonyms=["coding", "program"],
        prompt_template="Write Python code for: {query}"
    ),
    "explain": ModeInfo(
        name="explain",
        description="📖 Explain concepts",
        group="basic",
        synonyms=["explanation", "describe"],
        prompt_template="Explain this concept: {query}"
    ),
    "debug": ModeInfo(
        name="debug",
        description="🐛 Debug code",
        group="basic",
        synonyms=["fix", "troubleshoot"],
        prompt_template="Debug this issue: {query}"
    ),
    "function": ModeInfo(
        name="function",
        description="🔧 Generate functions",
        group="codegen",
        synonyms=["func", "method"],
        prompt_template="Write a Python function that {query}"
    ),
    "class": ModeInfo(
        name="class",
        description="📦 Generate classes",
        group="codegen",
        synonyms=["cls", "object"],
        prompt_template="Write a Python class for {query}"
    ),
    "script": ModeInfo(
        name="script",
        description="📄 Generate scripts",
        group="codegen",
        synonyms=["program", "file"],
        prompt_template="Write a complete Python script that {query}"
    ),
    "test": ModeInfo(
        name="test",
        description="🧪 Generate tests",
        group="codegen",
        synonyms=["unittest", "pytest"],
        prompt_template="Write pytest unit tests for this:\n{query}"
    ),
    "fix": ModeInfo(
        name="fix",
        description="🔧 Fix broken code",
        group="refactor",
        synonyms=["repair", "correct"],
        prompt_template="Fix this broken Python code:\n{query}"
    ),
    "improve": ModeInfo(
        name="improve",
        description="⚡ Improve code",
        group="refactor",
        synonyms=["optimize", "enhance", "refactor"],
        prompt_template="Improve this Python code for better efficiency:\n{query}"
    ),
    "convert": ModeInfo(
        name="convert",
        description="🔄 Convert code",
        group="refactor",
        synonyms=["translate", "port"],
        prompt_template="Convert this code to Python:\n{query}"
    ),
    "plan": ModeInfo(
        name="plan",
        description="📋 Plan implementation",
        group="assistant",
        synonyms=["design", "architect"],
        prompt_template="Create a step-by-step plan to: {query}"
    ),
    "review": ModeInfo(
        name="review",
        description="🔍 Review code",
        group="assistant",
        synonyms=["critique", "analyze"],
        prompt_template="Review this Python code and suggest improvements:\n{query}"
    ),
    "optimize": ModeInfo(
        name="optimize",
        description="⚡ Optimize performance",
        group="assistant",
        synonyms=["performance", "speed"],
        prompt_template="Optimize this Python code for performance:\n{query}"
    ),
    "doc": ModeInfo(
        name="doc",
        description="📝 Add documentation",
        group="assistant",
        synonyms=["document", "docstring", "comment"],
        prompt_template="Add docstrings and comments to this code:\n{query}"
    )
}


def resolve_mode(token: str, allowed_modes: Optional[Set[str]] = None) -> Optional[str]:
    """
    Resolve a mode name or synonym to canonical mode name.
    Case-insensitive and tolerant of variations.
    
    Args:
        token: Mode name or synonym to resolve
        allowed_modes: Optional set of allowed mode names
        
    Returns:
        Canonical mode name if found, None otherwise
    """
    token = token.lower().strip()
    
    # Direct match
    if token in MODE_REGISTRY:
        mode = token
    else:
        # Check synonyms
        mode = None
        for mode_name, info in MODE_REGISTRY.items():
            if token in [syn.lower() for syn in info.synonyms]:
                mode = mode_name
                break
    
    # Check if mode is allowed
    if mode and allowed_modes and mode not in allowed_modes:
        return None
        
    return mode


def list_modes(allowed_modes: Optional[Set[str]] = None, show_all: bool = False) -> str:
    """
    Format mode list for display.
    
    Args:
        allowed_modes: Optional set of allowed modes
        show_all: Show all modes regardless of visibility
        
    Returns:
        Formatted mode list string
    """
    lines = []
    
    # Group modes
    groups: Dict[str, List[Tuple[str, ModeInfo]]] = {}
    for name, info in MODE_REGISTRY.items():
        if allowed_modes and name not in allowed_modes:
            continue
        if not show_all and not info.visible_by_default:
            continue
            
        if info.group not in groups:
            groups[info.group] = []
        groups[info.group].append((name, info))
    
    # Format by group
    group_order = ["basic", "codegen", "refactor", "assistant"]
    for group in group_order:
        if group not in groups:
            continue
            
        modes = groups[group]
        if modes:
            # Group header
            group_names = {
                "basic": "🎯 Basic Modes",
                "codegen": "💻 Code Generation",
                "refactor": "🔧 Code Refactoring",
                "assistant": "🤖 Assistant Features"
            }
            if group in group_names:
                lines.append(f"\n{group_names[group]}:")
            
            # List modes in group
            for name, info in sorted(modes):
                lines.append(f"  {name:12} - {info.description}")
    
    return "\n".join(lines)


def format_prompt(
    query: str,
    mode: str,
    is_finetuned: bool = False,
    custom_templates: Optional[Dict[str, str]] = None
) -> str:
    """
    Format a prompt based on mode and model type.
    
    Args:
        query: User query
        mode: Mode name (must be canonical)
        is_finetuned: Whether using a fine-tuned model
        custom_templates: Optional custom templates by mode
        
    Returns:
        Formatted prompt string
    """
    # Get mode info
    mode_info = MODE_REGISTRY.get(mode)
    if not mode_info:
        mode_info = MODE_REGISTRY["chat"]  # Fallback
    
    # Check custom templates first
    if custom_templates and mode in custom_templates:
        template = custom_templates[mode]
    elif mode_info.prompt_template:
        template = mode_info.prompt_template
    else:
        template = None
    
    # Format based on model type
    if is_finetuned:
        # Fine-tuned model uses Q&A format
        if template:
            formatted_query = template.format(query=query)
        else:
            formatted_query = query
        return f"Question: {formatted_query}\nAnswer:"
    else:
        # Base model uses chat template
        system_prompt = "You are a helpful programming assistant specializing in Python. Provide clear, concise answers focused on practical solutions."
        
        if template:
            user_prompt = template.format(query=query)
        else:
            user_prompt = query
            
        return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"


def get_mode_emoji(mode: str) -> str:
    """Get emoji for a mode."""
    mode_info = MODE_REGISTRY.get(mode)
    if mode_info and mode_info.description:
        # Extract emoji from description
        desc = mode_info.description
        if desc and len(desc) > 2 and desc[1] == " ":
            return desc[0]
    return "?"


def get_app_modes(app_type: str) -> List[str]:
    """
    Get default mode list for a specific app type.
    
    Args:
        app_type: Type of app (chat, codegen, assistant, unified)
        
    Returns:
        List of mode names
    """
    if app_type == "chat":
        return ["chat", "code", "explain", "debug"]
    elif app_type == "codegen":
        return ["code", "function", "class", "script", "test", "fix", "improve", "convert", "chat"]
    elif app_type == "assistant":
        return ["chat", "code", "explain", "debug", "plan", "review", "optimize", "test", "doc"]
    elif app_type == "unified":
        # All modes
        return list(MODE_REGISTRY.keys())
    else:
        return ["chat"]  # Safe default


# Utility functions for state management
def serialize_mode_state(mode: str, context: Optional[Dict] = None) -> Dict:
    """Serialize mode state for persistence."""
    return {
        "mode": mode,
        "context": context or {}
    }


def deserialize_mode_state(state: Dict) -> Tuple[str, Dict]:
    """Deserialize mode state."""
    mode = state.get("mode", "chat")
    context = state.get("context", {})
    return mode, context


class ModeManager:
    """Manages mode state and transitions for TinyLlama applications."""
    
    def __init__(self, modes: List[str], default_mode: str = "chat"):
        """
        Initialize mode manager.
        
        Args:
            modes: List of allowed mode names
            default_mode: Initial mode
        """
        self.allowed_modes = set(modes)
        self.current_mode = default_mode if default_mode in self.allowed_modes else "chat"
        self.context = {}
    
    def change_mode(self, mode_spec: str) -> Optional[str]:
        """
        Change the current mode.
        
        Args:
            mode_spec: Mode specification (e.g., "chat", "to code", "=function")
            
        Returns:
            Success message or None if mode not found
        """
        # Clean up the mode specification
        mode_spec = mode_spec.strip().lower()
        
        # Remove common prefixes
        for prefix in ["to ", "=", ": ", "-> "]:
            if mode_spec.startswith(prefix):
                mode_spec = mode_spec[len(prefix):].strip()
                break
        
        # Resolve the mode
        resolved_mode = resolve_mode(mode_spec, self.allowed_modes)
        
        if resolved_mode:
            old_mode = self.current_mode
            self.current_mode = resolved_mode
            mode_info = MODE_REGISTRY[resolved_mode]
            return f"✅ Switched from {old_mode} to {resolved_mode} - {mode_info.description}"
        else:
            # Try to find similar modes
            suggestions = []
            for mode in self.allowed_modes:
                if mode_spec in mode or mode in mode_spec:
                    suggestions.append(mode)
            
            if suggestions:
                return f"❌ Mode '{mode_spec}' not found. Did you mean: {', '.join(suggestions)}?"
            else:
                return f"❌ Mode '{mode_spec}' not found. Use /modes to see available modes."
    
    def list_modes(self):
        """Print available modes."""
        print("\n🎭 Available Modes:")
        print(list_modes(self.allowed_modes))
        print(f"\n📍 Current mode: {self.current_mode}")
    
    def get_mode_info(self) -> str:
        """Get current mode information."""
        mode_info = MODE_REGISTRY.get(self.current_mode)
        if mode_info:
            return f"{self.current_mode} - {mode_info.description}"
        return self.current_mode
    
    def format_prompt(self, query: str, is_finetuned: bool = False) -> str:
        """Format prompt for current mode."""
        return format_prompt(query, self.current_mode, is_finetuned)

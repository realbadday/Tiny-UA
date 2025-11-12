#!/usr/bin/env python3
"""
Robust slash-command router for TinyLlama interactive apps.
Parses and dispatches commands consistently across entry points.
"""

import re
from typing import Callable, Dict, Optional, Set

from mode_manager import resolve_mode


class CommandRouter:
    """Router that parses slash commands and delegates to callbacks/handlers."""

    def __init__(
        self,
        allowed_modes: Set[str],
        tts_commands: Optional[object],
        callbacks: Dict[str, Callable[..., None]]
    ):
        """
        Initialize the command router.
        
        Args:
            allowed_modes: Set of allowed mode names
            tts_commands: TTSCommands-like object or None
            callbacks: Dict of handlers: 'set_mode', 'list_modes', 'enter_multiline',
                       'clear_cache', 'toggle_cache', 'print_help', 'exit_app'
        """
        self.allowed_modes = allowed_modes
        self.tts_commands = tts_commands
        self.cb = callbacks

    def parse_and_handle(self, line: str) -> bool:
        """
        Parse a line. If it's a command, handle it and return True. Otherwise return False.
        """
        line = line.strip()
        if not line.startswith('/'):
            return False

        low = line.lower()

        # Quit aliases
        if low in ['/quit', '/exit', '/bye']:
            self.cb.get('exit_app', lambda: None)()
            return True

        # Help
        if low == '/help' or low == '/?':
            self.cb.get('print_help', lambda: None)()
            return True

        # Modes listing
        if low == '/modes':
            self.cb.get('list_modes', lambda: None)()
            return True

        # Multiline
        if low == '/multiline':
            self.cb.get('enter_multiline', lambda: None)()
            return True

        # Cache controls
        if low == '/clear':
            self.cb.get('clear_cache', lambda: None)()
            return True
        if low == '/nocache':
            self.cb.get('toggle_cache', lambda: None)()
            return True

        # TTS commands (delegated to TTSCommands if present)
        if self.tts_commands:
            # Accept anything starting with these tokens as TTS-related
            if any(low.startswith(cmd) for cmd in ['/tts', '/speak', '/mute', '/rate', '/volume', '/voice', '/voices', '/tts-help']):
                # Delegate exact line to TTSCommands
                try:
                    handled = self.tts_commands.handle_command(line)
                    return True if handled else False
                except Exception:
                    return True

        # Mode switching (tolerant)
        if low.startswith('/mode'):
            # Accept '/mode chat', '/mode: chat', '/mode=chat', '/mode to chat', '/mode switch to CHAT please'
            tokens = re.split(r'[\s=:,]+', line.strip())
            target = None
            for tok in tokens[1:]:
                if not tok:
                    continue
                resolved = resolve_mode(tok, self.allowed_modes)
                if resolved:
                    target = resolved
                    break
            # If not resolved, try extracting after the word 'to'
            if not target:
                m = re.search(r"\bto\b\s+([A-Za-z\-]+)", line, flags=re.IGNORECASE)
                if m:
                    tok = m.group(1)
                    target = resolve_mode(tok, self.allowed_modes)
            if target:
                self.cb.get('set_mode', lambda m: None)(target)
            else:
                # Fall back to listing modes
                self.cb.get('list_modes', lambda: None)()
            return True

        # Unknown command – show help
        self.cb.get('print_help', lambda: None)()
        return True


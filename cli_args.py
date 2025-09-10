#!/usr/bin/env python3
"""
Shared CLI argument parser for TinyLlama applications.
Provides consistent command-line interface across all entry points.
"""

import argparse
from typing import List, Optional, Dict, Any


def build_parser(
    app_name: str,
    description: str,
    allowed_modes: List[str],
    default_mode: str = "chat",
    supports_query: bool = True,
    epilog: Optional[str] = None
) -> argparse.ArgumentParser:
    """
    Build a standardized argument parser for TinyLlama applications.
    
    Args:
        app_name: Name of the application
        description: Application description
        allowed_modes: List of valid modes for this app
        default_mode: Default mode (usually "chat")
        supports_query: Whether app accepts direct queries
        epilog: Optional epilog text
        
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=app_name,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    # Positional arguments
    if supports_query:
        parser.add_argument(
            'query',
            nargs='*',
            help='Direct query (optional). Interactive mode if not provided.'
        )
    
    # Mode selection
    parser.add_argument(
        '-m', '--mode',
        choices=allowed_modes,
        default=default_mode,
        help=f'Initial mode (default: {default_mode})'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path',
        help='Path to model directory or HuggingFace model name'
    )
    
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='Disable INT8 quantization (uses more memory)'
    )
    
    # Cache control
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable response caching'
    )
    
    # TTS configuration
    parser.add_argument(
        '--tts',
        action='store_true',
        help='Enable text-to-speech output'
    )
    
    parser.add_argument(
        '--no-tts',
        action='store_true',
        help='Explicitly disable text-to-speech (default)'
    )
    
    parser.add_argument(
        '--tts-rate',
        type=int,
        default=175,
        metavar='WPM',
        help='TTS speech rate in words per minute (50-300, default: 175)'
    )
    
    parser.add_argument(
        '--tts-volume',
        type=float,
        default=0.9,
        metavar='VOL',
        help='TTS volume level (0.0-1.0, default: 0.9)'
    )
    
    return parser


def apply_tts_settings(args: argparse.Namespace, tts_manager: Any) -> None:
    """
    Apply TTS settings from command line arguments to TTS manager.
    
    Args:
        args: Parsed command line arguments
        tts_manager: TTSManager instance
    """
    if not tts_manager:
        return
        
    # Apply rate and volume if TTS is being enabled
    if hasattr(tts_manager, 'set_rate') and args.tts_rate:
        tts_manager.set_rate(args.tts_rate)
        
    if hasattr(tts_manager, 'set_volume') and args.tts_volume is not None:
        tts_manager.set_volume(args.tts_volume)


def determine_tts_enabled(args: argparse.Namespace) -> bool:
    """
    Determine if TTS should be enabled based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Whether TTS should be enabled
    """
    # --no-tts explicitly disables, --tts explicitly enables
    # Default is disabled unless --tts is specified
    return args.tts and not args.no_tts


def get_standard_epilog() -> str:
    """
    Get standard epilog text for help output.
    
    Returns:
        Formatted epilog string
    """
    return """
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s "explain decorators"     # Single query
  %(prog)s --mode code "quicksort"  # Code generation
  %(prog)s --tts                    # With voice output

TTS Commands (in interactive mode):
  /tts         Toggle voice output
  /voices      List available voices
  /voice NAME  Set voice
  /rate WPM    Set speech rate
  /volume 0-1  Set volume level
"""


# For testing or validation
def validate_args(args: argparse.Namespace, allowed_modes: List[str]) -> None:
    """
    Validate parsed arguments.
    
    Args:
        args: Parsed arguments
        allowed_modes: List of valid modes
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.mode not in allowed_modes:
        raise ValueError(f"Invalid mode: {args.mode}. Must be one of: {', '.join(allowed_modes)}")
        
    if args.tts_rate < 50 or args.tts_rate > 300:
        raise ValueError(f"TTS rate must be between 50 and 300 WPM, got: {args.tts_rate}")
        
    if args.tts_volume < 0.0 or args.tts_volume > 1.0:
        raise ValueError(f"TTS volume must be between 0.0 and 1.0, got: {args.tts_volume}")

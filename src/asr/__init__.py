"""
Downstream ASR Adaptation Module
"""

from .adaptation import WhisperAccentAdaptor, REGIONAL_PROMPTS, PHONETIC_ERROR_PATTERNS

__all__ = ["WhisperAccentAdaptor", "REGIONAL_PROMPTS", "PHONETIC_ERROR_PATTERNS"]

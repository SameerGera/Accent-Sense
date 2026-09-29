"""
Downstream ASR Adaptation Module
"""

from .adaptation import PHONETIC_ERROR_PATTERNS, REGIONAL_PROMPTS, WhisperAccentAdaptor

__all__ = ["PHONETIC_ERROR_PATTERNS", "REGIONAL_PROMPTS", "WhisperAccentAdaptor"]

from .baseline_mfcc import ClassicalAcousticBaseline, extract_acoustic_features

try:
    from .wavlm_classifier import WavLMForL1Influence, AttentiveStatisticsPooling
    __all__ = [
        "ClassicalAcousticBaseline",
        "extract_acoustic_features",
        "WavLMForL1Influence",
        "AttentiveStatisticsPooling",
    ]
except ImportError:
    __all__ = [
        "ClassicalAcousticBaseline",
        "extract_acoustic_features",
    ]

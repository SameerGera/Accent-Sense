__all__ = []

try:
    from .svarah_dataset import (
        SvarahSpeechDataset,
        build_speaker_disjoint_splits,
        LANGUAGE_FAMILY_MAP,
        REGIONAL_4CLASS_TARGETS,
        label_regional_4class,
    )
    __all__.extend([
        "SvarahSpeechDataset",
        "build_speaker_disjoint_splits",
        "LANGUAGE_FAMILY_MAP",
        "REGIONAL_4CLASS_TARGETS",
        "label_regional_4class",
    ])
except ImportError:
    pass

try:
    from .vctk_dataset import UKAccentDataset, collate_pad, load_manifest, save_manifest, CLASSES
    __all__.extend(["UKAccentDataset", "collate_pad", "load_manifest", "save_manifest", "CLASSES"])
except ImportError:
    pass


CLASSES = (
    "Southern",
    "Scottish",
    "Welsh",
    "Northern",
    "Midlands",
    "Irish",
)

CLASS_TO_IDX = {
    name: i
    for i, name in enumerate(CLASSES)
}

DISPLAY_NAMES = {
    "Southern": "Southern England / RP-like",
    "Scottish": "Scottish",
    "Welsh": "Welsh",
    "Northern": "Northern England",
    "Midlands": "Midlands / West Midlands",
    "Irish": "Irish / Northern Irish",
}

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 6.0
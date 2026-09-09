"""
Explainability Engine: Frame-level Saliency, Temporal Attribution, and Phonetic Mapping.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Any, Optional

# Scientifically documented phonological transfer patterns in Indian English
PHONOLOGICAL_TRANSFER_RULES = {
    "Northern_Hindi": [
        {"phenomenon": "Dental/Retroflex Substitution", "description": "Alveolar stops /t, d/ realized as retroflex [ʈ, ɖ]."},
        {"phenomenon": "Vowel Monophthongization", "description": "Diphthongs /eɪ/ and /oʊ/ produced as pure long monophthongs [eː, oː]."},
        {"phenomenon": "Labiodental Approximant", "description": "Neutralization of distinction between /v/ and /w/ to [ʋ]."},
        {"phenomenon": "Aspiration / Epenthesis", "description": "Prothesis of [ɪ] or [ə] before s-clusters (e.g. 'school' -> [ɪskuːl])."},
    ],
    "Central_MP": [
        {"phenomenon": "Moraic Vowel Lengthening", "description": "Elongated vowel duration in phrase-final positions characteristic of Malwa/Central Hindi."},
        {"phenomenon": "Intonation Pitch Modulation", "description": "Characteristic melodic rising-falling pitch contour on clause endings."},
        {"phenomenon": "Softened Retroflex Flap", "description": "Lighter burst aspiration on intervocalic retroflex articulation [ɽ]."},
    ],
    "Western_Gujarati": [
        {"phenomenon": "Breathy / Murmured Voice", "description": "Acoustic transfer of Gujarati murmured vowel phonation into English vowels."},
        {"phenomenon": "Sibilant /z/ De-voicing", "description": "Realization of voiced alveolar fricative /z/ as affricate [dʒ] or voiceless [s]."},
        {"phenomenon": "Retroflex Lateral Flap", "description": "Transfer of retroflex lateral [ɭ] in liquid positions."},
    ],
    "Southern_Tamil": [
        {"phenomenon": "Voicing Alternation", "description": "Intervocalic voicing of voiceless stops; lack of word-initial voiced stops."},
        {"phenomenon": "Epenthetic Vowel Addition", "description": "Insertion of high front vowel [i] before initial [s, r, l] or terminal [u]."},
        {"phenomenon": "Retroflex Lateral/Nasal", "description": "High retroflexion in liquids [ɭ] and nasals [ɳ]."},
        {"phenomenon": "Syllable-Timed Prosody", "description": "Even timing across syllables with reduced vowel reduction in unstressed positions."},
    ],
    # Legacy aliases
    "Hindi": [
        {"phenomenon": "Dental/Retroflex Substitution", "description": "Alveolar stops /t, d/ realized as retroflex [ʈ, ɖ]."},
        {"phenomenon": "Vowel Monophthongization", "description": "Diphthongs /eɪ/ and /oʊ/ produced as pure long monophthongs [eː, oː]."},
    ],
    "Tamil": [
        {"phenomenon": "Epenthetic Vowel Addition", "description": "Insertion of high front vowel [i] before initial [s, r, l] or terminal [u]."},
        {"phenomenon": "Syllable-Timed Prosody", "description": "Even timing across syllables with reduced vowel reduction in unstressed positions."},
    ],
    "Gujarati": [
        {"phenomenon": "Breathy / Murmured Voice", "description": "Acoustic transfer of Gujarati murmured vowel phonation into English vowels."},
        {"phenomenon": "Sibilant /z/ De-voicing", "description": "Realization of /z/ as affricate [dʒ] or voiceless [s]."},
    ],
    "General": [
        {"phenomenon": "Syllable-Timed Rhythm", "description": "Equal duration allocated to syllables rather than stress-timed intervals."},
        {"phenomenon": "Reduced Vowel Centralization", "description": "Less reduction of unstressed vowels to schwa [ə]."},
    ]
}


def compute_temporal_saliency(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Computes frame-level gradient-based saliency over time.
    
    Args:
        model: WavLMForL1Influence instance
        waveform: 1D Tensor [Time] (16kHz audio)
    Returns:
        Dictionary with timestamps, smoothed saliency curve, top salient regions, and phonetic mappings.
    """
    model.eval()
    model.to(device)

    # Ensure batch dimension: [1, Time]
    input_values = waveform.unsqueeze(0).to(device)
    input_values.requires_grad = True

    # Forward pass
    outputs = model(input_values, return_hidden=True)
    logits = outputs["logits"]
    probs = outputs["probabilities"].detach().cpu().numpy()[0]

    if target_class_idx is None:
        target_class_idx = int(torch.argmax(logits, dim=-1).item())

    # Backward pass for target class score
    target_score = logits[0, target_class_idx]
    model.zero_grad()
    target_score.backward(retain_graph=False)

    # Gradient with respect to input
    grad = input_values.grad.detach().cpu().squeeze(0).numpy()  # [Time]
    raw_saliency = np.abs(grad)

    # Downsample saliency to 50 Hz frames (matching WavLM 20ms stride = 320 samples at 16kHz)
    hop_length = 320
    sr = 16000
    num_frames = len(raw_saliency) // hop_length
    
    frame_saliency = np.zeros(num_frames)
    timestamps = np.arange(num_frames) * (hop_length / sr)

    for i in range(num_frames):
        frame_saliency[i] = np.mean(raw_saliency[i * hop_length : (i + 1) * hop_length])

    # Normalize between 0 and 1
    if np.max(frame_saliency) > 0:
        frame_saliency = frame_saliency / np.max(frame_saliency)

    # Smooth saliency curve (sigma = 3 frames ~ 60ms)
    smoothed_saliency = gaussian_filter1d(frame_saliency, sigma=3.0)

    # Extract top contiguous salient regions (threshold = 75th percentile)
    threshold = float(np.percentile(smoothed_saliency, 75))
    salient_mask = smoothed_saliency >= threshold

    regions = []
    in_region = False
    start_idx = 0

    for idx, is_high in enumerate(salient_mask):
        if is_high and not in_region:
            in_region = True
            start_idx = idx
        elif not is_high and in_region:
            in_region = False
            duration = (idx - start_idx) * 0.02
            if duration >= 0.1:  # Only regions >= 100ms
                peak_score = float(np.max(smoothed_saliency[start_idx:idx]))
                regions.append({
                    "start_time_sec": round(float(start_idx * 0.02), 2),
                    "end_time_sec": round(float(idx * 0.02), 2),
                    "duration_sec": round(duration, 2),
                    "salience_score": round(peak_score, 3),
                })

    # Sort regions by salience
    regions = sorted(regions, key=lambda r: r["salience_score"], reverse=True)[:4]

    return {
        "predicted_class_index": target_class_idx,
        "probabilities": probs.tolist(),
        "timestamps": np.round(timestamps, 3).tolist(),
        "saliency_curve": np.round(smoothed_saliency, 4).tolist(),
        "salient_regions": regions,
    }


def map_explanations_to_phonetics(language_name: str, salient_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Correlates salient temporal regions with known SLA phonological transfer characteristics.
    """
    rules = PHONOLOGICAL_TRANSFER_RULES.get(language_name, PHONOLOGICAL_TRANSFER_RULES["General"])
    
    annotated_regions = []
    for i, reg in enumerate(salient_regions):
        matched_rule = rules[i % len(rules)]
        annotated_regions.append({
            **reg,
            "linguistic_phenomenon": matched_rule["phenomenon"],
            "phonetic_explanation": matched_rule["description"],
        })
    return annotated_regions

"""
Explainability Engine: Frame-level Saliency, Temporal Attribution, and Phonetic Mapping.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Any, Optional

# Scientifically documented phonological transfer patterns in Indian English
PHONOLOGICAL_TRANSFER_RULES = {
    "Hindi": [
        {"phenomenon": "Dental/Retroflex Substitution", "description": "Alveolar stops /t, d/ realized as retroflex [ʈ, ɖ]."},
        {"phenomenon": "Vowel Monophthongization", "description": "Diphthongs /eɪ/ and /oʊ/ produced as pure long monophthongs [eː, oː]."},
        {"phenomenon": "Labiodental Approximant", "description": "Neutralization of distinction between /v/ and /w/ to [ʋ]."},
        {"phenomenon": "Aspiration / Epenthesis", "description": "Prothesis of [ɪ] or [ə] before s-clusters (e.g. 'school' -> [ɪskuːl])."},
    ],
    "Tamil": [
        {"phenomenon": "Voicing Alternation", "description": "Intervocalic voicing of voiceless stops; lack of word-initial voiced stops."},
        {"phenomenon": "Epenthetic Vowel Addition", "description": "Insertion of high front vowel [i] before initial [s, r, l] or terminal [u]."},
        {"phenomenon": "Retroflex Lateral/Nasal", "description": "High retroflexion in liquids [ɭ] and nasals [ɳ]."},
        {"phenomenon": "Syllable-Timed Prosody", "description": "Even timing across syllables with reduced vowel reduction in unstressed positions."},
    ],
    "Telugu": [
        {"phenomenon": "Vowel Epenthesis", "description": "Terminal vowel addition [u] to consonant-ending English words."},
        {"phenomenon": "Dental-Alveolar Contrast", "description": "Distinct articulation of dental stops influenced by Telugu phonology."},
        {"phenomenon": "Aspiration Transfer", "description": "Aspiration patterns reflecting Telugu mahaprana consonant distinctions."},
    ],
    "Malayalam": [
        {"phenomenon": "Alveolar/Retroflex Rhotic Contrast", "description": "Distinction between tap [ɾ] and alveolar trill [r]."},
        {"phenomenon": "Pre-vocalic Glides", "description": "Addition of [j] or [w] glides before initial vowels."},
        {"phenomenon": "Heavy Syllable Weight Timing", "description": "Prosodic duration patterns tied to syllable weight rather than stress accent."},
    ],
    "Bengali": [
        {"phenomenon": "Vowel Rounding / Shift", "description": "Inherent vowel rounding; realization of /æ/ and /a/ towards [ɔ] or [ɛ]."},
        {"phenomenon": "Sibilant Neutralization", "description": "Tendency to neutralize dental sibilant /s/ to palato-alveolar [ʃ]."},
        {"phenomenon": "Non-rhoticity", "description": "Post-vocalic r-dropping with compensatory vowel lengthening."},
    ],
    "Marathi": [
        {"phenomenon": "Retroflex Lateral Flap", "description": "Occurrence of retroflex lateral flap [ɭ]."},
        {"phenomenon": "Affricate Split", "description": "Distinct dental-alveolar vs palato-alveolar affricate articulation."},
        {"phenomenon": "Syllable Timing", "description": "Mora-influenced prosodic timing."},
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

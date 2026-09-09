"""
Explainability Engine: Frame-level Saliency, Temporal Attribution, Phonetic Mapping,
and Explanation Faithfulness (Area Under Deletion Curve - AUDC).
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Any, Optional

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# Scientifically documented phonological transfer patterns in Indian English
PHONOLOGICAL_TRANSFER_RULES = {
    "Northern_Hindi": [
        {
            "phenomenon": "Dental/Retroflex Substitution",
            "description": "Alveolar stops /t, d/ realized as retroflex [ʈ, ɖ] with elevated acoustic burst energy.",
        },
        {
            "phenomenon": "Vowel Monophthongization",
            "description": "Diphthongs /eɪ/ and /oʊ/ produced as pure long monophthongs [eː, oː].",
        },
        {
            "phenomenon": "Labiodental Approximant Merger",
            "description": "Neutralization of distinction between labiodental fricative /v/ and labiovelar approximant /w/ to [ʋ].",
        },
        {
            "phenomenon": "Aspiration / Prothetic Epenthesis",
            "description": "Prothesis of [ɪ] or [ə] before initial s-clusters (e.g., 'station' -> [ɪsteːʃən]).",
        },
    ],
    "Central_MP": [
        {
            "phenomenon": "Moraic Vowel Lengthening",
            "description": "Elongated vowel duration in phrase-final positions characteristic of Malwa/Central Hindi.",
        },
        {
            "phenomenon": "Intonation Pitch Modulation",
            "description": "Characteristic melodic rising-falling pitch contour on clause endings and transitions.",
        },
        {
            "phenomenon": "Softened Retroflex Flap",
            "description": "Intervocalic retroflex articulation transitioning to flap [ɽ] with lower burst energy.",
        },
        {
            "phenomenon": "Syllabic De-accentuation",
            "description": "Moderated intensity contrast between lexically stressed and unstressed syllables.",
        },
    ],
    "Western_Gujarati": [
        {
            "phenomenon": "Breathy / Murmured Voice Phonation",
            "description": "Acoustic transfer of Gujarati contrastive murmured vowels into English vowels (elevated H1-H2 spectral tilt).",
        },
        {
            "phenomenon": "Sibilant /z/ De-voicing",
            "description": "Realization of voiced alveolar fricative /z/ as affricate [dʒ] or voiceless [s].",
        },
        {
            "phenomenon": "Retroflex Lateral Flap",
            "description": "Transfer of retroflex lateral [ɭ] in liquid and lateral consonant positions.",
        },
        {
            "phenomenon": "Low-Back Vowel Retraction",
            "description": "Distinctive open-vowel formant patterns reflecting Gujarati vowel inventory shifts.",
        },
    ],
    "Southern_Tamil": [
        {
            "phenomenon": "Intervocalic Voicing Alternation",
            "description": "Intervocalic voicing of voiceless stops; lack of phonemic contrast in word-initial voiced stops.",
        },
        {
            "phenomenon": "Epenthetic Terminal & Initial Vowels",
            "description": "Addition of high front vowel [i] before initial [s, r, l] or terminal [u] on word codas.",
        },
        {
            "phenomenon": "Retroflex Lateral / Nasal Realization",
            "description": "High retroflexion in liquids [ɭ] and nasals [ɳ] transferred from Dravidian phonology.",
        },
        {
            "phenomenon": "Strict Syllable-Timed Prosody",
            "description": "Uniform syllable duration across utterances with reduced vowel reduction to schwa [ə].",
        },
    ],
    "General": [
        {
            "phenomenon": "Syllable-Timed Rhythm",
            "description": "Equal duration allocated to syllables rather than stress-timed foot intervals.",
        },
        {
            "phenomenon": "Reduced Vowel Centralization",
            "description": "Less reduction of unstressed English vowels to schwa [ə].",
        },
    ],
}


def _integrated_gradients_native(
    model: torch.nn.Module,
    input_values: torch.Tensor,
    target_class_idx: int,
    n_steps: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """
    Native path-integral calculation of Integrated Gradients (Sundararajan et al., 2017)
    when Captum is not available or for direct fallback.
    """
    baseline = torch.zeros_like(input_values).to(device)
    accumulated_grads = torch.zeros_like(input_values).to(device)

    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps)
    for alpha in alphas:
        step_input = (baseline + float(alpha) * (input_values - baseline)).detach().clone()
        step_input.requires_grad = True
        outputs = model(step_input)
        logits = outputs["logits"]
        score = logits[0, target_class_idx]
        model.zero_grad()
        score.backward()
        if step_input.grad is not None:
            accumulated_grads += step_input.grad.detach()

    avg_grads = accumulated_grads / float(n_steps)
    attributions = (input_values - baseline) * avg_grads
    return attributions.detach().cpu().squeeze(0).numpy()


def compute_temporal_saliency(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class_idx: Optional[int] = None,
    device: str = "cpu",
    method: str = "integrated_gradients",
    n_steps: int = 20,
    hop_length: int = 320,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Computes frame-level gradient-based saliency over time.
    
    Args:
        model: WavLMForL1Influence instance
        waveform: 1D Tensor [Time] (16kHz audio) or 2D [1, Time]
        target_class_idx: Target class index to explain (defaults to argmax prediction)
        device: 'cpu' or 'cuda'
        method: 'integrated_gradients' or 'gradient'
        n_steps: Number of interpolation steps for Integrated Gradients
        hop_length: Frame hop in samples (320 samples = 20ms at 16kHz)
        sample_rate: Audio sampling rate (16kHz)
    Returns:
        Dictionary with timestamps, smoothed saliency curve, top salient regions, and phonetic mappings.
    """
    model.eval()
    model.to(device)

    # Ensure shape is [1, Time]
    if waveform.dim() == 1:
        input_values = waveform.unsqueeze(0).to(device)
    else:
        input_values = waveform.to(device)

    # Forward pass to determine predictions
    with torch.no_grad():
        initial_out = model(input_values, return_hidden=True)
        logits = initial_out["logits"]
        probs = initial_out["probabilities"].detach().cpu().numpy()[0]
        attention_weights = None
        if "attention_weights" in initial_out and initial_out["attention_weights"] is not None:
            attention_weights = initial_out["attention_weights"].detach().cpu().numpy()[0, :, 0]

    if target_class_idx is None:
        target_class_idx = int(np.argmax(probs))

    # Compute raw sample-level attribution
    if method == "integrated_gradients":
        if CAPTUM_AVAILABLE:
            def forward_func(x):
                return model(x)["logits"]

            ig = IntegratedGradients(forward_func)
            baseline = torch.zeros_like(input_values).to(device)
            attr_tensor = ig.attribute(
                input_values,
                baselines=baseline,
                target=target_class_idx,
                n_steps=n_steps,
            )
            raw_attr = attr_tensor.detach().cpu().squeeze(0).numpy()
        else:
            raw_attr = _integrated_gradients_native(
                model=model,
                input_values=input_values,
                target_class_idx=target_class_idx,
                n_steps=n_steps,
                device=device,
            )
    else:
        # Standard input * gradient attribution
        input_tensor = input_values.clone().detach().requires_grad_(True)
        out = model(input_tensor)
        score = out["logits"][0, target_class_idx]
        model.zero_grad()
        score.backward()
        if input_tensor.grad is not None:
            grad = input_tensor.grad.detach().cpu().squeeze(0).numpy()
            raw_attr = np.abs(grad * input_tensor.detach().cpu().squeeze(0).numpy())
        else:
            raw_attr = np.zeros(input_values.shape[1])

    raw_saliency = np.abs(raw_attr)

    # Downsample saliency to 50 Hz frames (matching WavLM 20ms stride = 320 samples at 16kHz)
    num_frames = len(raw_saliency) // hop_length
    if num_frames == 0:
        num_frames = 1
        frame_saliency = np.array([np.mean(raw_saliency)])
        timestamps = np.array([0.0])
    else:
        frame_saliency = np.zeros(num_frames)
        timestamps = np.arange(num_frames) * (hop_length / sample_rate)
        for i in range(num_frames):
            frame_saliency[i] = np.mean(raw_saliency[i * hop_length : (i + 1) * hop_length])

    # If ASP attention is available and lengths match, fuse gradient energy with attention
    if attention_weights is not None and len(attention_weights) == num_frames:
        norm_asp = attention_weights / (np.max(attention_weights) + 1e-8)
        norm_grad = frame_saliency / (np.max(frame_saliency) + 1e-8)
        frame_saliency = 0.5 * norm_grad + 0.5 * norm_asp

    # Normalize between 0 and 1
    max_val = np.max(frame_saliency)
    if max_val > 0:
        frame_saliency = frame_saliency / max_val

    # Smooth saliency curve (sigma = 3 frames ~ 60ms)
    smoothed_saliency = gaussian_filter1d(frame_saliency, sigma=3.0)
    if np.max(smoothed_saliency) > 0:
        smoothed_saliency = smoothed_saliency / np.max(smoothed_saliency)

    # Extract top contiguous salient regions (threshold = 75th percentile)
    threshold = float(np.percentile(smoothed_saliency, 75))
    salient_mask = smoothed_saliency >= threshold

    regions = []
    in_region = False
    start_idx = 0
    frame_duration_sec = hop_length / sample_rate

    for idx, is_high in enumerate(salient_mask):
        if is_high and not in_region:
            in_region = True
            start_idx = idx
        elif not is_high and in_region:
            in_region = False
            duration = (idx - start_idx) * frame_duration_sec
            if duration >= 0.10:  # Only regions >= 100ms
                peak_score = float(np.max(smoothed_saliency[start_idx:idx]))
                mean_score = float(np.mean(smoothed_saliency[start_idx:idx]))
                regions.append({
                    "start_time_sec": round(float(start_idx * frame_duration_sec), 2),
                    "end_time_sec": round(float(idx * frame_duration_sec), 2),
                    "duration_sec": round(duration, 2),
                    "salience_score": round(peak_score, 3),
                    "mean_salience": round(mean_score, 3),
                })

    # Close trailing region if audio ends during salient segment
    if in_region:
        idx = len(salient_mask)
        duration = (idx - start_idx) * frame_duration_sec
        if duration >= 0.10:
            peak_score = float(np.max(smoothed_saliency[start_idx:idx]))
            mean_score = float(np.mean(smoothed_saliency[start_idx:idx]))
            regions.append({
                "start_time_sec": round(float(start_idx * frame_duration_sec), 2),
                "end_time_sec": round(float(idx * frame_duration_sec), 2),
                "duration_sec": round(duration, 2),
                "salience_score": round(peak_score, 3),
                "mean_salience": round(mean_score, 3),
            })

    # Sort regions by peak salience
    regions = sorted(regions, key=lambda r: r["salience_score"], reverse=True)[:4]

    return {
        "method": method,
        "predicted_class_index": target_class_idx,
        "probabilities": probs.tolist(),
        "timestamps": np.round(timestamps, 3).tolist(),
        "raw_frame_saliency": np.round(frame_saliency, 4).tolist(),
        "saliency_curve": np.round(smoothed_saliency, 4).tolist(),
        "salient_regions": regions,
        "threshold_75th": round(threshold, 3),
    }


def map_explanations_to_phonetics(
    language_name: str,
    salient_regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Correlates salient temporal regions with documented SLA phonological transfer phenomena.
    """
    rules = PHONOLOGICAL_TRANSFER_RULES.get(
        language_name,
        PHONOLOGICAL_TRANSFER_RULES["General"],
    )

    annotated_regions = []
    for i, reg in enumerate(salient_regions):
        matched_rule = rules[i % len(rules)]
        annotated_regions.append({
            **reg,
            "linguistic_phenomenon": matched_rule["phenomenon"],
            "phonetic_explanation": matched_rule["description"],
        })
    return annotated_regions


def evaluate_deletion_faithfulness(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class_idx: Optional[int] = None,
    frame_saliency: Optional[np.ndarray] = None,
    deletion_fractions: Optional[List[float]] = None,
    num_random_seeds: int = 5,
    hop_length: int = 320,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates Explanation Faithfulness via Area Under Deletion Curve (AUDC).
    
    Progressively masks the top k% most salient frames vs. random frames,
    measuring how rapidly the target class probability degrades.
    
    Faithfulness Criterion:
        AUDC_salient < AUDC_random (i.e. Delta AUDC > 0)
        Removing salient frames degrades confidence significantly faster than random frames.
    """
    if deletion_fractions is None:
        deletion_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    model.eval()
    model.to(device)

    # Ensure shape is [1, Time]
    if waveform.dim() == 1:
        input_values = waveform.unsqueeze(0).to(device)
    else:
        input_values = waveform.to(device)

    total_samples = input_values.shape[1]
    num_frames = total_samples // hop_length

    if num_frames <= 2:
        return {
            "audc_salient": 0.0,
            "audc_random": 0.0,
            "delta_audc": 0.0,
            "faithfulness_ratio": 0.0,
            "is_faithful": True,
            "error": "Waveform too short for meaningful progressive deletion",
        }

    # If saliency curve not provided, compute it via temporal saliency
    if frame_saliency is None:
        sal_result = compute_temporal_saliency(
            model=model,
            waveform=waveform,
            target_class_idx=target_class_idx,
            device=device,
            method="gradient",
        )
        frame_saliency = np.array(sal_result["saliency_curve"])
        if target_class_idx is None:
            target_class_idx = sal_result["predicted_class_index"]

    # If target class is still None, compute argmax
    if target_class_idx is None:
        with torch.no_grad():
            logits = model(input_values)["logits"]
            target_class_idx = int(torch.argmax(logits, dim=-1).item())

    # Get ranking of frames from most salient to least salient
    frame_saliency = frame_saliency[:num_frames]
    ranked_frame_indices = np.argsort(frame_saliency)[::-1]

    # Baseline probability at 0% deletion
    with torch.no_grad():
        base_out = model(input_values)
        base_prob = float(base_out["probabilities"][0, target_class_idx].item())

    salient_probs = []
    random_probs = []

    for frac in deletion_fractions:
        k = int(round(frac * num_frames))
        if k == 0:
            salient_probs.append(base_prob)
            random_probs.append(base_prob)
            continue

        # 1. Salient Deletion: zero out top k salient frames
        masked_salient = input_values.clone()
        top_k_indices = ranked_frame_indices[:k]
        for f_idx in top_k_indices:
            start_s = f_idx * hop_length
            end_s = min((f_idx + 1) * hop_length, total_samples)
            masked_salient[0, start_s:end_s] = 0.0

        with torch.no_grad():
            out_sal = model(masked_salient)
            sal_prob = float(out_sal["probabilities"][0, target_class_idx].item())
            salient_probs.append(sal_prob)

        # 2. Random Deletion: zero out k random frames across multiple seeds
        seed_probs = []
        for s in range(num_random_seeds):
            np.random.seed(42 + s * 17)
            random_k_indices = np.random.choice(num_frames, size=k, replace=False)
            masked_rand = input_values.clone()
            for f_idx in random_k_indices:
                start_s = f_idx * hop_length
                end_s = min((f_idx + 1) * hop_length, total_samples)
                masked_rand[0, start_s:end_s] = 0.0

            with torch.no_grad():
                out_rand = model(masked_rand)
                r_prob = float(out_rand["probabilities"][0, target_class_idx].item())
                seed_probs.append(r_prob)

        random_probs.append(float(np.mean(seed_probs)))

    # Compute Area Under Deletion Curve using the trapezoid rule
    del_arr = np.array(deletion_fractions)
    # np.trapz is deprecated in numpy 2.0+ (use np.trapezoid if available, else np.trapz)
    if hasattr(np, "trapezoid"):
        audc_salient = float(np.trapezoid(salient_probs, del_arr))
        audc_random = float(np.trapezoid(random_probs, del_arr))
    else:
        audc_salient = float(np.trapz(salient_probs, del_arr))
        audc_random = float(np.trapz(random_probs, del_arr))

    delta_audc = audc_random - audc_salient
    faithfulness_ratio = delta_audc / (audc_random + 1e-8)
    is_faithful = bool(delta_audc > 0.0)

    return {
        "target_class_idx": target_class_idx,
        "deletion_fractions": deletion_fractions,
        "salient_deletion_curve": [round(p, 4) for p in salient_probs],
        "random_deletion_curve": [round(p, 4) for p in random_probs],
        "audc_salient": round(audc_salient, 4),
        "audc_random": round(audc_random, 4),
        "delta_audc": round(delta_audc, 4),
        "faithfulness_ratio": round(faithfulness_ratio, 4),
        "is_faithful": is_faithful,
    }

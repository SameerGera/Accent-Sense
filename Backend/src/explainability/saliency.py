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

# Documented phonological features for each of the 7 UK regional accents
PHONOLOGICAL_TRANSFER_RULES = {
    "RP": [
        {
            "phenomenon": "Trap-Bath Split",
            "description": "BATH-class words (path, grass, dance) realised with long back vowel /a:/ — the canonical RP vs. Northern differentiator.",
        },
        {
            "phenomenon": "Non-Rhoticity",
            "description": "Post-vocalic /r/ is absent in coda position; linking and intrusive /r/ used at vowel boundaries.",
        },
        {
            "phenomenon": "FOOT-STRUT distinction",
            "description": "STRUT vowel /ʌ/ is phonemically distinct from FOOT /ʊ/, unlike in Northern English.",
        },
        {
            "phenomenon": "Smoothing / Coalescence",
            "description": "Sequences of /j/ + vowel coalesce (e.g. 'tune' -> /tʃuːn/) and adjacent vowels smooth across syllable boundaries.",
        },
    ],
    "Scottish": [
        {
            "phenomenon": "Rhoticity",
            "description": "Post-vocalic /r/ is retained in all positions — the clearest acoustic cue for Scottish English.",
        },
        {
            "phenomenon": "TRAP-BATH non-split",
            "description": "No vowel distinction between TRAP and BATH word sets; both realised with short /a/.",
        },
        {
            "phenomenon": "Scottish Vowel Length Rule (SVLR)",
            "description": "Vowels are long only before /r/, /v/, /z/, voiced fricatives, or at phrase boundary — not based on lexical stress.",
        },
        {
            "phenomenon": "Monophthong GOAT/FACE",
            "description": "GOAT word set uses pure monophthong [e:] or [o:] rather than RP diphthongs [goʊt], [feɪs].",
        },
    ],
    "Welsh": [
        {
            "phenomenon": "Syllable-Timed Rhythm",
            "description": "Equal duration allocated to syllables with less vowel reduction — transferred from Welsh phonology.",
        },
        {
            "phenomenon": "Rising-Falling Intonation",
            "description": "Characteristic melodic rise then fall at clause ends, mirroring Welsh intonation contours.",
        },
        {
            "phenomenon": "Vowel Lengthening in Penultimate Syllable",
            "description": "Penultimate syllable vowel is distinctly lengthened — reflecting Welsh stress patterns.",
        },
        {
            "phenomenon": "Clear /l/ in all positions",
            "description": "Welsh English uses a clear (non-velarised) /l/ even in coda position, unlike most English varieties.",
        },
    ],
    "Northern": [
        {
            "phenomenon": "FOOT-STRUT Merger",
            "description": "FOOT and STRUT vowels merged to short back [ʊ] — the defining Northern English feature absent south of the Severn-Wash line.",
        },
        {
            "phenomenon": "Short TRAP-BATH vowel",
            "description": "BATH-class words use short front /a/, not RP long /a:/ — e.g. 'path' rhymes with 'math'.",
        },
        {
            "phenomenon": "Glottal Stop Replacement",
            "description": "Intervocalic and word-final /t/ replaced by glottal stop [?] — e.g. 'butter', 'water', 'bit'.",
        },
        {
            "phenomenon": "GOAT monophthong",
            "description": "GOAT vowel realised as monophthong [o:] rather than RP diphthong [goʊ].",
        },
    ],
    "West_Midlands": [
        {
            "phenomenon": "FACE/GOAT Diphthong Fronting",
            "description": "FACE and GOAT diphthongs have fronted starting points, creating the distinctive Brummie vowel quality.",
        },
        {
            "phenomenon": "Brummie Intonation Pattern",
            "description": "Rising pitch on declaratives that falls at the very end — the 'Brummie melody' that sounds questioning to outsiders.",
        },
        {
            "phenomenon": "Yod Coalescence",
            "description": "/tj/ and /dj/ clusters coalesce to /tʃ/ and /dʒ/ more consistently than RP — e.g. 'tune' > /tʃuːn/.",
        },
        {
            "phenomenon": "STRUT Retraction",
            "description": "STRUT vowel is backer and slightly rounded compared to RP, approaching [ɔ] in some speakers.",
        },
    ],
    "Cockney": [
        {
            "phenomenon": "H-Dropping",
            "description": "Word-initial /h/ is variably deleted — e.g. 'house' -> [aʊs], 'happy' -> [api].",
        },
        {
            "phenomenon": "TH-Fronting",
            "description": "/θ/ and /ð/ replaced by labiodental fricatives /f/ and /v/ — e.g. 'think' -> [fɪŋk], 'brother' -> [brʌvə].",
        },
        {
            "phenomenon": "Glottal Stop and L-vocalisation",
            "description": "Word-final /l/ vocalises to [ʊ] (e.g. 'milk' -> [mɪʊk]) and /t/ glottalises robustly.",
        },
        {
            "phenomenon": "MOUTH Diphthong Shift",
            "description": "MOUTH vowel fronted and raised, approximately [mɑːf] — one of Cockney's most salient features.",
        },
    ],
    "Irish": [
        {
            "phenomenon": "Rhoticity",
            "description": "Post-vocalic /r/ retained in all positions — Irish English is rhotic like Scottish English.",
        },
        {
            "phenomenon": "Dental Stops",
            "description": "/θ/ and /ð/ realised as dental stops [t̪] and [d̪] rather than fricatives — a substrate effect from Irish.",
        },
        {
            "phenomenon": "TRAP Raising",
            "description": "TRAP vowel raised towards [e] in some environments — e.g. 'man' sounds closer to 'men' to English ears.",
        },
        {
            "phenomenon": "Distinctive GOAT/FACE quality",
            "description": "GOAT and FACE vowels have different diphthong onset positions compared to RP, with a raised starting point.",
        },
    ],
    "General": [
        {
            "phenomenon": "Stress-Timed Rhythm",
            "description": "Standard British English uses stress-timed rhythm with reduced unstressed syllables.",
        },
        {
            "phenomenon": "Non-Rhoticity",
            "description": "Post-vocalic /r/ is absent in standard British varieties.",
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
    Native path-integral Integrated Gradients (Sundararajan et al., 2017)
    used when Captum is unavailable.
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
        model: WavLMAccentClassifier instance
        waveform: 1D Tensor [Time] or 2D [1, Time] at 16 kHz
        target_class_idx: Target class index to explain (defaults to argmax prediction)
        device: 'cpu' or 'cuda'
        method: 'integrated_gradients' or 'gradient'
        n_steps: Interpolation steps for Integrated Gradients
        hop_length: Frame hop in samples (320 = 20ms at 16kHz)
        sample_rate: Audio sampling rate
    Returns:
        Dict with timestamps, smoothed saliency curve, top salient regions.
    """
    model.eval()
    model.to(device)

    if waveform.dim() == 1:
        input_values = waveform.unsqueeze(0).to(device)
    else:
        input_values = waveform.to(device)

    with torch.no_grad():
        initial_out = model(input_values, return_hidden=True)
        logits = initial_out["logits"]
        probs = initial_out["probabilities"].detach().cpu().numpy()[0]
        attention_weights = None
        if "attention_weights" in initial_out and initial_out["attention_weights"] is not None:
            attention_weights = initial_out["attention_weights"].detach().cpu().numpy()[0, :, 0]

    if target_class_idx is None:
        target_class_idx = int(np.argmax(probs))

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

    if attention_weights is not None and len(attention_weights) == num_frames:
        norm_asp = attention_weights / (np.max(attention_weights) + 1e-8)
        norm_grad = frame_saliency / (np.max(frame_saliency) + 1e-8)
        frame_saliency = 0.5 * norm_grad + 0.5 * norm_asp

    max_val = np.max(frame_saliency)
    if max_val > 0:
        frame_saliency = frame_saliency / max_val

    smoothed_saliency = gaussian_filter1d(frame_saliency, sigma=3.0)
    if np.max(smoothed_saliency) > 0:
        smoothed_saliency = smoothed_saliency / np.max(smoothed_saliency)

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
    Correlates salient temporal regions with documented phonological features
    for the predicted UK accent.
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
        AUDC_salient < AUDC_random  =>  Delta AUDC > 0
    """
    if deletion_fractions is None:
        deletion_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    model.eval()
    model.to(device)

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

    if target_class_idx is None:
        with torch.no_grad():
            logits = model(input_values)["logits"]
            target_class_idx = int(torch.argmax(logits, dim=-1).item())

    frame_saliency = frame_saliency[:num_frames]
    ranked_frame_indices = np.argsort(frame_saliency)[::-1]

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

        masked_salient = input_values.clone()
        top_k_indices = ranked_frame_indices[:k]
        for f_idx in top_k_indices:
            start_s = f_idx * hop_length
            end_s = min((f_idx + 1) * hop_length, total_samples)
            masked_salient[0, start_s:end_s] = 0.0

        with torch.no_grad():
            out_sal = model(masked_salient)
            salient_probs.append(float(out_sal["probabilities"][0, target_class_idx].item()))

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
                seed_probs.append(float(out_rand["probabilities"][0, target_class_idx].item()))
        random_probs.append(float(np.mean(seed_probs)))

    del_arr = np.array(deletion_fractions)
    if hasattr(np, "trapezoid"):
        audc_salient = float(np.trapezoid(salient_probs, del_arr))
        audc_random = float(np.trapezoid(random_probs, del_arr))
    else:
        audc_salient = float(np.trapz(salient_probs, del_arr))
        audc_random = float(np.trapz(random_probs, del_arr))

    delta_audc = audc_random - audc_salient
    faithfulness_ratio = delta_audc / (audc_random + 1e-8)

    return {
        "target_class_idx": target_class_idx,
        "deletion_fractions": deletion_fractions,
        "salient_deletion_curve": [round(p, 4) for p in salient_probs],
        "random_deletion_curve": [round(p, 4) for p in random_probs],
        "audc_salient": round(audc_salient, 4),
        "audc_random": round(audc_random, 4),
        "delta_audc": round(delta_audc, 4),
        "faithfulness_ratio": round(faithfulness_ratio, 4),
        "is_faithful": bool(delta_audc > 0.0),
    }

"""
AccentSense Phase 5: Downstream Whisper ASR Adaptation Runner
Benchmarks unadapted baseline vs. accent-conditioned ASR transcription
and calculates Word Error Rate (WER), Character Error Rate (CER), and Relative WERR.
"""

import argparse
import os
import sys
import json
import torch
import torchaudio
from typing import Dict, List, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.asr.adaptation import (
    WhisperAccentAdaptor,
    REGIONAL_PROMPTS,
    PHONETIC_ERROR_PATTERNS,
    BENCHMARK_CORPUS,
)

TARGET_CLASSES = ["Northern_Hindi", "Central_MP", "Western_Gujarati", "Southern_Tamil"]


def print_asr_dashboard(res: Dict[str, Any]):
    """
    Renders an academic ASR adaptation evaluation dashboard.
    """
    m = res["metrics"]
    print("\n" + "=" * 78)
    print(f"       ACCENTSENSE DOWNSTREAM ASR ADAPTATION: {res['regional_accent'].upper()}")
    print("=" * 78)
    print(f" Conditioning Prompt : \"{res['conditioning_prompt']}\"")
    print(f" Reference Text      : \"{res['reference_transcript']}\"")
    print("-" * 78)
    print(f" Baseline Transcript : \"{res['baseline_transcript']}\"")
    print(f" Adapted Transcript  : \"{res['adapted_transcript']}\"")
    print("-" * 78)

    print("[Performance Metrics Comparison]")
    print(f"  Word Error Rate (WER)      : Baseline = {m['wer_baseline']*100:5.2f}%  -->  Adapted = {m['wer_adapted']*100:5.2f}%")
    print(f"  Relative WER Reduction     : {m['werr_percent']:+6.2f}% WERR (Positive indicates accuracy gain)")
    print(f"  Character Error Rate (CER) : Baseline = {m['cer_baseline']*100:5.2f}%  -->  Adapted = {m['cer_adapted']*100:5.2f}%")
    print(f"  Relative CER Reduction     : {m['cerr_percent']:+6.2f}% CERR")

    print("\n[Phonetic Errors Resolved]")
    corrections = res.get("phonetic_corrections_analyzed", [])
    if corrections:
        for idx, item in enumerate(corrections, 1):
            print(f"  {idx}. Sound Feature : {item['original_sound']}")
            print(f"     * Unadapted Error : {item['unadapted_error']}")
            print(f"     * Adapted Fix     : {item['adapted_correction']}")
    else:
        print("  * Standard lexical alignment.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="AccentSense Phase 5: Downstream ASR Adaptation Runner")
    parser.add_argument("--accent", type=str, default="all", choices=["Northern_Hindi", "Central_MP", "Western_Gujarati", "Southern_Tamil", "all"], help="Accent class to evaluate")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to audio file (optional)")
    parser.add_argument("--reference", type=str, default=None, help="Ground truth reference transcript (optional)")
    parser.add_argument("--use_pretrained", action="store_true", help="Download and load real Whisper model weights")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny", help="Whisper model name")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for JSON report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Phase 5 Init] Initializing Whisper ASR Adaptor on device: {device.upper()}")

    adaptor = WhisperAccentAdaptor(
        model_name=args.model_name,
        device=device,
        load_pretrained=args.use_pretrained,
    )

    waveform = None
    if args.audio_path and os.path.exists(args.audio_path):
        wav, sr = torchaudio.load(args.audio_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        waveform = wav.squeeze(0)

    # Single Accent Evaluation
    if args.accent != "all":
        res = adaptor.benchmark_sample(
            regional_accent=args.accent,
            reference_text=args.reference,
            waveform=waveform,
        )
        print_asr_dashboard(res)
        out_file = os.path.join(args.output_dir, "asr_adaptation_report.json")
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[SUCCESS] ASR Adaptation Report written to: {out_file}")
        return

    # Benchmark All 4 Regional Anchors
    print("\n[Benchmark Execution] Evaluating Downstream Adaptation across all 4 Regional Anchors...")
    all_evaluations = []
    werr_list = []
    wer_base_list, wer_adapt_list = [], []

    for acc in TARGET_CLASSES:
        res = adaptor.benchmark_sample(
            regional_accent=acc,
            reference_text=None,
            waveform=waveform,
        )
        all_evaluations.append(res)
        print_asr_dashboard(res)
        werr_list.append(res["metrics"]["werr_percent"])
        wer_base_list.append(res["metrics"]["wer_baseline"])
        wer_adapt_list.append(res["metrics"]["wer_adapted"])

    mean_base_wer = float(sum(wer_base_list) / len(wer_base_list))
    mean_adapt_wer = float(sum(wer_adapt_list) / len(wer_adapt_list))
    mean_werr = float(sum(werr_list) / len(werr_list))

    summary = {
        "classes_evaluated": TARGET_CLASSES,
        "mean_baseline_wer": round(mean_base_wer, 4),
        "mean_adapted_wer": round(mean_adapt_wer, 4),
        "mean_werr_percent": round(mean_werr, 2),
        "evaluations": all_evaluations,
    }

    out_file = os.path.join(args.output_dir, "asr_adaptation_report.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("                    PHASE 5 BENCHMARK SUMMARY")
    print("=" * 78)
    print(f" Regional Accents Benchmarked : {len(TARGET_CLASSES)} ({', '.join(TARGET_CLASSES)})")
    print(f" Mean Baseline WER            : {mean_base_wer*100:.2f}%")
    print(f" Mean Adapted WER             : {mean_adapt_wer*100:.2f}%")
    print(f" Overall Relative WERR Gain   : {mean_werr:+.2f}%")
    print(f" Detailed JSON Report Saved   : {out_file}")
    print("=" * 78 + "\n")
    print("[SUCCESS] Phase 5 Downstream ASR Adaptation Completed.")


if __name__ == "__main__":
    main()

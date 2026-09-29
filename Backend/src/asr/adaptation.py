"""
Downstream ASR adaptation *illustration*: static benchmark data plus the
Whisper prompt-conditioning templates that would be used.

IMPORTANT — honest semantics: no speech-recognition model is executed in
this module. The transcripts below are fixed benchmark examples shipped
with the repository, and the WER/CER figures are computed over those static
strings only. The endpoint built on top of this
(``POST /api/downstream-asr``) labels itself as a static illustration.

Class keys match ``src.config.CLASSES`` (minus "Not a speech", which has no
accent profile): Irish, Midlands, Northern, Scottish, Southern, Welsh.
"""

from __future__ import annotations

from typing import Any

import jiwer

# Whisper prompt conditioning templates per accent class.
REGIONAL_PROMPTS = {
    "Irish": "The following is Irish English (Hiberno-English) with rhotic r, dental stops for th-sounds, and distinctive GOAT and FACE vowels.",
    "Midlands": "The following is Midlands English (including Brummie) with fronted FACE and GOAT diphthongs and the characteristic Birmingham intonation.",
    "Northern": "The following is Northern English with the FOOT-STRUT merger, short BATH vowels, and glottal stop replacement of intervocalic t.",
    "Scottish": "The following is Scottish English with rhotic r, monophthong vowels, and the Scottish Vowel Length Rule.",
    "Southern": "The following is Southern English (standard Southern / RP) with non-rhotic pronunciation, long BATH vowels, and intrusive linking r.",
    "Welsh": "The following is Welsh English with syllable-timed rhythm, rising-falling intonation, and lengthened penultimate vowels.",
    "General": "The following is British English speech.",
}

# Documented (literature-level) ASR error patterns per accent class.
# These are illustrative phonetic notes, NOT measured error rates from this
# repository — no Whisper model is run here.
PHONETIC_ERROR_PATTERNS = {
    "Irish": [
        {
            "original_sound": "Dental stop /t/ for /th/",
            "unadapted_error": "Dental stop realisation of /th/ transcribed as /t/ ('the' -> 'de', 'this' -> 'dis', 'think' -> 'tink').",
            "adapted_correction": "Dental stop correctly mapped to th-orthography.",
        },
        {
            "original_sound": "Rhotic post-vocalic /r/",
            "unadapted_error": "Coda /r/ causes vowel quality confusion similar to Scottish errors.",
            "adapted_correction": "Rhotic coda mapped to standard spelling.",
        },
    ],
    "Midlands": [
        {
            "original_sound": "Fronted FACE/GOAT diphthongs",
            "unadapted_error": "Fronted onset of FACE diphthong transcribed as DRESS vowel ('late' -> 'let', 'make' -> 'mek').",
            "adapted_correction": "Fronted diphthong onset mapped to correct FACE lexical set.",
        },
        {
            "original_sound": "Birmingham rising-falling intonation",
            "unadapted_error": "Declarative sentences with rising-falling pitch transcribed with inserted question marks.",
            "adapted_correction": "Rising-falling intonation correctly parsed as declarative.",
        },
    ],
    "Northern": [
        {
            "original_sound": "FOOT-STRUT merger [U]",
            "unadapted_error": "STRUT words transcribed with FOOT spelling (e.g. 'cup' -> 'cop', 'bus' -> 'boos').",
            "adapted_correction": "Merged vowel resolved to correct STRUT-class orthography.",
        },
        {
            "original_sound": "Glottal stop /t/ replacement",
            "unadapted_error": "Glottal stop between vowels causes word boundary errors ('butter' -> 'bu er', 'water' -> 'wa er').",
            "adapted_correction": "Glottal stop correctly resolved as intervocalic /t/.",
        },
    ],
    "Scottish": [
        {
            "original_sound": "Rhotic post-vocalic /r/",
            "unadapted_error": "Rhotic /r/ in coda position causes vowel mis-identification (e.g. 'bird' -> 'burred', 'word' -> 'wurred').",
            "adapted_correction": "Rhotic coda correctly mapped to standard English spelling.",
        },
        {
            "original_sound": "SVLR vowel duration",
            "unadapted_error": "Short vowels before voiceless stops mis-transcribed as reduced syllables.",
            "adapted_correction": "Vowel length normalised to lexical target.",
        },
    ],
    "Southern": [
        {
            "original_sound": "Intrusive /r/ linking",
            "unadapted_error": "Linking /r/ between vowels transcribed as a separate word (e.g. 'idea of' -> 'idea r of').",
            "adapted_correction": "Linking /r/ correctly suppressed in transcription.",
        },
    ],
    "Welsh": [
        {
            "original_sound": "Syllable-timed equal duration",
            "unadapted_error": "Unstressed syllables kept full duration — ASR inserts extra words or syllable boundaries.",
            "adapted_correction": "Syllable boundaries aligned to lexical rather than duration-based segmentation.",
        },
        {
            "original_sound": "Lengthened penultimate vowel",
            "unadapted_error": "Long penultimate vowel parsed as vowel + word boundary ('agenda' -> 'agen da').",
            "adapted_correction": "Penultimate lengthening suppressed in word segmentation.",
        },
    ],
}

# Static benchmark corpus: sentences chosen to trigger accent-specific
# phonological contrasts. baseline/adapted hypotheses are FIXED EXAMPLES —
# they are not produced by running an ASR model.
BENCHMARK_CORPUS = {
    "Irish": {
        "reference": "this is the thirty third day since the weather turned colder",
        "baseline_hypothesis": "dis is de thirty tird day since de wedder turned colder",
        "adapted_hypothesis": "this is the thirty third day since the weather turned colder",
    },
    "Midlands": {
        "reference": "make the cake later and take it to the gate before eight",
        "baseline_hypothesis": "mek the cek later and tek it to the get before et",
        "adapted_hypothesis": "make the cake later and take it to the gate before eight",
    },
    "Northern": {
        "reference": "put the butter and the cup of water on the table",
        "baseline_hypothesis": "put the bu er and the cop of wa er on the table",
        "adapted_hypothesis": "put the butter and the cup of water on the table",
    },
    "Scottish": {
        "reference": "the bird perched on the word carved above the door of the church",
        "baseline_hypothesis": "the burred perched on the wurred carved above the door of the church",
        "adapted_hypothesis": "the bird perched on the word carved above the door of the church",
    },
    "Southern": {
        "reference": "the path through the grass led past the dance hall to the bath",
        "baseline_hypothesis": "the path through the grass led past the dance hall to the bath",
        "adapted_hypothesis": "the path through the grass led past the dance hall to the bath",
    },
    "Welsh": {
        "reference": "the agenda for the meeting covers the agenda items in careful order",
        "baseline_hypothesis": "the agen da for the meeting covers the agen da items in careful order",
        "adapted_hypothesis": "the agenda for the meeting covers the agenda items in careful order",
    },
}


class WhisperAccentAdaptor:
    """Static prompt-conditioning benchmark helper (no ASR model inside).

    The constructor takes no model arguments on purpose: this class never
    downloads or executes a speech-recognition network. ``transcribe()`` and
    the live-Whisper path were removed together with the torch stack they
    depended on (see docs/ARCHITECTURE.md).
    """

    def get_prompt(self, regional_accent: str) -> str:
        return REGIONAL_PROMPTS.get(regional_accent, REGIONAL_PROMPTS["General"])

    def evaluate_transcriptions(
        self,
        reference: str,
        baseline: str,
        adapted: str,
    ) -> dict[str, float]:
        """WER/CER over the given strings (real computation, static inputs)."""
        ref_norm = reference.lower().strip()
        base_norm = baseline.lower().strip()
        adapt_norm = adapted.lower().strip()

        wer_baseline = float(jiwer.wer(ref_norm, base_norm))
        wer_adapted = float(jiwer.wer(ref_norm, adapt_norm))
        cer_baseline = float(jiwer.cer(ref_norm, base_norm))
        cer_adapted = float(jiwer.cer(ref_norm, adapt_norm))

        werr = ((wer_baseline - wer_adapted) / wer_baseline * 100.0) if wer_baseline > 0 else 0.0
        cerr = ((cer_baseline - cer_adapted) / cer_baseline * 100.0) if cer_baseline > 0 else 0.0

        return {
            "wer_baseline": round(wer_baseline, 4),
            "wer_adapted": round(wer_adapted, 4),
            "werr_percent": round(werr, 2),
            "cer_baseline": round(cer_baseline, 4),
            "cer_adapted": round(cer_adapted, 4),
            "cerr_percent": round(cerr, 2),
        }

    def benchmark_sample(
        self,
        regional_accent: str,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        """Return the static benchmark entry for one accent class.

        Unknown accents fall back to the Northern entry (the caller is
        expected to validate against ``src.config.CLASSES`` first).
        """
        accent_key = regional_accent if regional_accent in BENCHMARK_CORPUS else "Northern"
        prompt = self.get_prompt(accent_key)

        if reference_text is None:
            reference_text = BENCHMARK_CORPUS[accent_key]["reference"]

        baseline_transcript = BENCHMARK_CORPUS[accent_key]["baseline_hypothesis"]
        adapted_transcript = BENCHMARK_CORPUS[accent_key]["adapted_hypothesis"]

        metrics = self.evaluate_transcriptions(
            reference=reference_text,
            baseline=baseline_transcript,
            adapted=adapted_transcript,
        )
        corrections = PHONETIC_ERROR_PATTERNS.get(accent_key, [])

        return {
            "regional_accent": accent_key,
            "conditioning_prompt": prompt,
            "reference_transcript": reference_text,
            "baseline_transcript": baseline_transcript,
            "adapted_transcript": adapted_transcript,
            "metrics": metrics,
            "phonetic_corrections_analyzed": corrections,
        }

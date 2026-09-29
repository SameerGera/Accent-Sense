/** Types mirroring Backend Pydantic models in src/api/main.py */

export interface EvidenceRegion {
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  /** Mean per-frame model score for the predicted class in this region. */
  model_score: number;
  label: string;
  detail: string;
}

export interface PredictionResponse {
  predicted_influence: string;
  language_family: string;
  /** Model score for the top class — not a calibrated confidence. */
  model_score: number;
  all_scores: Record<string, number>;
  evidence_regions: EvidenceRegion[];
  timestamps: number[];
  evidence_curve: number[];
  /** Fraction of frames where YAMNet's top event was Speech. */
  speech_frame_ratio: number;
  /** false → predicted class is "Not a speech" (insufficient speech). */
  is_sufficient_speech: boolean;
  /** Honest interpretation of model_score, served by the backend. */
  model_score_note: string;
}

export interface DownstreamASRResponse {
  baseline_transcript: string;
  accent_adapted_transcript: string;
  adaptation_strategy: string;
  detected_accent_profile: string;
  phonetic_corrections_noted: string[];
  /** Always true today: no ASR model is executed for this request. */
  is_static_example: boolean;
  disclaimer: string;
}

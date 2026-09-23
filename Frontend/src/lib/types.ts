/** Types mirroring Backend Pydantic models in src/api/main.py */

export interface SalientRegion {
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  salience_score: number;
  linguistic_phenomenon: string;
  phonetic_explanation: string;
}

export interface PredictionResponse {
  is_mock_prototype: boolean;
  predicted_influence: string;
  language_family: string;
  confidence: number;
  all_scores: Record<string, number>;
  salient_regions: SalientRegion[];
  timestamps: number[];
  saliency_curve: number[];
}

export interface DownstreamASRResponse {
  baseline_transcript: string;
  accent_adapted_transcript: string;
  adaptation_strategy: string;
  detected_accent_profile: string;
  phonetic_corrections_noted: string[];
}

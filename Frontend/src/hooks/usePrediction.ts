import { useCallback, useState } from 'react';
import { ApiError, fetchASR, fetchPredict } from '../lib/api';
import type { DownstreamASRResponse, PredictionResponse } from '../lib/types';

export type PredictionStatus = 'idle' | 'loading' | 'success' | 'error';

export interface PredictionApi {
  status: PredictionStatus;
  prediction: PredictionResponse | null;
  error: string | null;
  /** 0..100 progress for the hairline bar */
  progress: number;
  analyze: (file: File) => Promise<void>;
  reset: () => void;
}

/**
 * Prediction state machine: idle → loading → success | error.
 * Progress bar advances during the fetch and completes on response.
 */
export function usePrediction(): PredictionApi {
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const analyze = useCallback(async (file: File) => {
    setStatus('loading');
    setPrediction(null);
    setError(null);
    setProgress(0);

    // Single state update — CSS transition handles the smooth animation
    setProgress(85);

    try {
      const result = await fetchPredict(file);
      setProgress(100);
      setPrediction(result);
      setStatus('success');
      // Brief hold at 100% so the completion is visible
      window.setTimeout(() => setProgress(0), 600);
    } catch (e) {
      setProgress(0);
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : 'Something went wrong';
      setError(msg);
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setPrediction(null);
    setError(null);
    setProgress(0);
  }, []);

  return { status, prediction, error, progress, analyze, reset };
}

export interface ASRApi {
  data: DownstreamASRResponse | null;
  loading: boolean;
  error: string | null;
  load: (accent: string) => Promise<void>;
  clear: () => void;
}

/** Lazy ASR comparison loader — call with the predicted accent. */
export function useASR(): ASRApi {
  const [data, setData] = useState<DownstreamASRResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (accent: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchASR(accent);
      setData(result);
    } catch (e) {
      setData(null);
      setError(
        e instanceof ApiError || e instanceof Error
          ? e.message
          : 'ASR service unavailable',
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, load, clear };
}

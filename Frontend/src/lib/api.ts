import type { DownstreamASRResponse, PredictionResponse } from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function fetchPredict(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', file);

  let res: Response;
  try {
    res = await fetch('/api/predict', { method: 'POST', body: form });
  } catch {
    throw new ApiError('Cannot reach the analysis server. Is the backend running?');
  }

  if (!res.ok) {
    let detail = `Server error (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(detail, res.status);
  }

  return (await res.json()) as PredictionResponse;
}

export async function fetchASR(
  detectedAccent: string,
  referenceText?: string,
): Promise<DownstreamASRResponse> {
  const form = new FormData();
  form.append('detected_accent', detectedAccent);
  if (referenceText) form.append('reference_text', referenceText);

  let res: Response;
  try {
    res = await fetch('/api/downstream-asr', { method: 'POST', body: form });
  } catch {
    throw new ApiError('Cannot reach the analysis server. Is the backend running?');
  }

  if (!res.ok) {
    let detail = `Server error (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(detail, res.status);
  }

  return (await res.json()) as DownstreamASRResponse;
}

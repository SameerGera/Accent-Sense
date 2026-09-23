import { useCallback, useEffect, useRef, useState } from 'react';

export interface WaveformData {
  /** Peak amplitudes 0..1 (length ≈ barCount) */
  peaks: number[] | null;
  loading: boolean;
  error: boolean;
  /** Re-extract peaks (e.g. new file) */
  load: (url: string) => void;
  clear: () => void;
}

const BAR_COUNT = 120;

/**
 * Decodes audio via Web Audio API and extracts peak amplitudes
 * for canvas waveform rendering. Falls back to a procedural
 * envelope when decode fails (some webm/mp3 encodings).
 */
export function useWaveform(): WaveformData {
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const ctxRef = useRef<AudioContext | null>(null);
  const abortRef = useRef(0);

  const getContext = useCallback((): AudioContext => {
    if (!ctxRef.current) {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      ctxRef.current = new Ctx();
    }
    return ctxRef.current;
  }, []);

  const proceduralFallback = useCallback((): number[] => {
    let s = 7;
    const rand = () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
    return Array.from({ length: BAR_COUNT }, (_, i) => {
      const envelope =
        Math.sin((i / BAR_COUNT) * Math.PI * 3.5) * 0.4 + 0.5;
      const noise = rand() * 0.5 + 0.15;
      return Math.min(1, Math.max(0.1, envelope * noise));
    });
  }, []);

  const extractPeaks = useCallback(
    async (buffer: AudioBuffer): Promise<number[]> => {
      const data = buffer.getChannelData(0);
      const blockSize = Math.max(1, Math.floor(data.length / BAR_COUNT));
      const result: number[] = [];
      for (let i = 0; i < BAR_COUNT; i++) {
        const start = i * blockSize;
        const end = Math.min(start + blockSize, data.length);
        let max = 0;
        for (let j = start; j < end; j++) {
          const v = Math.abs(data[j]);
          if (v > max) max = v;
        }
        result.push(Math.min(1, max));
      }
      // Normalize to use full height
      const peak = Math.max(...result, 0.01);
      return result.map((v) => Math.max(0.06, v / peak));
    },
    [],
  );

  const load = useCallback(
    (url: string) => {
      const token = ++abortRef.current;
      setLoading(true);
      setError(false);
      setPeaks(null);

      void (async () => {
        try {
          const res = await fetch(url);
          const bytes = await res.arrayBuffer();
          if (token !== abortRef.current) return;

          const ctx = getContext();
          const buffer = await ctx.decodeAudioData(bytes);
          if (token !== abortRef.current) return;

          const extracted = await extractPeaks(buffer);
          if (token !== abortRef.current) return;

          setPeaks(extracted);
          setLoading(false);
        } catch {
          if (token !== abortRef.current) return;
          // Fallback: procedural envelope so the UI still works
          setPeaks(proceduralFallback());
          setError(true);
          setLoading(false);
        }
      })();
    },
    [getContext, extractPeaks, proceduralFallback],
  );

  const clear = useCallback(() => {
    abortRef.current++;
    setPeaks(null);
    setLoading(false);
    setError(false);
  }, []);

  // Close AudioContext on unmount
  useEffect(() => {
    const abort = abortRef;
    const ctx = ctxRef;
    return () => {
      abort.current++;
      void ctx.current?.close();
      ctx.current = null;
    };
  }, []);

  return { peaks, loading, error, load, clear };
}

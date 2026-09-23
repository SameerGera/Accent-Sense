import { useCallback, useEffect, useRef } from 'react';
import { useTheme } from '../hooks/useTheme';

export interface WaveformCanvasProps {
  /** Peak amplitudes 0..1; null → empty stage */
  peaks: number[] | null;
  /** Played ratio 0..1 */
  progress: number;
  /** Optional saliency attribution to overlay after prediction */
  saliency?: { timestamps: number[]; curve: number[]; duration: number } | null;
  onSeek: (ratio: number) => void;
  /** Current time / duration for keyboard seek + a11y */
  currentTime: number;
  duration: number;
}

/**
 * DPR-aware canvas waveform. Redraws on peaks, progress,
 * theme change, and debounced resize.
 */
export function WaveformCanvas({
  peaks,
  progress,
  saliency,
  onSeek,
  currentTime,
  duration,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const rect = wrap.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const w = rect.width;
    const h = rect.height;
    const isDark = theme === 'dark';
    const unplayed = isDark ? '#57423b' : '#8a726a';
    const played = '#b85028';

    if (!peaks || peaks.length === 0) {
      // Empty stage: faint baseline
      ctx.fillStyle = isDark ? '#2e2b27' : '#e5dfd5';
      ctx.fillRect(0, h / 2, w, 1);
      return;
    }

    const n = peaks.length;
    const slot = w / n;
    const barWidth = Math.max(2, slot - 1.5);

    for (let i = 0; i < n; i++) {
      const frac = i / n;
      const barHeight = Math.max(2, peaks[i] * (h * 0.76));
      const x = i * slot;
      const y = (h - barHeight) / 2;
      ctx.fillStyle = frac <= progress ? played : unplayed;
      ctx.fillRect(x, y, barWidth, barHeight);
    }

    // Saliency overlay: thin curve along the bottom
    if (saliency && saliency.timestamps.length > 1 && duration > 0) {
      const { timestamps, curve } = saliency;
      ctx.beginPath();
      ctx.strokeStyle = isDark ? 'rgba(255,181,156,0.55)' : 'rgba(184,80,40,0.45)';
      ctx.lineWidth = 1;
      const maxT = timestamps[timestamps.length - 1] || duration;
      for (let i = 0; i < timestamps.length; i++) {
        const x = (timestamps[i] / maxT) * w;
        const y = h - 3 - curve[i] * (h * 0.18);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [peaks, progress, saliency, theme, duration]);

  // Redraw on relevant changes
  useEffect(() => {
    draw();
  }, [draw]);

  // Debounced resize
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    let timer = 0;
    const ro = new ResizeObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(draw, 80);
    });
    ro.observe(wrap);
    return () => {
      window.clearTimeout(timer);
      ro.disconnect();
    };
  }, [draw]);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const wrap = wrapRef.current;
    if (!wrap || duration <= 0) return;
    const rect = wrap.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(ratio);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (duration <= 0) return;
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      onSeek(Math.min(1, (currentTime + 1) / duration));
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      onSeek(Math.max(0, (currentTime - 1) / duration));
    } else if (e.key === 'Home') {
      e.preventDefault();
      onSeek(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      onSeek(1);
    }
  };

  const playheadPct = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;

  return (
    <div
      ref={wrapRef}
      className="relative h-24 w-full overflow-hidden cursor-pointer select-none border border-outline-variant rounded"
      style={{
        background: 'var(--waveform-bg)',
      }}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="slider"
      tabIndex={0}
      aria-label="Audio waveform scrubber"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(currentTime)}
      aria-valuetext={`${Math.round(currentTime)} seconds of ${Math.round(duration)}`}
    >
      <canvas ref={canvasRef} className="block w-full h-full" />
      {/* Playhead: 1.5px line + 6px square cap (Stitch spec) */}
      {duration > 0 && (
        <div
          className="absolute top-0 bottom-0 pointer-events-none"
          style={{
            left: `${playheadPct}%`,
            width: 1.5,
            background: 'var(--playhead)',
            transition: 'left 75ms linear',
          }}
        >
          <div
            className="absolute -top-px -left-[2px] w-[6px] h-[6px] bg-primary-container"
          />
        </div>
      )}
    </div>
  );
}

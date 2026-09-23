import { motion } from 'framer-motion';
import type { AudioPlayerApi } from '../hooks/useAudioPlayer';
import { formatTime } from '../lib/format';
import { WaveformCanvas } from './WaveformCanvas';

export interface AudioPlayerProps {
  player: AudioPlayerApi;
  peaks: number[] | null;
  fileName: string;
  saliency?: { timestamps: number[]; curve: number[]; duration: number } | null;
  onAnalyze: () => void;
  analyzing: boolean;
  canAnalyze: boolean;
}

export function AudioPlayer({
  player,
  peaks,
  fileName,
  saliency,
  onAnalyze,
  analyzing,
  canAnalyze,
}: AudioPlayerProps) {
  const { playing, currentTime, duration, loop, toggleLoop, toggle } = player;

  return (
    <motion.div
      className="w-full p-space-lg space-y-space-md bg-surface-container border border-outline-variant rounded"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Top row: play + title | timecode */}
      <div className="flex items-center justify-between gap-space-md">
        <div className="flex items-center space-x-space-md min-w-0">
          <motion.button
            type="button"
            onClick={toggle}
            aria-label={playing ? 'Pause audio' : 'Play audio'}
            className="w-10 h-10 rounded-full flex items-center justify-center shrink-0 border border-outline-variant text-on-surface bg-transparent"
            whileHover={{
              scale: 1.06,
              borderColor: 'var(--primary-container)',
              color: 'var(--primary-container)',
            }}
            whileTap={{ scale: 0.94 }}
            transition={{ type: 'spring', stiffness: 500, damping: 28 }}
            animate={playing ? { boxShadow: '0 0 0 4px rgba(184, 80, 40, 0.15)' } : { boxShadow: '0 0 0 0px rgba(184, 80, 40, 0)' }}
          >
            <span className="material-symbols-outlined !text-[20px]" aria-hidden="true">
              {playing ? 'pause' : 'play_arrow'}
            </span>
          </motion.button>
          <div className="min-w-0">
            <div className="font-body-md text-body-md font-medium truncate text-on-surface">
              {fileName}
            </div>
            <div className="font-label-code text-label-code text-on-surface-variant">
              Acoustic sample
            </div>
          </div>
        </div>
        <div
          className="font-label-code text-label-code tabular shrink-0 text-on-surface"
          aria-live="off"
        >
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>

      {/* Waveform */}
      <WaveformCanvas
        peaks={peaks}
        progress={duration > 0 ? currentTime / duration : 0}
        saliency={saliency}
        onSeek={player.seekRatio}
        currentTime={currentTime}
        duration={duration}
      />

      {/* Footer: loop | Analyze CTA — even split, matches drop-zone width */}
      <div className="flex flex-col-reverse sm:flex-row sm:items-center justify-between gap-space-sm pt-space-xs">
        <motion.button
          type="button"
          onClick={toggleLoop}
          aria-pressed={loop}
          className={`btn-tertiary !justify-start ${loop ? 'text-primary-container' : 'text-on-surface-variant'}`}
          whileTap={{ scale: 0.97 }}
        >
          <motion.span
            className="material-symbols-outlined !text-[16px]"
            aria-hidden="true"
            animate={loop ? { rotate: [0, -15, 15, 0] } : { rotate: 0 }}
            transition={{ duration: 0.4 }}
          >
            repeat
          </motion.span>
          <span>Continuous loop</span>
        </motion.button>

        <motion.button
          type="button"
          onClick={onAnalyze}
          disabled={!canAnalyze || analyzing}
          className="btn-primary group"
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.97 }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        >
          <span>{analyzing ? 'Analysing…' : 'Analyze speech'}</span>
          {!analyzing && (
            <motion.span
              className="material-symbols-outlined !text-[18px]"
              aria-hidden="true"
              animate={undefined}
            >
              arrow_forward
            </motion.span>
          )}
          {analyzing && <span className="spinner !w-4 !h-4 !border-white/40 !border-t-white" />}
        </motion.button>
      </div>
    </motion.div>
  );
}

import { motion } from 'framer-motion';
import type { ASRApi } from '../hooks/usePrediction';
import { accentDisplayName } from '../lib/format';

export interface ASRComparisonProps {
  asr: ASRApi;
  accent: string;
  onRetry: () => void;
}

/** Downstream Whisper comparison: baseline vs accent-prompted transcript. */
export function ASRComparison({ asr, accent, onRetry }: ASRComparisonProps) {
  const { data, loading, error } = asr;

  return (
    <motion.section
      className="space-y-space-md no-print"
      aria-label="Speech recognition comparison"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="flex items-center justify-between">
        <h3 className="font-sans text-headline-sm text-on-surface">
          Speech recognition
        </h3>
        <span className="font-label-code text-label-code text-on-surface-variant">
          Whisper · prompt conditioning
        </span>
      </div>

      {loading && (
        <motion.div
          className="p-space-lg space-y-space-sm bg-surface-container-low border border-outline-variant rounded"
          aria-busy="true"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-space-sm">
            <div className="spinner !w-4 !h-4" />
            <span className="font-body-sm text-body-sm text-on-surface-variant">
              Running recognition comparison for {accentDisplayName(accent)}…
            </span>
          </div>
          <div className="progress-track">
            <motion.div
              className="progress-fill"
              initial={{ width: '0%' }}
              animate={{ width: '70%' }}
              transition={{ duration: 1.2, ease: 'easeOut' }}
              style={{ opacity: 0.6 }}
            />
          </div>
        </motion.div>
      )}

      {error && !loading && (
        <motion.div
          className="p-space-md flex items-center justify-between gap-space-md bg-error-container border border-error rounded text-on-error-container"
          role="alert"
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <span className="font-body-sm text-body-sm">{error}</span>
          <button type="button" onClick={onRetry} className="btn-secondary !h-[28px] !text-label-code">
            Retry
          </button>
        </motion.div>
      )}

      {data && !loading && (
        <motion.div
          className="p-space-lg space-y-space-md bg-surface-container-low border border-outline-variant rounded"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-space-lg">
            <motion.div
              className="space-y-space-xs"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1, duration: 0.4 }}
            >
              <div className="kicker">Baseline transcript</div>
              <p className="font-body-md text-body-md text-on-surface-variant">
                {data.baseline_transcript}
              </p>
            </motion.div>
            <motion.div
              className="space-y-space-xs"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.4 }}
            >
              <div className="kicker text-primary-container">
                Accent-adapted transcript
              </div>
              <p className="font-body-md text-body-md text-on-surface">
                {data.accent_adapted_transcript}
              </p>
            </motion.div>
          </div>

          <div
            className="pt-space-sm space-y-space-xs border-t border-surface-container-highest"
          >
            <div className="font-label-code text-label-code text-on-surface-variant">
              {data.adaptation_strategy}
            </div>
            <div className="font-label-code text-label-code text-on-surface-variant">
              Profile: {data.detected_accent_profile}
            </div>
          </div>

          {data.phonetic_corrections_noted.length > 0 && (
            <motion.div
              className="space-y-space-xs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35 }}
            >
              <div className="kicker">Phonetic corrections</div>
              <ul className="space-y-1">
                {data.phonetic_corrections_noted.map((c, i) => (
                  <motion.li
                    key={i}
                    className="font-body-sm text-body-sm text-on-surface-variant"
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + i * 0.06 }}
                  >
                    {c}
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          )}
        </motion.div>
      )}
    </motion.section>
  );
}

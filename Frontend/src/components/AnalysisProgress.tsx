import { motion } from 'framer-motion';

export interface AnalysisProgressProps {
  progress: number;
}

export function AnalysisProgress({ progress }: AnalysisProgressProps) {
  const pct = Math.min(100, Math.round(progress));
  return (
    <motion.section
      className="py-space-xl border-t border-b border-outline-variant"
      aria-live="polite"
      aria-busy="true"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="text-center space-y-space-md">
        <div className="flex justify-center">
          <motion.span
            className="material-symbols-outlined !text-[32px] text-primary-container"
            aria-hidden="true"
            animate={{ rotate: [0, 10, -10, 0], scale: [1, 1.08, 1] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
          >
            model_training
          </motion.span>
        </div>
        <div className="space-y-space-xs">
          <h3 className="font-sans text-headline-sm text-on-surface">
            Analysing acoustic features
          </h3>
          <p className="font-body-md text-body-md text-on-surface-variant">
            Comparing rhythm, vowels, and consonants against regional accent patterns…
          </p>
        </div>
        <div
          className="progress-track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label="Analysis progress"
        >
          <div
            className="progress-fill bg-primary"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="font-label-code text-label-code tabular text-on-surface-variant">
          {pct}%
        </div>
      </div>
    </motion.section>
  );
}

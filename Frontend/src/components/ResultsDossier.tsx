import { motion } from 'framer-motion';
import { listItem, stagger } from '../lib/motion';
import type { PredictionResponse } from '../lib/types';
import { accentDisplayName, formatPercent, sortedScores } from '../lib/format';
import { RegionInspector } from './RegionInspector';

export interface ResultsDossierProps {
  prediction: PredictionResponse;
  duration: number;
  onSeek: (seconds: number) => void;
  onReset: () => void;
}

function downloadJson(prediction: PredictionResponse) {
  const blob = new Blob([JSON.stringify(prediction, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `accentsense_${prediction.predicted_influence}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function ResultsDossier({
  prediction,
  duration,
  onSeek,
  onReset,
}: ResultsDossierProps) {
  const scores = sortedScores(prediction.all_scores);
  const regions = prediction.salient_regions;
  const displayName = accentDisplayName(prediction.predicted_influence);
  const others = scores
    .filter(([name]) => name !== prediction.predicted_influence)
    .slice(0, 3)
    .map(([name, p]) => `${accentDisplayName(name)} (${formatPercent(p)})`)
    .join(', ');

  return (
    <section className="space-y-space-xl" aria-live="polite" id="results">
      {/* Editorial headline */}
      <div
        className="pb-space-lg border-b border-outline-variant"
      >
        <div className="kicker mb-space-xs">Likely accent</div>
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-space-sm">
          <div className="space-y-space-xs">
            <h2
              className="font-headline text-headline-lg-mobile md:text-headline-lg font-normal text-on-surface"
            >
              {displayName}
            </h2>
            <p className="font-body-md text-body-md text-on-surface-variant">
              {prediction.language_family}
              {others && (
                <>
                  {' · '}Other possibilities:{' '}
                  <span className="text-on-surface font-medium">{others}</span>
                </>
              )}
            </p>
            {prediction.is_mock_prototype && (
              <span className="chip" title="No trained checkpoint found on the server">
                Prototype estimate — model not yet trained
              </span>
            )}
          </div>
          <div className="text-left md:text-right">
            <motion.div
              className="font-headline text-headline-lg tabular font-light text-primary-container"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
            >
              {formatPercent(prediction.confidence)}
            </motion.div>
            <div className="font-label-code text-label-code text-on-surface-variant">
              Confidence
            </div>
          </div>
        </div>
      </div>

      {/* All scores — animated bars */}
      <div className="space-y-space-md">
        <h3 className="font-sans text-headline-sm text-on-surface">
          Accent scores
        </h3>
        <motion.div
          className="divide-y divide-outline-variant border-t border-b border-outline-variant"
          variants={stagger(0.1, 0.05)}
          initial="hidden"
          animate="show"
        >
          {scores.map(([name, prob], idx) => {
            const isTop = name === prediction.predicted_influence;
            return (
              <motion.div
                key={name}
                className="py-space-sm grid grid-cols-[7rem_1fr_3rem] md:grid-cols-[10rem_1fr_4rem] items-center gap-space-md transition-colors hover:bg-surface-container-low dark:hover:bg-inverse-surface border-surface-container-highest"
                variants={{
                  hidden: { opacity: 0, x: -8 },
                  show: {
                    opacity: 1,
                    x: 0,
                    transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1], delay: idx * 0.05 },
                  },
                }}
              >
                <span
                  className={`font-label-code text-label-code truncate ${isTop ? 'text-primary-container font-semibold' : 'text-on-surface-variant font-medium'}`}
                >
                  {accentDisplayName(name)}
                </span>
                <div
                  className="h-2 rounded-sm overflow-hidden bg-surface-container-high"
                >
                  <motion.div
                    className={`h-full origin-left ${isTop ? 'bg-primary-container' : 'bg-outline'}`}
                    style={{
                      width: `${prob * 100}%`,
                    }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{
                      duration: 0.75,
                      ease: [0.22, 1, 0.36, 1],
                      delay: 0.2 + idx * 0.06,
                    }}
                  />
                </div>
                <span
                  className="font-label-code text-label-code tabular text-right text-on-surface-variant"
                >
                  {formatPercent(prob)}
                </span>
              </motion.div>
            );
          })}
        </motion.div>
      </div>

      {/* Why this result — numbered rows */}
      {regions.length > 0 && (
        <div className="space-y-space-md">
          <h3 className="font-sans text-headline-sm text-on-surface">
            Why this result?
          </h3>
          <motion.div
            className="divide-y divide-outline-variant border-t border-b border-outline-variant"
            variants={stagger(0.15, 0.08)}
            initial="hidden"
            animate="show"
          >
            {regions.slice(0, 5).map((region, i) => (
              <motion.div
                key={`${region.start_time_sec}-${i}`}
                className="py-space-md grid grid-cols-1 md:grid-cols-12 gap-space-sm md:gap-space-md items-start transition-colors hover:bg-surface-container-low dark:hover:bg-inverse-surface border-surface-container-highest"
                custom={i}
                variants={listItem}
              >
                <div className="md:col-span-4 flex items-baseline gap-space-xs">
                  <span className="font-label-code text-label-code text-primary-container">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="font-body-md text-body-md font-medium text-on-surface">
                    {region.linguistic_phenomenon}
                  </span>
                </div>
                <div className="md:col-span-8 font-body-md text-body-md text-on-surface-variant">
                  {region.phonetic_explanation}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      )}

      {/* Interactive region cards */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: 0.35 }}
      >
        <RegionInspector regions={regions} onSeek={onSeek} />
      </motion.div>

      {/* Actions */}
      <motion.div
        className="flex flex-col sm:flex-row items-center justify-between gap-space-md pt-space-md no-print border-t border-outline-variant"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.5 }}
      >
        <button type="button" onClick={onReset} className="btn-secondary group">
          <span className="material-symbols-outlined !text-[18px] transition-transform duration-300 group-hover:-rotate-180" aria-hidden="true">
            replay
          </span>
          <span>Try another recording</span>
        </button>
        <div className="flex items-center gap-space-sm">
          <button
            type="button"
            onClick={() => downloadJson(prediction)}
            className="btn-secondary !text-label-md text-on-surface-variant"
          >
            Download data
          </button>
          <button
            type="button"
            onClick={() => window.print()}
            className="btn-secondary bg-surface-container-high"
          >
            Save PDF summary
          </button>
        </div>
      </motion.div>

      {/* Duration footnote */}
      <p className="font-label-code text-label-code text-on-surface-variant">
        Sample duration {duration > 0 ? duration.toFixed(1) : '—'}s
      </p>
    </section>
  );
}

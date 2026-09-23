import { useState } from 'react';
import { motion } from 'framer-motion';
import { listItem, stagger } from '../lib/motion';
import type { SalientRegion } from '../lib/types';
import { formatTime } from '../lib/format';

export interface RegionInspectorProps {
  regions: SalientRegion[];
  onSeek: (seconds: number) => void;
}

/** "Audio breakdown" — clickable region cards that seek the waveform. */
export function RegionInspector({ regions, onSeek }: RegionInspectorProps) {
  const [selected, setSelected] = useState(0);

  if (regions.length === 0) return null;

  const handleClick = (index: number) => {
    setSelected(index);
    onSeek(regions[index].start_time_sec);
  };

  return (
    <div
      className="p-space-lg space-y-space-md bg-surface-container-low border border-outline-variant rounded"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-space-xs">
        <div>
          <h4 className="font-sans text-headline-sm text-on-surface">
            Audio breakdown
          </h4>
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            Click a section of the recording to see what stood out.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm bg-primary-container"
            aria-hidden="true"
          />
          <span className="font-label-code text-label-code text-on-surface-variant">
            Selected part
          </span>
        </div>
      </div>

      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-space-sm"
        variants={stagger(0.05, 0.07)}
        initial="hidden"
        animate="show"
      >
        {regions.map((region, i) => (
          <motion.button
            key={`${region.start_time_sec}-${i}`}
            type="button"
            className={`region-card ${selected === i ? 'selected' : ''}`}
            onClick={() => handleClick(i)}
            aria-pressed={selected === i}
            custom={i}
            variants={listItem}
            whileHover={{ y: -3, transition: { duration: 0.2 } }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
          >
            <div className="flex justify-between items-center mb-space-xs">
              <span
                className={`font-label-code text-label-code font-bold ${
                  selected === i
                    ? 'text-primary-container'
                    : 'text-on-surface-variant'
                }`}
              >
                PART {i + 1}
              </span>
              <span className="font-label-code text-label-code tabular text-on-surface-variant">
                {formatTime(region.start_time_sec)} — {formatTime(region.end_time_sec)}
              </span>
            </div>
            <div className="font-body-md text-body-md font-medium text-on-surface">
              {region.linguistic_phenomenon}
            </div>
            <div
              className="font-body-sm text-body-sm mt-1 text-on-surface-variant"
              style={{
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              {region.phonetic_explanation}
            </div>
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}

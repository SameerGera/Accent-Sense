/**
 * Published validation metrics for each class, from the model's own
 * validation run (keras.io example "English speaker accent recognition
 * using transfer learning", validation split).
 *
 * - precision: when the verdict is X, how often it is actually right
 * - recall:    how often a true X is detected at all
 *
 * Values are percentages (e.g. 74.3 === 74.3%).
 * Surfaced in the results UI so verdict uncertainty stays visible.
 */
export interface ClassMetrics {
  precision: number;
  recall: number;
}

export const PUBLISHED_METRICS: Record<string, ClassMetrics> = {
  Irish: { precision: 17.2, recall: 63.4 },
  Midlands: { precision: 13.4, recall: 51.7 },
  Northern: { precision: 30.2, recall: 50.6 },
  Scottish: { precision: 28.9, recall: 32.6 },
  Southern: { precision: 76.3, recall: 28.1 },
  Welsh: { precision: 74.3, recall: 83.3 },
  'Not a speech': { precision: 98.8, recall: 99.9 },
};

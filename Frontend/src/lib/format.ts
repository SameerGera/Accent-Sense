/** 142.5 → "2:22"; 8.05 → "0:08" */
export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** 0.8123 → "81%" */
export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** West_Midlands → "West Midlands"; RP → "RP" */
export function accentDisplayName(accent: string): string {
  return accent.replace(/_/g, ' ');
}

/** Sort all_scores entries descending by probability */
export function sortedScores(
  scores: Record<string, number>,
): Array<[string, number]> {
  return Object.entries(scores).sort((a, b) => b[1] - a[1]);
}

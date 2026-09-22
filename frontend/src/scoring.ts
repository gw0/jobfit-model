// Pure scoring logic, kept free of transformers.js/pdf.js so it is unit-testable.

export const NUM_QUANTILES = 3; // 0.05, 0.5, 0.95

export interface Aspect {
  id: string;
  name: string;
}

export interface AspectScore extends Aspect {
  score: number;
  low: number;
  high: number;
  insufficientData: boolean;
}

// calibration.json from pipeline/export.py; null where an aspect had no calibration data.
export interface CalibrationParams {
  [aspectId: string]: { delta: number | null; insufficient_data_threshold: number | null };
}

// Interval width alone drives "insufficient data" (specs §3/§8); there is no
// per-aspect special case. With no calibrated threshold for an aspect, every prediction
// reads "insufficient data" rather than an unbacked score.
export function isInsufficientData(low: number, high: number, threshold: number): boolean {
  if (!Number.isFinite(threshold)) return true;
  return high - low > threshold;
}

/** Raw model logits (aspect-major, NUM_QUANTILES per aspect) -> calibrated scores. */
export function toAspectScores(logits: ArrayLike<number>, aspects: Aspect[], calibration: CalibrationParams): AspectScore[] {
  return aspects.map((aspect, i) => {
    // Sorted so low <= mid <= high even where the raw quantile outputs cross.
    const [low, mid, high] = Array.from(logits)
      .slice(i * NUM_QUANTILES, (i + 1) * NUM_QUANTILES)
      .map(Number)
      .sort((a, b) => a - b);
    const delta = calibration[aspect.id]?.delta ?? NaN;
    const threshold = calibration[aspect.id]?.insufficient_data_threshold ?? NaN;
    return {
      id: aspect.id,
      name: aspect.name,
      score: mid,
      low: low - delta,
      high: high + delta,
      insufficientData: isInsufficientData(low - delta, high + delta, threshold),
    };
  });
}

export interface DownloadProgress {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}

/**
 * A transformers.js progress_callback that sums bytes over every file being
 * downloaded and reports (loaded, total) after each update.
 */
export function downloadTracker(onProgress: (loadedBytes: number, totalBytes: number) => void) {
  const files = new Map<string, { loaded: number; total: number }>();
  return (p: DownloadProgress) => {
    if (p.status !== "progress" || !p.file) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    onProgress(loaded, total);
  };
}

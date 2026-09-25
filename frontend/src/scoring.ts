// JobFit's reading of Jev answers, kept free of transformers.js/pdf.js so it is unit-testable.
import type { Answer, ScoreQuestion } from "./jev.mjs";

/** A questions.json entry: a Jev score question plus JobFit's own metadata. */
export type JobFitQuestion = ScoreQuestion & { name: string; scope: "pairwise" | "cv" | "job" };

// calibration.json from pipeline/export.py; null where there was nothing to fit on.
export interface Calibration {
  temperature: number | null;
  confidence_threshold: number | null;
}

export interface ResultRow {
  id: string;
  name: string;
  score: number; // expected level index, 0 .. levels - 1
  levels: number;
  label: string; // the nearest level's criteria text
  confidence: number;
  insufficientData: boolean;
}

// Confidence alone drives "insufficient data" (specs/20260923-jev-model.md §4); there is
// no per-question special case. Without a calibrated threshold every answer reads
// "insufficient data" rather than an unbacked score.
export function isInsufficientData(confidence: number, threshold: number | null): boolean {
  if (threshold === null || !Number.isFinite(threshold)) return true;
  return !(confidence >= threshold);
}

/** One table row per question, in questions.json order. */
export function toResultRows(
  answers: Record<string, Answer>,
  questions: Record<string, JobFitQuestion>,
  calibration: Calibration,
): ResultRow[] {
  return Object.entries(questions).map(([id, question]) => {
    const answer = answers[id];
    if (answer?.type !== "score") throw new Error(`${id}: expected a score answer`);
    const levels = question.criteria.length;
    return {
      id,
      name: question.name,
      score: answer.score,
      levels,
      label: question.criteria[Math.min(levels - 1, Math.max(0, Math.round(answer.score)))],
      confidence: answer.confidence,
      insufficientData: isInsufficientData(answer.confidence, calibration.confidence_threshold),
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

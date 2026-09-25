import { describe, expect, it } from "vitest";
import { downloadTracker, isInsufficientData, toResultRows, type JobFitQuestion } from "../src/scoring";

const QUESTIONS: Record<string, JobFitQuestion> = {
  a: { type: "score", name: "Question A", scope: "pairwise", question: "A?", criteria: ["none", "some", "all"] },
  b: { type: "score", name: "Question B", scope: "cv", question: "B?", criteria: ["low", "high"] },
};
const score = (value: number, confidence: number) => ({ type: "score" as const, score: value, confidence, probabilities: {} });

describe("isInsufficientData", () => {
  it("flags answers below the threshold only", () => {
    expect(isInsufficientData(0.3, 0.5)).toBe(true);
    expect(isInsufficientData(0.5, 0.5)).toBe(false);
    expect(isInsufficientData(0.9, 0.5)).toBe(false);
  });

  it("flags every answer when no calibrated threshold exists", () => {
    expect(isInsufficientData(0.99, null)).toBe(true);
    expect(isInsufficientData(0.99, NaN)).toBe(true);
  });
});

describe("toResultRows", () => {
  it("labels each score with its nearest level and applies the threshold", () => {
    const rows = toResultRows({ a: score(1.4, 0.8), b: score(0.2, 0.3) }, QUESTIONS,
      { temperature: 1.2, confidence_threshold: 0.5 });
    expect(rows[0]).toMatchObject({ id: "a", name: "Question A", score: 1.4, levels: 3, label: "some", insufficientData: false });
    expect(rows[1]).toMatchObject({ id: "b", levels: 2, label: "low", confidence: 0.3, insufficientData: true });
  });

  it("reads an uncalibrated model as insufficient data", () => {
    const rows = toResultRows({ a: score(2, 1), b: score(1, 1) }, QUESTIONS, { temperature: null, confidence_threshold: null });
    expect(rows.every((r) => r.insufficientData)).toBe(true);
  });
});

describe("downloadTracker", () => {
  it("sums the latest progress of every file and ignores other events", () => {
    const reports: [number, number][] = [];
    const track = downloadTracker((loaded, total) => reports.push([loaded, total]));
    track({ status: "initiate", file: "onnx/model_quantized.onnx" });
    track({ status: "progress", file: "tokenizer.json", loaded: 5, total: 10 });
    track({ status: "progress", file: "onnx/model_quantized.onnx", loaded: 100, total: 600 });
    track({ status: "progress", file: "tokenizer.json", loaded: 10, total: 10 });
    track({ status: "done", file: "tokenizer.json" });
    expect(reports).toEqual([[5, 10], [105, 610], [110, 610]]);
  });
});

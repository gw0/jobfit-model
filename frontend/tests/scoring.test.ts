import { describe, expect, it } from "vitest";
import { downloadTracker, isInsufficientData, toAspectScores } from "../src/scoring";

const ASPECTS = [
  { id: "a", name: "Aspect A" },
  { id: "b", name: "Aspect B" },
];

describe("isInsufficientData", () => {
  it("flags an interval wider than the threshold", () => {
    expect(isInsufficientData(0.1, 0.9, 0.5)).toBe(true);
  });

  it("accepts an interval within or exactly at the threshold", () => {
    expect(isInsufficientData(0.4, 0.5, 0.5)).toBe(false);
    expect(isInsufficientData(0.0, 0.5, 0.5)).toBe(false);
  });

  it("flags every prediction when no calibrated threshold exists", () => {
    expect(isInsufficientData(0.4, 0.5, NaN)).toBe(true);
  });
});

describe("toAspectScores", () => {
  const calibration = {
    a: { delta: 0.1, insufficient_data_threshold: 1.0 },
    b: { delta: 0.0, insufficient_data_threshold: 0.1 },
  };

  it("widens each interval by its aspect's conformal delta", () => {
    const [a, b] = toAspectScores([0.2, 0.5, 0.7, 0.4, 0.45, 0.6], ASPECTS, calibration);
    expect(a).toMatchObject({ id: "a", name: "Aspect A", score: 0.5, insufficientData: false });
    expect(a.low).toBeCloseTo(0.1);
    expect(a.high).toBeCloseTo(0.8);
    expect(b).toMatchObject({ score: 0.45, low: 0.4, high: 0.6, insufficientData: true }); // width 0.2 > 0.1
  });

  it("sorts crossed quantile outputs so low <= score <= high", () => {
    const [a] = toAspectScores([0.9, 0.3, 0.5, 0, 0, 0], ASPECTS, calibration);
    expect(a.score).toBe(0.5);
    expect(a.low).toBeCloseTo(0.2);
    expect(a.high).toBeCloseTo(1.0);
  });

  it("reads uncalibrated (null or missing) aspects as insufficient data", () => {
    const scores = toAspectScores([0.4, 0.5, 0.6, 0.4, 0.5, 0.6], ASPECTS, {
      a: { delta: null, insufficient_data_threshold: null },
    });
    expect(scores.every((s) => s.insufficientData)).toBe(true);
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

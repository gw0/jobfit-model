// Mirrors pipeline/tests/test_prepare.py's encode_pair cases; the real tokenizer's
// parity with Python is checked end-to-end by scripts/verify-parity.mjs.
import { describe, expect, it } from "vitest";
import { CV_BUDGET, encodePair, JD_BUDGET, MAX_LENGTH } from "../src/tokenize.mjs";

// One token per word, ids are word lengths.
const tokenizer = {
  pad_token_id: 0,
  encode: (text: string) => text.split(/\s+/).filter(Boolean).map((w) => w.length),
};
const words = (n: number) => Array.from({ length: n }, () => "x").join(" ");

describe("encodePair", () => {
  it("puts the JD after the CV and pads to MAX_LENGTH", () => {
    const { paddedIds, paddedMask, cvTruncated, jdTruncated } = encodePair(tokenizer, "a bb", "ccc");
    expect(paddedIds.slice(0, 4)).toEqual([1, 2, 3, 0]);
    expect(paddedIds).toHaveLength(MAX_LENGTH);
    expect(paddedMask.slice(0, 4)).toEqual([1, 1, 1, 0]);
    expect([cvTruncated, jdTruncated]).toEqual([false, false]);
  });

  it("truncates each side to its own budget", () => {
    const { paddedMask, cvTruncated, jdTruncated } = encodePair(tokenizer, words(CV_BUDGET + 5), words(10));
    expect(cvTruncated).toBe(true);
    expect(jdTruncated).toBe(false);
    expect(paddedMask.reduce((a: number, b: number) => a + b, 0)).toBe(CV_BUDGET + 10);

    const both = encodePair(tokenizer, words(CV_BUDGET + 1), words(JD_BUDGET + 1));
    expect([both.cvTruncated, both.jdTruncated]).toEqual([true, true]);
    expect(both.paddedMask.every((m: number) => m === 1)).toBe(true);
  });

  it("pads with the tokenizer's own pad id", () => {
    const { paddedIds } = encodePair({ ...tokenizer, pad_token_id: 151643 }, "a", "b");
    expect(paddedIds[2]).toBe(151643);
  });
});

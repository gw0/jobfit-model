// Mirrors pipeline/tests/test_jev.py's encode and readout cases; the real tokenizer's
// parity with Python is checked end-to-end by scripts/verify-parity.mjs.
import { describe, expect, it } from "vitest";
import { answers, answerKeys, candidateIds, candidates, encode, MAX_CANDIDATES, renderQuestion, scoreStats } from "../src/jev.mjs";
import type { Question } from "../src/jev.mjs";

const SCORE: Question = { type: "score", question: "How good?", criteria: ["bad", "ok", "good"] };
const CHOICE: Question = { type: "choice", question: "Which?", criteria: { x: "first", y: "second" } };
const NOUL: Question = { type: "noul", question: "Is it?", criteria: { true: "yes", false: "no" } };

// One token per whitespace-separated word; ids assigned on first sight.
function fakeTokenizer() {
  const vocab = new Map<string, number>();
  return {
    vocab,
    encode: (text: string) => text.split(/\s+/).filter(Boolean).map((w) => {
      if (!vocab.has(w)) vocab.set(w, vocab.size + 1);
      return vocab.get(w)!;
    }),
  };
}

describe("questions", () => {
  it("has keys and candidates per type", () => {
    expect([answerKeys(SCORE), candidates(SCORE)]).toEqual([["0", "1", "2"], ["0", "1", "2"]]);
    expect([answerKeys(CHOICE), candidates(CHOICE)]).toEqual([["x", "y"], ["A", "B"]]);
    expect([answerKeys(NOUL), candidates(NOUL)]).toEqual([["true", "false"], ["true", "false"]]);
  });

  it("renders Markdown ending in the answer prompt", () => {
    expect(renderQuestion(CHOICE)).toBe("\n\n---\n\n# Question\nWhich?\n\n# Criteria\nA = first\nB = second\n\n# Answer\n");
  });

  it("resolves single-token candidate ids", () => {
    const tok = fakeTokenizer();
    const { ids, counts } = candidateIds(tok, { s: SCORE, n: NOUL });
    expect(counts).toEqual([3, 2]);
    expect(ids[1]).toEqual([tok.vocab.get("true"), tok.vocab.get("false"), ...new Array(8).fill(0)]);
    const twoTokenDigits = { encode: (t: string) => [...tok.encode(t), ...(/\d$/.test(t) ? [999] : [])] };
    expect(() => candidateIds(twoTokenDigits, { s: SCORE })).toThrow(/single token/);
  });
});

describe("encode", () => {
  it("lays out state, branches and padding", () => {
    const tok = fakeTokenizer();
    const out = encode(tok, { CV: "a b", Job: "c" }, { s: SCORE, n: NOUL }, 5, 40, 0);
    const seg = out.segment_ids;
    expect(seg.slice(0, 8)).toEqual(new Array(8).fill(0));
    expect(out.truncated).toEqual({ CV: false, Job: false });
    expect(out.input_ids).toHaveLength(2 * 5 + 40);
    out.answer_positions.forEach((pos: number, k: number) => {
      expect(seg[pos]).toBe(k + 1);
      expect(seg[pos + 1]).not.toBe(k + 1);
    });
    expect(out.input_ids[out.answer_positions[0]]).toBe(tok.vocab.get("Answer"));
  });

  it("truncates each part including its header", () => {
    const out = encode(fakeTokenizer(), { CV: "w ".repeat(10), Job: "j" }, { s: SCORE }, 5, 40);
    expect(out.truncated).toEqual({ CV: true, Job: false });
    expect(out.segment_ids.filter((s: number) => s === 0)).toHaveLength(9);
  });

  it("refuses to truncate questions", () => {
    expect(() => encode(fakeTokenizer(), { CV: "a" }, { s: SCORE }, 5, 5)).toThrow(/budget/);
  });
});

describe("readout", () => {
  it.each([
    [[0, 0, 1, 0, 0], 1.0], [[0, 0.5, 0.5, 0, 0], 0.75], [[0.2, 0.2, 0.2, 0.2, 0.2], 1 - Math.SQRT2 / 2], [[0.5, 0, 0, 0, 0.5], 0],
  ])("score confidence of %j is %d", (probs, confidence) => {
    expect(scoreStats(probs)[1]).toBeCloseTo(confidence, 12);
  });

  it("answers every type from flat logits", () => {
    const row = (p: number[]) => [...p.map(Math.log), ...new Array(MAX_CANDIDATES - p.length).fill(Math.log(1e-9))];
    const out = answers({ s: SCORE, c: CHOICE, n: NOUL }, [...row([0.1, 0.2, 0.7]), ...row([0.25, 0.75]), ...row([0.9, 0.1])]);
    expect(out.s).toMatchObject({ type: "score" });
    expect(out.s.type === "score" && out.s.score).toBeCloseTo(1.6);
    expect(out.c).toMatchObject({ type: "choice", choice: "y" });
    expect(out.c.type === "choice" && out.c.confidence).toBeCloseTo(0.5);
    expect(out.n.type === "noul" && out.n.noul).toBeCloseTo(0.9);
  });
});

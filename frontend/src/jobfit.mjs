import { Tensor } from "@huggingface/transformers";

import { MAX_CANDIDATES } from "./jev.mjs";

// JobFit's use of the Jev model: its state is {CV, Job description}, truncated as a
// whole to STATE_BUDGET tokens, and its questions (questions.json) must fit
// QUESTIONS_BUDGET. Mirrors pipeline/common.py.
export const STATE_BUDGET = 3072;
export const QUESTIONS_BUDGET = 1024;

export function jobfitState(/** @type {string} */ cvText, /** @type {string} */ jdText) {
  return { CV: cvText, "Job description": jdText };
}

const int64 = (/** @type {number[]} */ values, /** @type {number[]} */ dims) =>
  new Tensor("int64", BigInt64Array.from(values.map(BigInt)), dims);

/** The exported graph's inputs for one encoded state (jev.mjs `encode`) and its `candidateIds`. */
export function modelInputs(encoded, /** @type {number[][]} */ ids) {
  const length = encoded.input_ids.length;
  return {
    input_ids: int64(encoded.input_ids, [1, length]),
    segment_ids: int64(encoded.segment_ids, [1, length]),
    answer_positions: int64(encoded.answer_positions, [1, ids.length]),
    candidate_ids: int64(ids.flat(), [ids.length, MAX_CANDIDATES]),
  };
}

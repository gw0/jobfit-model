// JobFit's use of the Jev model: its state is {CV, Job description}, truncated as a
// whole to STATE_BUDGET tokens, and its questions (questions.json) must fit
// QUESTIONS_BUDGET. Mirrors pipeline/common.py.
export const STATE_BUDGET = 3072;
export const QUESTIONS_BUDGET = 1024;

export function jobfitState(/** @type {string} */ cvText, /** @type {string} */ jdText) {
  return { CV: cvText, "Job description": jdText };
}

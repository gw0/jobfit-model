// Jev-shaped typed-decision model, the generic core: an exact mirror of pipeline/jev.py's
// rendering, encoding and readout (specs/20260923-jev-model.md). Plain JS so the Node
// parity script (scripts/verify-parity.mjs) runs the same code as the browser build.

/**
 * @typedef {{type: "score", question: string, criteria: string[]}} ScoreQuestion
 * @typedef {{type: "choice", question: string, criteria: Record<string, string>}} ChoiceQuestion
 * @typedef {{type: "noul", question: string, criteria: {true: string, false: string}}} NoulQuestion
 * @typedef {ScoreQuestion | ChoiceQuestion | NoulQuestion} Question
 * @typedef {{type: "score", score: number, confidence: number, probabilities: Record<string, number>}} ScoreAnswer
 * @typedef {{type: "choice", choice: string, confidence: number, probabilities: Record<string, number>}} ChoiceAnswer
 * @typedef {{type: "noul", noul: number}} NoulAnswer
 * @typedef {ScoreAnswer | ChoiceAnswer | NoulAnswer} Answer
 * @typedef {{encode: (text: string, options: {add_special_tokens: boolean}) => number[]}} Tokenizer
 */

export const MAX_CANDIDATES = 10;
export const SEPARATOR = "\n\n---\n\n";
export const ANSWER_PROMPT = "# Answer\n";
const CHOICE_LETTERS = "ABCDEFGHIJ";

/** The answer keys, in candidate order: level indices, option keys, or true/false. */
export function answerKeys(/** @type {Question} */ question) {
  if (question.type === "score") return question.criteria.map((_, k) => String(k));
  if (question.type === "choice") return Object.keys(question.criteria);
  return ["true", "false"];
}

/** The candidate answer strings, each of which must be one token after ANSWER_PROMPT. */
export function candidates(/** @type {Question} */ question) {
  if (question.type === "choice") return [...CHOICE_LETTERS.slice(0, Object.keys(question.criteria).length)];
  return answerKeys(question);
}

function descriptions(/** @type {Question} */ question) {
  if (question.type === "score") return question.criteria;
  const criteria = /** @type {Record<string, string>} */ (question.criteria);
  return answerKeys(question).map((key) => criteria[key]);
}

/** One question branch, starting with the separator that follows the state. */
export function renderQuestion(/** @type {Question} */ question) {
  const descs = descriptions(question);
  const lines = candidates(question).map((c, i) => `${c} = ${descs[i]}`);
  return `${SEPARATOR}# Question\n${question.question}\n\n# Criteria\n${lines.join("\n")}\n\n${ANSWER_PROMPT}`;
}

export function renderPartHeader(/** @type {string} */ title, /** @type {boolean} */ first) {
  return (first ? "" : SEPARATOR) + `# ${title}\n`;
}

const encodeText = (/** @type {Tokenizer} */ tokenizer, /** @type {string} */ text) =>
  Array.from(tokenizer.encode(text, { add_special_tokens: false }));

/**
 * (Q, MAX_CANDIDATES) token ids, zero-padded, plus each question's candidate count.
 * Throws unless every candidate is exactly one token after ANSWER_PROMPT.
 * @param {Tokenizer} tokenizer
 * @param {Record<string, Question>} questions
 */
export function candidateIds(tokenizer, questions) {
  const prompt = encodeText(tokenizer, ANSWER_PROMPT);
  const ids = [];
  const counts = [];
  for (const [qid, q] of Object.entries(questions)) {
    const row = candidates(q).map((c) => {
      const full = encodeText(tokenizer, ANSWER_PROMPT + c);
      if (full.length !== prompt.length + 1 || prompt.some((id, i) => full[i] !== id)) {
        throw new Error(`${qid}: candidate ${JSON.stringify(c)} is not a single token after the answer prompt`);
      }
      return full[full.length - 1];
    });
    ids.push([...row, ...new Array(MAX_CANDIDATES - row.length).fill(0)]);
    counts.push(row.length);
  }
  return { ids, counts };
}

/**
 * Tokenizes `state` ({title: text}, each part header + text truncated to `partBudget`
 * tokens) followed by one branch per question, padded to
 * (number of parts) * partBudget + questionsBudget. Throws if the questions don't fit
 * `questionsBudget` -- question text is never truncated.
 * @param {Tokenizer} tokenizer
 * @param {Record<string, string>} state
 * @param {Record<string, Question>} questions
 * @param {number} partBudget
 * @param {number} questionsBudget
 * @param {number} padTokenId
 */
export function encode(tokenizer, state, questions, partBudget, questionsBudget, padTokenId = 0) {
  /** @type {number[]} */ const inputIds = [];
  /** @type {number[]} */ const segmentIds = [];
  /** @type {Record<string, boolean>} */ const truncated = {};
  Object.entries(state).forEach(([title, text], i) => {
    const header = encodeText(tokenizer, renderPartHeader(title, i === 0));
    const body = encodeText(tokenizer, text);
    const keep = Math.max(0, partBudget - header.length);
    truncated[title] = body.length > keep;
    const part = [...header, ...body.slice(0, keep)].slice(0, partBudget);
    inputIds.push(...part);
    segmentIds.push(...new Array(part.length).fill(0));
  });

  const answerPositions = [];
  let questionsLen = 0;
  Object.values(questions).forEach((q, i) => {
    const branch = encodeText(tokenizer, renderQuestion(q));
    questionsLen += branch.length;
    inputIds.push(...branch);
    segmentIds.push(...new Array(branch.length).fill(i + 1));
    answerPositions.push(inputIds.length - 1);
  });
  if (questionsLen > questionsBudget) {
    throw new Error(`questions take ${questionsLen} tokens, over the ${questionsBudget} budget`);
  }

  const pad = Object.keys(state).length * partBudget + questionsBudget - inputIds.length;
  return {
    input_ids: [...inputIds, ...new Array(pad).fill(padTokenId)],
    segment_ids: [...segmentIds, ...new Array(pad).fill(-1)],
    answer_positions: answerPositions,
    truncated,
  };
}

/** Softmax over the first `count` candidate logits. */
export function probabilities(/** @type {ArrayLike<number>} */ logits, /** @type {number} */ count, temperature = 1) {
  const z = Array.from(logits).slice(0, count).map((x) => Number(x) / temperature);
  const max = Math.max(...z);
  const exp = z.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / sum);
}

/** [score, confidence] of a level distribution: the expected level index, and 1 - sd/sd_max. */
export function scoreStats(/** @type {number[]} */ probs) {
  const score = probs.reduce((acc, p, k) => acc + p * k, 0);
  const variance = probs.reduce((acc, p, k) => acc + p * (k - score) ** 2, 0);
  return [score, 1 - Math.sqrt(variance) / ((probs.length - 1) / 2)];
}

/**
 * One Jev answer from a question's candidate logits.
 * @param {Question} question
 * @param {ArrayLike<number>} logits
 * @returns {Answer}
 */
export function answer(question, logits, temperature = 1) {
  const keys = answerKeys(question);
  const probs = probabilities(logits, keys.length, temperature);
  const byKey = Object.fromEntries(keys.map((key, i) => [key, probs[i]]));
  if (question.type === "score") {
    const [score, confidence] = scoreStats(probs);
    return { type: "score", score, confidence, probabilities: byKey };
  }
  if (question.type === "choice") {
    const k = keys.length;
    const max = Math.max(...probs);
    return { type: "choice", choice: keys[probs.indexOf(max)], confidence: (k * max - 1) / (k - 1), probabilities: byKey };
  }
  return { type: "noul", noul: probs[0] };
}

/**
 * {id: answer} from one example's flat (Q * MAX_CANDIDATES) answer logits.
 * @param {Record<string, Question>} questions
 * @param {ArrayLike<number>} answerLogits
 * @returns {Record<string, Answer>}
 */
export function answers(questions, answerLogits, temperature = 1) {
  const flat = Array.from(answerLogits);
  return Object.fromEntries(Object.entries(questions).map(([qid, q], i) =>
    [qid, answer(q, flat.slice(i * MAX_CANDIDATES, (i + 1) * MAX_CANDIDATES), temperature)]));
}

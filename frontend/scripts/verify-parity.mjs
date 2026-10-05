// Checks that transformers.js reproduces the pipeline for the sample state in
// parity.json (written by pipeline/export.py): identical encoding (token ids, segment
// ids, answer positions, candidate ids) and pad id, answer logits within the stated
// tolerance of ONNX Runtime in Python, and the same answers -- for JobFit's questions
// plus a choice and a noul question. Exits 1 on mismatch.
//
// Runs on onnxruntime-node (device "cpu"); browsers use the wasm/webgpu providers,
// which only a real browser can exercise.
//
// Usage: node scripts/verify-parity.mjs [model-dir]   (default: public/models/default)
import { readFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { AutoTokenizer, env, PreTrainedModel, Tensor } from "@huggingface/transformers";
import { answers, candidateIds, encode, MAX_CANDIDATES } from "../src/jev.mjs";
import { STATE_BUDGET } from "../src/jobfit.mjs";

const modelDir = resolve(process.argv[2] ?? new URL("../public/models/default", import.meta.url).pathname);
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = dirname(modelDir) + "/";
const modelId = basename(modelDir);

const expected = JSON.parse(readFileSync(`${modelDir}/parity.json`, "utf-8"));
const failures = [];
const firstDiff = (a, b) => (a.length !== b.length ? Math.min(a.length, b.length) : a.findIndex((x, i) => x !== b[i]));

if (STATE_BUDGET !== expected.state_budget) failures.push(`STATE_BUDGET ${STATE_BUDGET} != ${expected.state_budget}`);
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const padId = tokenizer.pad_token_id ?? 0;
if (padId !== expected.pad_token_id) failures.push(`pad token id ${padId} != ${expected.pad_token_id}`);
const encoded = encode(tokenizer, expected.state, expected.questions, expected.state_budget, expected.questions_budget, padId);
const { ids } = candidateIds(tokenizer, expected.questions);
for (const [name, actual, want] of [
  ["input_ids", encoded.input_ids, expected.input_ids],
  ["segment_ids", encoded.segment_ids, expected.segment_ids],
  ["answer_positions", encoded.answer_positions, expected.answer_positions],
  ["candidate_ids", ids.flat(), expected.candidate_ids.flat()],
]) {
  const at = firstDiff(actual, want);
  if (at !== -1) failures.push(`${name} differ (length ${actual.length} vs ${want.length}, first mismatch at ${at})`);
}
console.log(`encoding: ${encoded.input_ids.length} ids, ${failures.length ? "MISMATCH" : "identical"}`);

const int64 = (values, dims) => new Tensor("int64", BigInt64Array.from(values.map(BigInt)), dims);
const model = await PreTrainedModel.from_pretrained(modelId, { dtype: "q8", device: "cpu" });
const length = encoded.input_ids.length;
const output = await model({
  input_ids: int64(encoded.input_ids, [1, length]),
  segment_ids: int64(encoded.segment_ids, [1, length]),
  answer_positions: int64(encoded.answer_positions, [1, ids.length]),
  candidate_ids: int64(ids.flat(), [ids.length, MAX_CANDIDATES]),
});
const logits = Array.from(output.answer_logits.data, Number);
const want = expected.answer_logits.flat();
const maxDiff = logits.length === want.length ? Math.max(...logits.map((x, i) => Math.abs(x - want[i]))) : Infinity;
console.log(`answer logits: max |diff| ${maxDiff.toExponential(2)} (atol ${expected.atol})`);
if (!(maxDiff <= expected.atol)) failures.push(`answer logits differ by ${maxDiff} > ${expected.atol}`);

// Same readout on the Python logits: must match the Python answers to float precision.
const close = (a, b) => (typeof b === "number" ? Math.abs(a - b) <= 1e-9
  : typeof b === "object" ? Object.keys(b).every((k) => close(a?.[k], b[k])) : a === b);
const got = answers(expected.questions, want, expected.temperature);
for (const [qid, answer] of Object.entries(expected.answers)) {
  if (!close(got[qid], answer)) failures.push(`answer ${qid}: ${JSON.stringify(got[qid])} != ${JSON.stringify(answer)}`);
}
console.log(`answers: ${Object.keys(expected.answers).length} questions (${[...new Set(Object.values(got).map((a) => a.type))].join(", ")})`);

if (failures.length) {
  console.error(`PARITY FAILED (${expected.cv} x ${expected.job}):\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log("parity OK");

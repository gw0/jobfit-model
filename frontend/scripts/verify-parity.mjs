// Checks that transformers.js reproduces the pipeline for the sample pair in
// parity.json (written by pipeline/export.py): identical token ids and pad id, and
// logits within the stated tolerance of ONNX Runtime in Python. Exits 1 on mismatch.
//
// Runs on onnxruntime-node (device "cpu"); browsers use the wasm/webgpu providers,
// which only a real browser can exercise.
//
// Usage: node scripts/verify-parity.mjs [model-dir]   (default: public/models/qwen3-0.6b)
import { readFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { AutoTokenizer, env, PreTrainedModel, Tensor } from "@huggingface/transformers";
import { encodePair, MAX_LENGTH } from "../src/tokenize.mjs";

const modelDir = resolve(process.argv[2] ?? new URL("../public/models/qwen3-0.6b", import.meta.url).pathname);
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = dirname(modelDir) + "/";
const modelId = basename(modelDir);

const expected = JSON.parse(readFileSync(`${modelDir}/parity.json`, "utf-8"));
const failures = [];

const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const { paddedIds, paddedMask } = encodePair(tokenizer, expected.cv_text, expected.jd_text);
const ids = paddedIds.slice(0, paddedMask.reduce((a, b) => a + b, 0));
const padId = tokenizer.pad_token_id ?? 0;
if (padId !== expected.pad_token_id) {
  failures.push(`pad token id ${padId} != ${expected.pad_token_id}`);
}
const firstDiff = ids.findIndex((id, i) => id !== expected.input_ids[i]);
if (ids.length !== expected.input_ids.length || firstDiff !== -1) {
  failures.push(`token ids differ (length ${ids.length} vs ${expected.input_ids.length}, first mismatch at ${firstDiff})`);
}
console.log(`tokens: ${ids.length} non-pad ids, ${failures.length ? "MISMATCH" : "identical"}`);

const model = await PreTrainedModel.from_pretrained(modelId, { dtype: "q8", device: "cpu" });
const output = await model({
  input_ids: new Tensor("int64", BigInt64Array.from(paddedIds.map(BigInt)), [1, MAX_LENGTH]),
  attention_mask: new Tensor("int64", BigInt64Array.from(paddedMask.map(BigInt)), [1, MAX_LENGTH]),
});
const logits = Array.from(output.logits.data, Number);
const maxDiff = logits.length === expected.logits.length
  ? Math.max(...logits.map((x, i) => Math.abs(x - expected.logits[i])))
  : Infinity;
console.log(`logits: max |diff| ${maxDiff.toExponential(2)} (atol ${expected.atol})`);
if (!(maxDiff <= expected.atol)) failures.push(`logits differ by ${maxDiff} > ${expected.atol}`);

if (failures.length) {
  console.error(`PARITY FAILED (${expected.cv} x ${expected.job}):\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log("parity OK");

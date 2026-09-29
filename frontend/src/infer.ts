// In-browser inference (specs §7): transformers.js on WebGPU, falling back to WASM, and
// pdf.js text extraction for uploaded CVs.
import { AutoTokenizer, env, PreTrainedModel, Tensor } from "@huggingface/transformers";
import * as pdfjsLib from "pdfjs-dist";

import { answers, candidateIds, encode, type Answer, type Question } from "./jev.mjs";
import { STATE_BUDGET, QUESTIONS_BUDGET } from "./jobfit.mjs";
import { downloadTracker, type Calibration } from "./scoring";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

// The repo ships no weights: a deployed build loads them from the HF Hub repo baked in
// via VITE_HF_MODEL_REPO; without it, from a local copy under /models/<slug>/.
const HF_MODEL_REPO = import.meta.env.VITE_HF_MODEL_REPO;
const MODEL_ID = HF_MODEL_REPO || "smollm2-135m-instruct";
env.allowRemoteModels = Boolean(HF_MODEL_REPO);
env.allowLocalModels = !HF_MODEL_REPO;

// calibration.json is this project's own file, which transformers.js doesn't fetch.
const CALIBRATION_URL = HF_MODEL_REPO
  ? `https://huggingface.co/${HF_MODEL_REPO}/resolve/main/calibration.json`
  : `/models/${MODEL_ID}/calibration.json`;

/** Called with the bytes downloaded so far and the total, summed over all model files. */
export type ProgressCallback = (loadedBytes: number, totalBytes: number) => void;

let modelPromise: ReturnType<typeof load> | null = null;

async function load(onProgress?: ProgressCallback) {
  const progress_callback = downloadTracker(onProgress ?? (() => {}));

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  // PreTrainedModel runs the exported JevModel graph generically: it passes our
  // custom-named inputs straight through and returns its `answer_logits` output.
  let model;
  try {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "webgpu", progress_callback });
  } catch {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "wasm", progress_callback });
  }
  const calibration: Calibration = await (await fetch(CALIBRATION_URL)).json();
  return { tokenizer, model, calibration };
}

/** Loads (once) and caches the tokenizer, model and calibration. */
export function loadModel(onProgress?: ProgressCallback) {
  if (!modelPromise) modelPromise = load(onProgress);
  return modelPromise;
}

const int64 = (values: number[], dims: number[]) => new Tensor("int64", BigInt64Array.from(values.map(BigInt)), dims);

/**
 * The Jev call: one typed answer per question id, all from one forward pass. The state
 * is a {title: text} mapping, truncated as a whole to STATE_BUDGET tokens.
 */
export async function evaluate(
  state: Record<string, string>,
  questions: Record<string, Question>,
): Promise<{ model: string; answers: Record<string, Answer> }> {
  const { tokenizer, model, calibration } = await loadModel();
  const encoded = encode(tokenizer, state, questions, STATE_BUDGET, QUESTIONS_BUDGET, tokenizer.pad_token_id ?? 0);
  const { ids } = candidateIds(tokenizer, questions);
  const length = encoded.input_ids.length;
  const numQuestions = ids.length;
  const output = await model({
    input_ids: int64(encoded.input_ids, [1, length]),
    segment_ids: int64(encoded.segment_ids, [1, length]),
    answer_positions: int64(encoded.answer_positions, [1, numQuestions]),
    candidate_ids: int64(ids.flat(), [numQuestions, ids[0].length]),
  });
  return {
    model: MODEL_ID,
    answers: answers(questions, output.answer_logits.data as Float32Array, calibration.temperature ?? 1),
  };
}

// Deliberately basic (specs §7): getTextContent() with no layout reconstruction, to
// mirror how real ATS parsers mangle PDFs.
export async function extractPdfText(file: File): Promise<string> {
  const doc = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const pageTexts: string[] = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    const content = await (await doc.getPage(pageNum)).getTextContent();
    pageTexts.push(content.items.map((item) => ("str" in item ? item.str : "")).join(" "));
  }
  return pageTexts.join("\n");
}

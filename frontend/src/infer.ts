// In-browser inference (specs §7): transformers.js on WebGPU, falling back to WASM, and
// pdf.js text extraction for uploaded CVs.
import { AutoTokenizer, env, PreTrainedModel } from "@huggingface/transformers";
import * as pdfjsLib from "pdfjs-dist";

import { answers, candidateIds, encode, type Answer, type Question } from "./jev.mjs";
import { modelInputs, STATE_BUDGET, QUESTIONS_BUDGET } from "./jobfit.mjs";
import { downloadTracker, type Calibration } from "./scoring";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

/** Called with the bytes downloaded so far and the total, summed over all model files. */
type ProgressCallback = (loadedBytes: number, totalBytes: number) => void;

// The model URL names the model, so the id is only a placeholder transformers.js accepts.
const MODEL_ID = "model";

let modelPromise: ReturnType<typeof load> | null = null;

async function load(onProgress?: ProgressCallback) {
  const progress_callback = downloadTracker(onProgress ?? (() => {}));

  // The repo ships no weights. config.json (a deploy-time setting, see docker/) holds the
  // base URL of the model files: an HF Hub-compatible repo URL or a path on this origin.
  const { modelUrl } = await (await fetch("config.json")).json();
  const { origin, pathname } = new URL(modelUrl, location.href);
  const path = pathname.replace(/\/+$/, "");
  env.remoteHost = origin;
  env.remotePathTemplate = `${path}/`;

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  // PreTrainedModel runs the exported JevModel graph generically: it passes our
  // custom-named inputs straight through and returns its `answer_logits` output.
  let model;
  try {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "webgpu", progress_callback });
  } catch {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "wasm", progress_callback });
  }
  // calibration.json is this project's own file, which transformers.js doesn't fetch.
  const calibration: Calibration = await (await fetch(`${origin}${path}/calibration.json`)).json();
  return { tokenizer, model, calibration };
}

/** Loads (once) and caches the tokenizer, model and calibration. */
export function loadModel(onProgress?: ProgressCallback) {
  modelPromise ??= load(onProgress).catch((err) => {
    modelPromise = null; // a failed load can be retried
    throw err;
  });
  return modelPromise;
}

/**
 * The Jev call: one typed answer per question id, all from one forward pass. The state
 * is a {title: text} mapping, truncated as a whole to STATE_BUDGET tokens.
 */
export async function evaluate(
  state: Record<string, string>,
  questions: Record<string, Question>,
): Promise<Record<string, Answer>> {
  const { tokenizer, model, calibration } = await loadModel();
  const encoded = encode(tokenizer, state, questions, STATE_BUDGET, QUESTIONS_BUDGET, tokenizer.pad_token_id ?? 0);
  const { ids } = candidateIds(tokenizer, questions);
  const output = await model(modelInputs(encoded, ids));
  return answers(questions, output.answer_logits.data as Float32Array, calibration.temperature ?? 1);
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

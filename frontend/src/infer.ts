// In-browser inference (specs §7): transformers.js on WebGPU, falling back to WASM, and
// pdf.js text extraction for uploaded CVs.
import { AutoTokenizer, env, PreTrainedModel, Tensor } from "@huggingface/transformers";
import * as pdfjsLib from "pdfjs-dist";

import aspects from "../../aspects.json";
import { downloadTracker, toAspectScores, type AspectScore, type CalibrationParams } from "./scoring";
import { encodePair as encodePairArrays, MAX_LENGTH } from "./tokenize.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

// The repo ships no weights: a deployed build loads them from the HF Hub repo baked in
// via VITE_HF_MODEL_REPO; without it, from a local copy under /models/<slug>/.
const HF_MODEL_REPO = import.meta.env.VITE_HF_MODEL_REPO;
const MODEL_ID = HF_MODEL_REPO || "qwen3-0.6b";
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
  // AutoModelForSequenceClassification has no entry for the qwen/llama architectures;
  // PreTrainedModel runs the ONNX graph generically, keyed on its `logits` output.
  let model;
  try {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "webgpu", progress_callback });
  } catch {
    model = await PreTrainedModel.from_pretrained(MODEL_ID, { dtype: "q8", device: "wasm", progress_callback });
  }
  const calibration: CalibrationParams = await (await fetch(CALIBRATION_URL)).json();
  return { tokenizer, model, calibration };
}

/** Loads (once) and caches the tokenizer, model and calibration. */
export function loadModel(onProgress?: ProgressCallback) {
  if (!modelPromise) modelPromise = load(onProgress);
  return modelPromise;
}

export async function scoreCvJob(cvText: string, jdText: string): Promise<AspectScore[]> {
  const { tokenizer, model, calibration } = await loadModel();
  const { paddedIds, paddedMask } = encodePairArrays(tokenizer, cvText, jdText);
  const output = await model({
    input_ids: new Tensor("int64", BigInt64Array.from(paddedIds.map(BigInt)), [1, MAX_LENGTH]),
    attention_mask: new Tensor("int64", BigInt64Array.from(paddedMask.map(BigInt)), [1, MAX_LENGTH]),
  });
  return toAspectScores(output.logits.data as Float32Array, aspects, calibration);
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

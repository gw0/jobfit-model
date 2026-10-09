<script setup lang="ts">
import { ref } from "vue";
import questionsJson from "../../questions.json";
import { evaluate, extractPdfText, loadModel } from "./infer";
import { jobfitState } from "./jobfit.mjs";
import { toResultRows, type JobFitQuestion, type ResultRow } from "./scoring";

const questions = questionsJson as Record<string, JobFitQuestion>;

const cvText = ref("");
const jdText = ref("");
const results = ref<ResultRow[] | null>(null);
const status = ref<"idle" | "loading" | "scoring" | "error">("idle");
const errorMessage = ref("");
const download = ref({ loaded: 0, total: 0 });

const toMb = (bytes: number) => (bytes / 1_000_000).toFixed(0);

async function onCvFileChange(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    cvText.value = await extractPdfText(file);
  } catch (err) {
    fail(err);
  }
}

function fail(err: unknown) {
  status.value = "error";
  errorMessage.value = err instanceof Error ? err.message : String(err);
}

async function onSubmit() {
  errorMessage.value = "";
  results.value = null;
  try {
    status.value = "loading";
    const { calibration } = await loadModel((loaded, total) => {
      download.value = { loaded, total };
    });
    status.value = "scoring";
    const answers = await evaluate(jobfitState(cvText.value, jdText.value), questions);
    results.value = toResultRows(answers, questions, calibration);
    status.value = "idle";
  } catch (err) {
    fail(err);
  }
}
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">JobFit</h1>
      <p class="mb-3">
        <a href="https://github.com/gw0/jobfit-model"><img src="https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github" alt="GitHub" /></a>
        <a href="https://huggingface.co/datasets/gw0/jobfit-jevbench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange" alt="HF Dataset" /></a>
        <a href="https://huggingface.co/gw0/jobfit-model"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow" alt="HF Model" /></a>
        <a href="https://huggingface.co/spaces/gw0/jobfit-app"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue" alt="HF Space" /></a>
        <a href="https://github.com/sponsors/gw0"><img src="https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors" alt="Sponsor" /></a>
      </p>
      <p class="mb-3">A CV/job-post fit-scoring typed-decision model, training pipeline, and web app -- <a href="https://github.com/gw0/jobfit-model">JobFit</a>.</p>
      <p class="mb-5">Paste a CV and job description for calibrated fit scores -- scoring runs entirely in your browser.</p>

      <div class="box">
        <div class="field">
          <label class="label">CV</label>
          <div class="control">
            <textarea class="textarea" rows="8" v-model="cvText" placeholder="Paste CV text..."></textarea>
          </div>
          <div class="control mt-2">
            <input class="input" type="file" accept="application/pdf" @change="onCvFileChange" />
          </div>
        </div>

        <div class="field">
          <label class="label">Job description</label>
          <div class="control">
            <textarea class="textarea" rows="8" v-model="jdText" placeholder="Paste job description, including anything about the company..."></textarea>
          </div>
        </div>

        <div class="field">
          <div class="control">
            <button
              class="button is-primary"
              :class="{ 'is-loading': status === 'loading' || status === 'scoring' }"
              :disabled="!cvText || !jdText || status === 'loading' || status === 'scoring'"
              @click="onSubmit"
            >
              Score fit
            </button>
          </div>
        </div>

        <div v-if="status === 'loading'">
          <p class="has-text-grey">
            Loading model (downloaded once, then cached)...
            <span v-if="download.total">{{ toMb(download.loaded) }} / {{ toMb(download.total) }} MB</span>
          </p>
          <progress class="progress is-primary" :value="download.total ? download.loaded : undefined" :max="download.total || 100"></progress>
        </div>
        <p v-if="status === 'scoring'" class="has-text-grey">Scoring...</p>
        <p v-if="status === 'error'" class="has-text-danger">{{ errorMessage }}</p>
      </div>

      <div v-if="results" class="box">
        <table class="table is-fullwidth is-striped">
          <thead>
            <tr>
              <th>Question</th>
              <th>Score</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in results" :key="row.id">
              <td>{{ row.name }}</td>
              <td>
                <span v-if="row.insufficientData" class="tag is-warning">insufficient data</span>
                <span v-else>{{ row.score.toFixed(1) }} / {{ row.levels - 1 }} <span class="has-text-grey">({{ row.label }})</span></span>
              </td>
              <td>
                <span v-if="!row.insufficientData">{{ row.confidence.toFixed(2) }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>

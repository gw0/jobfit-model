<script setup lang="ts">
import { ref } from "vue";
import { extractPdfText, loadModel, scoreCvJob } from "./infer";
import type { AspectScore } from "./scoring";

const cvText = ref("");
const jdText = ref("");
const results = ref<AspectScore[] | null>(null);
const status = ref<"idle" | "loading" | "scoring" | "error">("idle");
const errorMessage = ref("");
const download = ref({ loaded: 0, total: 0 });

const toMb = (bytes: number) => (bytes / 1_000_000).toFixed(0);

async function onCvFileChange(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  cvText.value = await extractPdfText(file);
}

async function onSubmit() {
  errorMessage.value = "";
  results.value = null;
  try {
    status.value = "loading";
    await loadModel((loaded, total) => {
      download.value = { loaded, total };
    });
    status.value = "scoring";
    results.value = await scoreCvJob(cvText.value, jdText.value);
    status.value = "idle";
  } catch (err) {
    status.value = "error";
    errorMessage.value = err instanceof Error ? err.message : String(err);
  }
}
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">JobFit</h1>
      <p class="subtitle">Paste a CV and job description for calibrated fit scores -- scoring runs entirely in your browser.</p>

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
              <th>Aspect</th>
              <th>Score</th>
              <th>Confidence interval</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in results" :key="row.id">
              <td>{{ row.name }}</td>
              <td>
                <span v-if="row.insufficientData" class="tag is-warning">insufficient data</span>
                <span v-else>{{ row.score.toFixed(2) }}</span>
              </td>
              <td>
                <span v-if="!row.insufficientData">[{{ row.low.toFixed(2) }}, {{ row.high.toFixed(2) }}]</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>

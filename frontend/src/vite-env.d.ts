/// <reference types="vite/client" />

interface ImportMetaEnv {
  // HF Hub model repo id (e.g. "org/jobfit-smollm2-135m-instruct") baked in by scripts/deploy-hf.sh;
  // unset for local serving.
  readonly VITE_HF_MODEL_REPO?: string;
}

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<Record<string, never>, Record<string, never>, unknown>;
  export default component;
}

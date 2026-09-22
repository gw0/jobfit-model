// CV/JD tokenization, identical to pipeline/prepare.py's encode_pair: each side
// truncated independently, JD last, padded to MAX_LENGTH. Plain JS so the Node parity
// script (scripts/verify-parity.mjs) runs the same code as the browser build.
export const CV_BUDGET = 1024;
export const JD_BUDGET = 1024;
export const MAX_LENGTH = 2048;

export function encodePair(tokenizer, cvText, jdText) {
  const cvIds = tokenizer.encode(cvText, { add_special_tokens: false });
  const jdIds = tokenizer.encode(jdText, { add_special_tokens: false });
  const inputIds = [...cvIds.slice(0, CV_BUDGET), ...jdIds.slice(0, JD_BUDGET)].slice(0, MAX_LENGTH);
  const attentionMask = new Array(inputIds.length).fill(1);

  const padTokenId = tokenizer.pad_token_id ?? 0;
  const padLen = MAX_LENGTH - inputIds.length;
  const paddedIds = inputIds.concat(new Array(padLen).fill(padTokenId));
  const paddedMask = attentionMask.concat(new Array(padLen).fill(0));

  return {
    paddedIds,
    paddedMask,
    cvTruncated: cvIds.length > CV_BUDGET,
    jdTruncated: jdIds.length > JD_BUDGET,
  };
}

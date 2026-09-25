"""AI-text detector for aspects #14/#15 (CV / job post likely LLM-generated), applied
once per document instead of an LLM judge (specs §5).

desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, MIT), run locally on CPU. Chosen
over openai-community/roberta-base-openai-detector, which targets GPT-2-era output,
because it leads the RAID benchmark on modern-LLM text. Its accuracy drops on heavily
paraphrased/human-edited AI text, so its scores are labels, not ground truth.
torch/transformers are imported lazily so importing this module stays cheap.
"""

import functools
import threading

MODEL_NAME = "desklib/ai-text-detector-v1.01"
MAX_LEN = 768  # per the model card's own example
# label_dataset.py labels on several threads; one model load, one forward at a time
# (torch already spreads a forward over every core).
_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _get_model_and_tokenizer():
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        """Mean-pooling + linear head + sigmoid, per the model card. The model
        card's own __init__ calls self.init_weights(), which is the pre-4.5x
        transformers convention; current transformers (validated against 5.17)
        requires self.post_init() instead -- init_weights() now assumes
        all_tied_weights_keys is already set, which only post_init() does."""

        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            return {"logits": self.classifier(pooled_output)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DesklibAIDetectionModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def score_llm_generated(text):
    """Return the detector's raw P(machine-generated) in [0,1]."""
    with _LOCK:
        return _score(text)


def _score(text):
    import torch

    model, tokenizer = _get_model_and_tokenizer()
    # padding=True (pad to this input's own length), not "max_length": the model
    # card's own example always pads to 768 even for a one-sentence input, which
    # measured ~6.5x slower on CPU for no accuracy benefit at batch size 1 -- the
    # model attends over the same padded length regardless of content length.
    encoded = tokenizer(
        text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])["logits"]
    return torch.sigmoid(logits).item()

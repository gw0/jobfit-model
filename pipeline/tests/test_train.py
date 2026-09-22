"""Unit tests for train.py's Trainer plumbing, with a tiny stand-in model. Skipped
without torch/transformers (CI)."""

import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import common  # noqa: E402
import train  # noqa: E402


class _TinyQuantileModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, common.NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(logits=self.head(input_ids.float() * attention_mask))


def _cache(n):
    return {"pairs": [{"cv": f"cvs/{i}.md", "job": "jobs/co/1.md"} for i in range(n)],
            "input_ids": torch.arange(4 * n).reshape(n, 4), "attention_mask": torch.ones(n, 4, dtype=torch.long)}


def test_make_dataset_rows():
    rows = train.make_dataset(_cache(3), np.full((3, common.NUM_ASPECTS), 0.5), np.ones((3, common.NUM_ASPECTS)))
    assert len(rows) == 3
    assert set(rows[0]) == {"input_ids", "attention_mask", *train.LABEL_NAMES}
    assert rows[0]["targets"].dtype == torch.float32


def test_compute_loss_is_the_pinball_loss(tmp_path):
    model = _TinyQuantileModel()
    trainer = train.QuantileTrainer(
        model=model, args=__import__("transformers").TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        confidence_floor=0.3,
    )
    inputs = {"input_ids": torch.arange(8).reshape(2, 4), "attention_mask": torch.ones(2, 4),
              "targets": torch.full((2, common.NUM_ASPECTS), 0.5), "confidences": torch.ones(2, common.NUM_ASPECTS)}
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    expected = common.pinball_loss(common.reshape_quantile_logits(outputs.logits),
                                   inputs["targets"], inputs["confidences"], 0.3)
    assert torch.isclose(loss, expected)


def test_compute_metrics_reports_mid_quantile_mae():
    n = 2
    logits = np.tile(np.array([0.0, 0.7, 1.0]), (n, common.NUM_ASPECTS))
    eval_pred = types.SimpleNamespace(predictions=logits, label_ids=(np.full((n, common.NUM_ASPECTS), 0.2),
                                                                     np.ones((n, common.NUM_ASPECTS))))
    result = train.make_compute_metrics(0.3)(eval_pred)
    assert math.isclose(result["mae"], 0.5)

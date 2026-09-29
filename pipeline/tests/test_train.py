"""Unit tests for train.py's Trainer plumbing, with a tiny stand-in model. Skipped
without torch/transformers (CI)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import common  # noqa: E402
import jev_model  # noqa: E402
import train  # noqa: E402

Q = common.NUM_QUESTIONS


class _TinyJevModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 10)

    def forward(self, input_ids, segment_ids, answer_positions, candidate_ids):
        return self.head(input_ids.float())[:, None].expand(-1, answer_positions.shape[1], -1)


def _cache(n):
    return {"pairs": [{"cv": f"cvs/{i}.md", "job": "jobs/co/1.md"} for i in range(n)],
            "input_ids": torch.arange(4 * n).reshape(n, 4), "segment_ids": torch.zeros(n, 4, dtype=torch.long),
            "answer_positions": torch.full((n, Q), 3, dtype=torch.long)}


def test_make_dataset_rows():
    targets = np.full((3, Q), 0.5)
    targets[0, 0] = np.nan
    rows = train.make_dataset(_cache(3), targets, np.ones((3, Q)), 0.3)
    assert len(rows) == 3
    assert set(rows[0]) == {*common.MODEL_INPUTS, *train.LABEL_NAMES}
    assert rows[0]["targets"].shape == (Q, 10) and rows[0]["weights"][0] == 0.0 and rows[1]["weights"][0] == 1.0


def test_compute_loss_is_the_answer_loss(tmp_path):
    model = _TinyJevModel()
    counts = [5] * Q
    trainer = train.JevTrainer(
        model=model, args=__import__("transformers").TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        candidate_ids=torch.zeros(Q, 10, dtype=torch.long), candidate_counts=counts,
    )
    rows = train.make_dataset(_cache(2), np.full((2, Q), 0.5), np.ones((2, Q)), 0.3)
    inputs = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    expected = jev_model.answer_loss(outputs["answer_logits"], inputs["targets"], inputs["weights"],
                                     jev_model.candidate_mask(counts))
    assert torch.isclose(loss, expected)


class _CausalModel(torch.nn.Module):
    """Each position sees only the tokens up to it, like the real decoder."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(10))

    def forward(self, input_ids, segment_ids, answer_positions, candidate_ids):
        seen = input_ids.float().cumsum(1).gather(1, answer_positions)
        return seen[..., None] * self.scale


def test_compute_loss_drops_trailing_padding(tmp_path):
    model, counts = _CausalModel(), [5] * Q
    trainer = train.JevTrainer(
        model=model, args=__import__("transformers").TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        candidate_ids=torch.zeros(Q, 10, dtype=torch.long), candidate_counts=counts,
    )
    rows = train.make_dataset(_cache(2), np.full((2, Q), 0.5), np.ones((2, Q)), 0.3)
    inputs = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((2, 6), 99)], dim=1)  # padding after the answers
    inputs["segment_ids"] = torch.cat([inputs["segment_ids"], torch.full((2, 6), -1)], dim=1)
    full = model(inputs["input_ids"], inputs["segment_ids"], inputs["answer_positions"], None)
    expected = jev_model.answer_loss(full, inputs["targets"], inputs["weights"], jev_model.candidate_mask(counts))
    assert torch.isclose(trainer.compute_loss(model, inputs), expected)


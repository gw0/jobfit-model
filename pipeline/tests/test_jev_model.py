"""jev_model.py on tiny random Qwen3/Llama backbones: one packed pass must equal running
each question branch alone after the state. Skipped without torch/transformers (CI)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import jev  # noqa: E402
import jev_model  # noqa: E402

TINY = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2)


def _lm(kind, attn):
    torch.manual_seed(0)
    if kind == "qwen3":
        config = transformers.Qwen3Config(head_dim=8, **TINY)
        lm = transformers.Qwen3ForCausalLM(config)
    else:
        lm = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    lm.config._attn_implementation = attn
    return lm.eval()


STATE = [5, 6, 7, 8, 9, 10]
BRANCHES = [[11, 12, 13], [14, 15, 16, 17], [18, 19]]
CANDIDATES = torch.tensor([[1, 2, 3] + [0] * 7, [4, 5, 0] + [0] * 7, [6, 7, 8] + [0] * 7])


def _packed(pad=3):
    ids, seg, answers = list(STATE), [0] * len(STATE), []
    for k, branch in enumerate(BRANCHES, start=1):
        ids += branch
        seg += [k] * len(branch)
        answers.append(len(ids) - 1)
    return (torch.tensor([ids + [0] * pad]), torch.tensor([seg + [-1] * pad]), torch.tensor([answers]))


@pytest.mark.parametrize("kind", ["qwen3", "llama"])
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_pass_equals_each_branch_alone(kind, attn):
    model = jev_model.JevModel(_lm(kind, attn))
    assert model.isolate_branches
    with torch.no_grad():
        packed = model(*_packed(), CANDIDATES)[0]
        for k, branch in enumerate(BRANCHES):
            alone = model(torch.tensor([STATE + branch]),
                          torch.tensor([[0] * len(STATE) + [1] * len(branch)]),
                          torch.tensor([[len(STATE) + len(branch) - 1]]), CANDIDATES[k:k + 1])[0, 0]
            torch.testing.assert_close(packed[k], alone, atol=1e-5, rtol=1e-5)


def test_hybrid_backbone_runs_one_plain_causal_pass():
    lm = _lm("qwen3", "sdpa")
    lm.config.layer_types = ["linear_attention", "full_attention"]
    model = jev_model.JevModel(lm)
    assert not model.isolate_branches
    ids, seg, answers = _packed()
    with torch.no_grad():
        logits = lm(input_ids=ids, attention_mask=(seg >= 0).long()).logits[0, answers[0]]
        expected = logits.gather(-1, CANDIDATES)
        torch.testing.assert_close(model(ids, seg, answers, CANDIDATES)[0], expected)


def test_answer_loss_ignores_padded_candidates_and_unweighted_targets():
    logits = torch.tensor([[[2.0, 0.0, 99.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss = jev_model.answer_loss(logits, targets, torch.tensor([[1.0, 0.0]]), mask)
    assert torch.isclose(loss, -torch.log_softmax(torch.tensor([2.0, 0.0]), 0)[0])
    loss.backward()
    assert jev_model.answer_loss(logits, targets, torch.zeros(1, 2), mask).item() == 0.0


def test_candidate_mask():
    mask = jev_model.candidate_mask([2, 5])
    assert mask.shape == (2, jev.MAX_CANDIDATES)
    assert mask.sum(-1).tolist() == [2, 5]


class _WordTokenizer:
    """One token per whitespace-separated word, ids 1..63 by first sight."""
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) % 63 + 1) for w in text.split()]


def test_evaluate_answers_every_question_type():
    model = jev_model.JevModel(_lm("qwen3", "sdpa"))
    questions = {"s": {"type": "score", "question": "How good?", "criteria": ["bad", "ok", "good"]},
                 "c": {"type": "choice", "question": "Which?", "criteria": {"x": "first", "y": "second"}},
                 "n": {"type": "noul", "question": "Is it?", "criteria": {"true": "yes", "false": "no"}}}
    answers = jev_model.evaluate(model, _WordTokenizer(), {"CV": "python dev", "Job": "python role"}, questions,
                                 state_budget=16, questions_budget=64, temperature=2.0)
    assert [a["type"] for a in answers.values()] == ["score", "choice", "noul"]
    assert 0.0 <= answers["s"]["score"] <= 2.0 and answers["c"]["choice"] in ("x", "y")
    assert 0.0 <= answers["n"]["noul"] <= 1.0

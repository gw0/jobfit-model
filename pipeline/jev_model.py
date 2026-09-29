"""Jev model, the torch side (specs/20260923-jev-model.md §2): a causal LM whose own
`lm_head` answers every question branch of an encoded sequence in one forward pass.

The block-diagonal mask is built inside the graph from `segment_ids`, so training and
the exported ONNX model share one implementation and callers only pass 1D inputs: the
state is ordinary causal attention, each branch attends to the state and causally to
itself, never to another branch; each branch's positions restart at the end of the
state. Backbones with non-attention (e.g. linear-attention) layers can't honor such a
mask, so they run as one plain causal pass instead -- branches then also see the
branches before them.
"""

import torch

import jev


def isolates_branches(config):
    """Whether every layer is full attention, so the block-diagonal mask applies."""
    return all(t == "full_attention" for t in getattr(config, "layer_types", None) or ["full_attention"])


class JevModel(torch.nn.Module):
    """forward(input_ids, segment_ids (B, L), answer_positions (B, Q), candidate_ids
    (Q, C)) -> answer_logits (B, Q, C): the next-token logits at each branch's answer
    position, restricted to that question's candidate tokens."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.isolate_branches = isolates_branches(lm.config)

    def forward(self, input_ids, segment_ids, answer_positions, candidate_ids):
        decoder = self.lm.get_decoder()
        if self.isolate_branches:
            idx = torch.arange(input_ids.shape[1], device=input_ids.device)
            causal = idx[None, :] <= idx[:, None]
            seg_q, seg_k = segment_ids[:, :, None], segment_ids[:, None, :]
            visible = causal & ((seg_k == 0) | (seg_k == seg_q))
            dtype = decoder.get_input_embeddings().weight.dtype
            mask = torch.zeros(visible.shape, dtype=dtype, device=input_ids.device)
            mask = mask.masked_fill(~visible, torch.finfo(dtype).min)[:, None]
            hidden = decoder(input_ids=input_ids, attention_mask=mask,
                             position_ids=visible.sum(-1) - 1).last_hidden_state
        else:
            hidden = decoder(input_ids=input_ids, attention_mask=(segment_ids >= 0).long()).last_hidden_state
        at_answers = hidden.gather(1, answer_positions[..., None].expand(-1, -1, hidden.shape[-1]))
        # Only the candidate rows of lm_head: no full-vocabulary logits, and a tied head
        # stays one (embedding) weight in the exported graph.
        head = self.lm.get_output_embeddings()
        logits = torch.einsum("bqh,qch->bqc", at_answers, head.weight[candidate_ids])
        return logits if head.bias is None else logits + head.bias[candidate_ids]


def candidate_mask(counts, device=None):
    """(Q, MAX_CANDIDATES) bool: which candidate slots are real, per question."""
    return torch.arange(jev.MAX_CANDIDATES, device=device)[None, :] < torch.tensor(counts, device=device)[:, None]


def answer_loss(answer_logits, targets, weights, mask):
    """Weighted soft cross-entropy of each question's candidate softmax against its
    target distribution. answer_logits/targets: (B, Q, C); weights: (B, Q), zero where
    a target is missing; mask: (Q, C) real candidate slots. Zero if nothing is weighted."""
    logits = answer_logits.masked_fill(~mask, torch.finfo(answer_logits.dtype).min)
    ce = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)
    return (ce * weights).sum() / weights.sum().clamp_min(1e-12)


def evaluate(model, tokenizer, state, questions, state_budget, questions_budget, temperature=1.0):
    """The in-process Jev call: {id: answer} for one state."""
    encoded = jev.encode(tokenizer, state, questions, state_budget, questions_budget, tokenizer.pad_token_id)
    ids, _ = jev.candidate_ids(tokenizer, questions)
    with torch.no_grad():
        logits = model(torch.tensor([encoded["input_ids"]]), torch.tensor([encoded["segment_ids"]]),
                       torch.tensor([encoded["answer_positions"]]), torch.tensor(ids))
    return jev.answers(questions, logits[0].float().numpy(), temperature)

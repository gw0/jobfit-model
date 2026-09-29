"""Jev-shaped typed-decision model, the generic core (specs/20260923-jev-model.md): a
`state` plus typed `questions` (score / choice / noul) in, one typed answer per question
id out. Pure Python/numpy -- rendering, encoding, candidate tokens, training targets,
readout and temperature fitting. The torch side is jev_model.py; frontend/src/jev.mjs
mirrors this module exactly.

The state is an ordered {title: text} mapping, rendered as Markdown sections; every
question becomes a branch off it, ending in an answer prompt whose next-token
distribution over that question's candidate tokens is the answer.
"""

import math

import numpy as np

TYPES = ("score", "choice", "noul")
MAX_CANDIDATES = 10  # Jev's own limit for score levels; also caps choice options
SEPARATOR = "\n\n---\n\n"
ANSWER_PROMPT = "# Answer\n"
CHOICE_LETTERS = "ABCDEFGHIJ"


# --- questions ---------------------------------------------------------------------------

def validate(questions):
    """Raises ValueError unless `questions` ({id: question}) is well-formed."""
    if not questions:
        raise ValueError("no questions")
    for qid, q in questions.items():
        kind, criteria = q.get("type"), q.get("criteria")
        if kind not in TYPES:
            raise ValueError(f"{qid}: type must be one of {TYPES}, got {kind!r}")
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"{qid}: missing question text")
        if kind == "score" and not (isinstance(criteria, list) and 2 <= len(criteria) <= MAX_CANDIDATES):
            raise ValueError(f"{qid}: score criteria must be a list of 2-{MAX_CANDIDATES} levels")
        if kind == "choice" and not (isinstance(criteria, dict) and 2 <= len(criteria) <= MAX_CANDIDATES):
            raise ValueError(f"{qid}: choice criteria must be an object of 2-{MAX_CANDIDATES} options")
        if kind == "noul" and not (isinstance(criteria, dict) and set(criteria) == {"true", "false"}):
            raise ValueError(f"{qid}: noul criteria must be exactly {{true, false}}")


def answer_keys(question):
    """The answer keys, in candidate order: level indices, option keys, or true/false."""
    if question["type"] == "score":
        return [str(k) for k in range(len(question["criteria"]))]
    if question["type"] == "choice":
        return list(question["criteria"])
    return ["true", "false"]


def candidates(question):
    """The candidate answer strings, each of which must be one token after ANSWER_PROMPT."""
    if question["type"] == "choice":
        return list(CHOICE_LETTERS[:len(question["criteria"])])
    return answer_keys(question)


def _descriptions(question):
    criteria = question["criteria"]
    if question["type"] == "score":
        return criteria
    return [criteria[key] for key in answer_keys(question)]


def render_question(question):
    """One question branch, starting with the separator that follows the state."""
    lines = [f"{c} = {d}" for c, d in zip(candidates(question), _descriptions(question))]
    return (f"{SEPARATOR}# Question\n{question['question']}\n\n# Criteria\n"
            + "\n".join(lines) + f"\n\n{ANSWER_PROMPT}")


def render_part_header(title, first):
    return ("" if first else SEPARATOR) + f"# {title}\n"


# --- encoding ------------------------------------------------------------------------------

def candidate_ids(tokenizer, questions):
    """(Q, MAX_CANDIDATES) token ids, zero-padded, plus each question's candidate count.
    Raises ValueError unless every candidate is exactly one token after ANSWER_PROMPT."""
    prompt = tokenizer.encode(ANSWER_PROMPT, add_special_tokens=False)
    ids, counts = [], []
    for qid, q in questions.items():
        row = []
        for c in candidates(q):
            full = tokenizer.encode(ANSWER_PROMPT + c, add_special_tokens=False)
            if len(full) != len(prompt) + 1 or full[:len(prompt)] != prompt:
                raise ValueError(f"{qid}: candidate {c!r} is not a single token after the answer prompt")
            row.append(full[-1])
        ids.append(row + [0] * (MAX_CANDIDATES - len(row)))
        counts.append(len(row))
    return ids, counts


def encode(tokenizer, state, questions, state_budget, questions_budget, pad_token_id=0):
    """Tokenizes `state` ({title: text}, each part header + text, in order) truncated as a
    whole to `state_budget` tokens -- only the last part is cut -- followed by one branch per
    question, padded to state_budget + questions_budget. Returns {"input_ids", "segment_ids"
    (-1 padding, 0 state, k for question k), "answer_positions" (the last token of each
    branch), "truncated" (whether the state was cut)}. Raises ValueError if the questions
    don't fit `questions_budget` -- question text is never truncated."""
    state_ids = []
    for i, (title, text) in enumerate(state.items()):
        state_ids += tokenizer.encode(render_part_header(title, i == 0), add_special_tokens=False)
        state_ids += tokenizer.encode(text, add_special_tokens=False)
    input_ids = state_ids[:state_budget]
    segment_ids = [0] * len(input_ids)

    answer_positions, questions_len = [], 0
    for k, q in enumerate(questions.values(), start=1):
        branch = tokenizer.encode(render_question(q), add_special_tokens=False)
        questions_len += len(branch)
        input_ids += branch
        segment_ids += [k] * len(branch)
        answer_positions.append(len(input_ids) - 1)
    if questions_len > questions_budget:
        raise ValueError(f"questions take {questions_len} tokens, over the {questions_budget} budget")

    pad = state_budget + questions_budget - len(input_ids)
    return {
        "input_ids": input_ids + [pad_token_id] * pad,
        "segment_ids": segment_ids + [-1] * pad,
        "answer_positions": answer_positions,
        "truncated": len(state_ids) > state_budget,
    }


# --- training targets --------------------------------------------------------------------

def target(question, label):
    """The training distribution over a question's candidates (zero-padded to
    MAX_CANDIDATES) for a label on the answer's own scale: a fractional level index
    (score, two-hot so its expected index is the label exactly), an option key (choice)
    or P(true) (noul)."""
    dist = np.zeros(MAX_CANDIDATES)
    if question["type"] == "score":
        top = len(question["criteria"]) - 1
        t = min(max(float(label), 0.0), float(top))
        low = math.floor(t)
        dist[low] = 1.0 - (t - low)
        dist[min(low + 1, top)] += t - low
    elif question["type"] == "choice":
        dist[answer_keys(question).index(label)] = 1.0
    else:
        dist[:2] = float(label), 1.0 - float(label)
    return dist


# --- readout -------------------------------------------------------------------------------

def probabilities(logits, count, temperature=1.0):
    """Softmax over the first `count` candidates of `logits` (..., >= count)."""
    z = np.asarray(logits, dtype=float)[..., :count] / temperature
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def score_stats(probs):
    """(score, confidence) of level distributions (..., K): the expected level index,
    and 1 - sd/sd_max with sd_max = (K-1)/2, the spread of half the mass on each end."""
    levels = np.arange(probs.shape[-1])
    score = (probs * levels).sum(-1)
    sd = np.sqrt((probs * (levels - score[..., None]) ** 2).sum(-1))
    return score, 1.0 - sd / ((probs.shape[-1] - 1) / 2)


def answer(question, logits, temperature=1.0):
    """One Jev answer from a question's candidate logits."""
    keys = answer_keys(question)
    probs = probabilities(logits, len(keys), temperature)
    if question["type"] == "score":
        score, confidence = score_stats(probs)
        return {"type": "score", "score": float(score), "confidence": float(confidence),
                "probabilities": dict(zip(keys, probs.tolist()))}
    if question["type"] == "choice":
        k = len(keys)
        return {"type": "choice", "choice": keys[int(probs.argmax())],
                "confidence": float((k * probs.max() - 1) / (k - 1)),
                "probabilities": dict(zip(keys, probs.tolist()))}
    return {"type": "noul", "noul": float(probs[0])}


def answers(questions, answer_logits, temperature=1.0):
    """{id: answer} from one example's (Q, MAX_CANDIDATES) answer logits."""
    return {qid: answer(q, row, temperature) for (qid, q), row in zip(questions.items(), answer_logits)}


# --- calibration ---------------------------------------------------------------------------

def fit_temperature(logits, targets, weights):
    """The one temperature minimizing the weighted NLL of softmax(logits / T) against
    the target distributions, pooled over every question -- a model-level parameter
    that applies to any question. Arguments are lists with one entry per question:
    (N, K) logits, (N, K) targets, (N,) weights. NaN with no weight at all."""
    from scipy.optimize import minimize_scalar

    total = sum(float(np.sum(w)) for w in weights)
    if total <= 0:
        return math.nan

    def nll(log_t):
        loss = 0.0
        for z, t, w in zip(logits, targets, weights):
            logp = np.log(np.clip(probabilities(z, z.shape[-1], math.exp(log_t)), 1e-12, None))
            loss -= float(np.sum(w * (t * logp).sum(-1)))
        return loss / total

    return float(math.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x))

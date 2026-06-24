"""Eval-set construction logic: deterministic split, prompts, validation gates.

This module holds the *pure* logic behind ``scripts/build_eval_set.py`` so it
can be unit-tested without a network call or an LLM:

* :func:`select_eval_docs`  — deterministic, per-modality hold-out of source docs.
* :func:`build_qa_prompt`   — modality-aware generation prompt.
* :func:`parse_qa_response` — tolerant JSON/heuristic parser of the LLM reply.
* :func:`validate_qa`       — automatic validation gates (the "semi-automatic"
  half: the LLM proposes, these gates dispose).

No-leakage policy (enforced + documented)
-----------------------------------------
Retrieval can only be scored if the gold document is in the index, so the gold
documents **are** part of the retrieval corpus — excluding them would force every
retrieval metric to 0.  "No leakage between the index split and the eval split"
is therefore enforced as *process* guarantees rather than by withholding docs:

1. **Deterministic hold-out of question sources.** A seeded, per-modality subset
   of documents is chosen as eval question *sources*; the rest act as
   distractors.  The assignment is recorded in ``split.json`` and is stable
   across rebuilds (same seed ⇒ same roles).
2. **Per-document generation isolation.** Each question is generated from exactly
   one source document; the generator never sees other documents, so the eval
   set cannot encode cross-corpus structure.
3. **Verbatim-leak guard.** A question is rejected if it copies a long contiguous
   span from its source — this prevents trivially-retrievable questions that
   would inflate BM25/dense scores.
4. **Grounding gate.** The ground-truth answer must be supported by the source
   text, so we never ship a fabricated gold answer.
5. **No reverse contamination.** Generated questions/answers are never written
   back into the corpus/manifest; the index is built from source docs only.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

# ── text normalisation ──────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase and split *text* into alphanumeric word tokens."""
    return _WORD_RE.findall(text.lower())


def normalize_str(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    return " ".join(normalize_tokens(text))


# ── deterministic split ─────────────────────────────────────────────────────────


@dataclass
class SplitResult:
    """Outcome of :func:`select_eval_docs`."""

    eval_doc_ids: list[str] = field(default_factory=list)
    index_doc_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "counts": self.counts,
            "n_eval": len(self.eval_doc_ids),
            "n_index_pool": len(self.index_doc_ids),
            "eval_doc_ids": self.eval_doc_ids,
            "index_doc_ids": self.index_doc_ids,
            "note": (
                "eval_doc_ids are question sources; ALL documents (eval + index "
                "pool) are indexed for retrieval — see eval_dataset module docstring."
            ),
        }


def select_eval_docs(
    docs: list[dict[str, Any]],
    per_modality: dict[str, int],
    seed: int = 1234,
) -> SplitResult:
    """Deterministically select eval question-source docs, per modality.

    Selection is reproducible: documents are sorted by ``id`` then shuffled with
    a fixed seed, so the same corpus + seed always yields the same hold-out.

    Args:
        docs: Corpus documents; each must have ``id`` and ``modality``.
        per_modality: Desired number of eval source docs per modality, e.g.
            ``{"text": 30, "image": 28, "audio": 22}`` (capped by availability).
        seed: RNG seed controlling the shuffle.

    Returns:
        A :class:`SplitResult` with eval / index-pool doc-id lists and per-modality
        realised counts.
    """
    rng = random.Random(seed)
    eval_ids: list[str] = []
    counts: dict[str, int] = {}

    by_modality: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_modality.setdefault(str(d.get("modality", "unknown")), []).append(d)

    for modality, want in per_modality.items():
        pool = sorted(by_modality.get(modality, []), key=lambda d: str(d["id"]))
        rng.shuffle(pool)
        chosen = pool[: max(0, want)]
        eval_ids.extend(str(d["id"]) for d in chosen)
        counts[modality] = len(chosen)

    eval_set = set(eval_ids)
    index_ids = [str(d["id"]) for d in docs if str(d["id"]) not in eval_set]
    return SplitResult(eval_doc_ids=eval_ids, index_doc_ids=index_ids, counts=counts, seed=seed)


# ── generation prompt ───────────────────────────────────────────────────────────

_MODALITY_INTRO = {
    "text": "The following is a text passage.",
    "image": "The following is a caption describing an image.",
    "audio": "The following is a transcript of an audio clip.",
}

_SYSTEM_PROMPT = (
    "You create evaluation data for a retrieval system. Given a single source "
    "document, write ONE specific, factual question that can be answered using "
    "ONLY that document, plus a SHORT answer (a few words) grounded in the "
    "document. Do not copy a long sentence from the document into the question. "
    'Return STRICT JSON only: {"question": "...", "answer": "..."}'
)


def build_qa_prompt(doc: dict[str, Any]) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for QA generation from *doc*."""
    modality = str(doc.get("modality", "text"))
    intro = _MODALITY_INTRO.get(modality, _MODALITY_INTRO["text"])
    text = str(doc.get("text", "")).strip()
    user = f"{intro}\n\n\"\"\"\n{text}\n\"\"\"\n\nReturn the JSON now."
    return _SYSTEM_PROMPT, user


# ── response parsing ────────────────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_Q_RE = re.compile(r'"?question"?\s*[:=]\s*"?(.+?)"?\s*(?:,|\n|$)', re.IGNORECASE)
_A_RE = re.compile(r'"?answer"?\s*[:=]\s*"?(.+?)"?\s*(?:,|\n|\}|$)', re.IGNORECASE)


def parse_qa_response(raw: str) -> dict[str, str] | None:
    """Parse an LLM reply into ``{"question", "answer"}`` or ``None``.

    Tries strict JSON first (on the first ``{...}`` block), then falls back to
    line-based ``question:``/``answer:`` heuristics.
    """
    if not raw or not raw.strip():
        return None

    match = _JSON_OBJ_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
            if q and a:
                return {"question": q, "answer": a}
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    q_match = _Q_RE.search(raw)
    a_match = _A_RE.search(raw)
    if q_match and a_match:
        q = q_match.group(1).strip().strip('"')
        a = a_match.group(1).strip().strip('"')
        if q and a:
            return {"question": q, "answer": a}
    return None


# ── validation gates ────────────────────────────────────────────────────────────


def answer_is_grounded(answer: str, source_text: str, min_overlap: float = 0.6) -> bool:
    """True if *answer* is supported by *source_text*.

    Grounded when the normalised answer is a substring of the source, or when at
    least *min_overlap* of its unique tokens appear in the source.
    """
    ans_tokens = normalize_tokens(answer)
    if not ans_tokens:
        return False
    src_norm = normalize_str(source_text)
    if normalize_str(answer) and normalize_str(answer) in src_norm:
        return True
    src_tokens = set(normalize_tokens(source_text))
    unique = set(ans_tokens)
    overlap = len(unique & src_tokens) / len(unique)
    return overlap >= min_overlap


def question_leaks_verbatim(question: str, source_text: str, max_span_words: int = 8) -> bool:
    """True if *question* copies a contiguous span of ≥ *max_span_words* words
    verbatim from *source_text* (a trivial-retrieval leak)."""
    q_tokens = normalize_tokens(question)
    if len(q_tokens) < max_span_words:
        return False
    src_tokens = normalize_tokens(source_text)
    src_grams = {
        tuple(src_tokens[i : i + max_span_words])
        for i in range(len(src_tokens) - max_span_words + 1)
    }
    for i in range(len(q_tokens) - max_span_words + 1):
        if tuple(q_tokens[i : i + max_span_words]) in src_grams:
            return True
    return False


def validate_qa(
    question: str,
    answer: str,
    source_text: str,
    *,
    min_question_words: int = 3,
    max_answer_chars: int = 300,
    min_overlap: float = 0.6,
    max_span_words: int = 8,
) -> tuple[bool, list[str]]:
    """Run all automatic validation gates on a generated QA pair.

    Returns:
        ``(ok, reasons)`` where *reasons* lists every gate that failed (empty
        when ``ok`` is True).
    """
    reasons: list[str] = []
    q, a = question.strip(), answer.strip()

    if len(normalize_tokens(q)) < min_question_words:
        reasons.append("question_too_short")
    if not a:
        reasons.append("empty_answer")
    elif len(a) > max_answer_chars:
        reasons.append("answer_too_long")
    if normalize_str(q) == normalize_str(a) and a:
        reasons.append("question_equals_answer")
    if a and not answer_is_grounded(a, source_text, min_overlap=min_overlap):
        reasons.append("answer_not_grounded")
    if question_leaks_verbatim(q, source_text, max_span_words=max_span_words):
        reasons.append("question_verbatim_leak")

    return (len(reasons) == 0, reasons)


__all__ = [
    "normalize_tokens",
    "normalize_str",
    "SplitResult",
    "select_eval_docs",
    "build_qa_prompt",
    "parse_qa_response",
    "answer_is_grounded",
    "question_leaks_verbatim",
    "validate_qa",
]

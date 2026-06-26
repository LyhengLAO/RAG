"""Build the evaluation QA set under ``data/eval/`` from the indexed corpus.

Semi-automatic pipeline
-----------------------
1. Read the corpus manifest (``data/processed/manifest.jsonl``).
2. Deterministically hold out a per-modality subset of documents as question
   *sources* (seeded — reproducible; recorded in ``data/eval/split.json``).
3. For each source document, ask the **local Ollama** LLM to propose one
   factual ``{question, answer}`` grounded only in that document.
4. Run automatic validation gates (grounding, verbatim-leak guard, length,
   shape).  Passing pairs go to ``eval_set.jsonl``; every attempt (with its
   verdict) goes to ``candidates.jsonl`` for human review.

No-leakage guarantees are documented in
:mod:`src.evaluation.eval_dataset` and surfaced in ``split.json``.

Usage
-----
    python scripts/build_eval_set.py                       # defaults (~80 targets)
    python scripts/build_eval_set.py --n-text 30 --n-image 28 --n-audio 22
    python scripts/build_eval_set.py --max-per-doc 1 --seed 1234

Requires Ollama running (``ollama serve``) with the judge model pulled. The
script refuses to fabricate questions: if generation/parse/validation fails for
a document it is logged and skipped — never replaced by a template.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.config import settings
from src.evaluation.eval_dataset import (
    build_qa_prompt,
    parse_qa_response,
    select_eval_docs,
    validate_qa,
)
from src.generation.llm import LLMClient
from src.ingestion.schema import RawDocument

app = typer.Typer(add_completion=False, help="Build the mmrag evaluation QA set (data/eval/).")
logger = logging.getLogger(__name__)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(RawDocument.from_json(line).to_dict())
            except Exception as exc:  # noqa: BLE001
                logger.warning("manifest line %d skipped — %s", lineno, exc)
    return docs


@app.command()
def main(
    manifest: Path = typer.Option(Path("data/processed/manifest.jsonl"), help="Corpus manifest."),
    out_dir: Path = typer.Option(Path("data/eval"), help="Output directory for the eval set."),
    n_text: int = typer.Option(30, help="Eval question-source docs for text."),
    n_image: int = typer.Option(28, help="Eval question-source docs for image."),
    n_audio: int = typer.Option(22, help="Eval question-source docs for audio."),
    max_per_doc: int = typer.Option(1, help="Questions generated per source document."),
    seed: int = typer.Option(1234, help="Deterministic split seed."),
    judge_model: str = typer.Option(None, help="Ollama model (defaults to settings.ollama_model)."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Generate ``data/eval/eval_set.jsonl`` + ``candidates.jsonl`` + ``split.json``."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not manifest.exists():
        typer.echo(f"ERROR: manifest not found at {manifest}. Run scripts/build_dataset.py first.", err=True)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    docs = _load_manifest(manifest)
    if not docs:
        typer.echo("ERROR: empty manifest.", err=True)
        raise typer.Exit(code=1)
    by_id = {str(d["id"]): d for d in docs}

    # ── deterministic, documented split ────────────────────────────────────────
    split = select_eval_docs(docs, {"text": n_text, "image": n_image, "audio": n_audio}, seed=seed)
    (out_dir / "split.json").write_text(
        json.dumps(split.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"Split: {split.counts} eval source docs (seed={seed}) → {out_dir/'split.json'}")

    # ── LLM judge/generator (local Ollama) ─────────────────────────────────────
    model = judge_model or settings.ollama_model
    llm = LLMClient(provider="ollama", model=model, temperature=0.2, max_tokens=256)
    typer.echo(f"Generating QA with Ollama model {model!r} at {settings.ollama_host} …")

    eval_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    n_gen_fail = n_parse_fail = n_validate_fail = n_dup = 0

    for i, doc_id in enumerate(split.eval_doc_ids, 1):
        doc = by_id[doc_id]
        source_text = str(doc.get("text", "")).strip()
        if not source_text:
            continue

        for attempt in range(max_per_doc):
            system, user = build_qa_prompt(doc)
            try:
                raw = llm.generate(user, system_prompt=system)
            except RuntimeError as exc:
                logger.error("doc %s: generation failed — %s", doc_id, exc)
                n_gen_fail += 1
                candidates.append({"source_doc_id": doc_id, "modality": doc["modality"],
                                   "status": "gen_failed", "error": str(exc)})
                break  # Ollama down/unreachable — stop hammering it

            parsed = parse_qa_response(raw)
            if not parsed:
                logger.warning("doc %s: unparseable response", doc_id)
                n_parse_fail += 1
                candidates.append({"source_doc_id": doc_id, "modality": doc["modality"],
                                   "status": "parse_failed", "raw": raw[:300]})
                continue

            ok, reasons = validate_qa(parsed["question"], parsed["answer"], source_text)
            qkey = parsed["question"].strip().lower()
            if ok and qkey in seen_questions:
                ok, reasons = False, ["duplicate_question"]
                n_dup += 1

            candidate = {
                "source_doc_id": doc_id,
                "modality": doc["modality"],
                "question": parsed["question"],
                "answer": parsed["answer"],
                "status": "accepted" if ok else "rejected",
                "reasons": reasons,
            }
            candidates.append(candidate)

            if not ok:
                n_validate_fail += 1 if reasons != ["duplicate_question"] else 0
                continue

            seen_questions.add(qkey)
            eval_records.append({
                "id": f"eval_{len(eval_records):04d}",
                "question": parsed["question"],
                "ground_truth_answer": parsed["answer"],
                "relevant_doc_ids": [doc_id],
                "modality": doc["modality"],
                "source": doc.get("source", ""),
                "source_doc_id": doc_id,
                "gen_model": model,
                "validated": True,
            })

        if i % 10 == 0:
            typer.echo(f"  … {i}/{len(split.eval_doc_ids)} source docs processed "
                       f"({len(eval_records)} accepted)")

    # ── write outputs ──────────────────────────────────────────────────────────
    eval_path = out_dir / "eval_set.jsonl"
    with eval_path.open("w", encoding="utf-8") as fh:
        for rec in eval_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cand_path = out_dir / "candidates.jsonl"
    with cand_path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_modality = {m: sum(1 for r in eval_records if r["modality"] == m)
                   for m in ("text", "image", "audio")}
    summary = {
        "accepted": len(eval_records),
        "by_modality": by_modality,
        "candidates": len(candidates),
        "rejected_generation": n_gen_fail,
        "rejected_parse": n_parse_fail,
        "rejected_validation": n_validate_fail,
        "rejected_duplicate": n_dup,
    }
    typer.echo("\n" + "─" * 54)
    typer.echo(json.dumps(summary, indent=2))

    if not eval_records:
        typer.echo(
            "\nERROR: 0 questions accepted. Is Ollama running and the model pulled? "
            "Inspect data/eval/candidates.jsonl for per-document failure reasons.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"\n✓  eval set   → {eval_path}  ({len(eval_records)} questions)")
    typer.echo(f"✓  candidates → {cand_path}  (review before relying on the set)")
    typer.echo(f"✓  split      → {out_dir/'split.json'}")


if __name__ == "__main__":
    app()

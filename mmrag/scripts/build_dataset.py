"""Materialise data/raw/ from HuggingFace and write data/processed/manifest.jsonl.

Usage
-----
    python scripts/build_dataset.py            # defaults
    python scripts/build_dataset.py --n-text 100 --n-image 50 --n-audio 30
    make build-dataset                         # same as defaults via Makefile
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.config import settings
from src.ingestion.loaders import load_audio_clips, load_image_captions, load_text_corpus
from src.ingestion.schema import RawDocument

app = typer.Typer(add_completion=False, help="Build the mmrag evaluation dataset from open sources.")
logger = logging.getLogger(__name__)


def _write_jsonl(docs: list[RawDocument], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.to_json() + "\n")


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Root directory for raw per-modality files"),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Output directory for manifests"),
    n_text: int = typer.Option(500, help="Max text passages to ingest"),
    n_image: int = typer.Option(500, help="Max image+caption pairs to ingest"),
    n_audio: int = typer.Option(300, help="Max audio clips to ingest"),
    seed: int = typer.Option(42, help="Random seed — controls both download shuffle and eval split"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)"),
) -> None:
    """Download open-source datasets and write data/processed/manifest.jsonl.

    Idempotent: per-modality caches under data/raw/<modality>/manifest.jsonl
    prevent re-downloading on subsequent runs. Set HF_DATASETS_OFFLINE=1 to
    force offline mode after the first successful build.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = settings.hf_home if settings.hf_home.exists() else None

    # ── load ────────────────────────────────────────────────────────────────

    typer.echo("── Text (SQuAD / Wikipedia, CC-BY-SA-4.0) ──────────────────")
    text_docs = load_text_corpus(
        n_samples=n_text, seed=seed, cache_dir=raw_dir / "text", hf_cache_dir=hf_cache
    )

    typer.echo("── Images (Flickr30k, Flickr Research-Use) ──────────────────")
    image_docs = load_image_captions(
        n_samples=n_image, seed=seed, cache_dir=raw_dir / "images", hf_cache_dir=hf_cache
    )

    typer.echo("── Audio (LibriSpeech clean, CC-BY-4.0) ─────────────────────")
    audio_docs = load_audio_clips(
        n_samples=n_audio, seed=seed, cache_dir=raw_dir / "audio", hf_cache_dir=hf_cache
    )

    all_docs: list[RawDocument] = text_docs + image_docs + audio_docs

    if not all_docs:
        typer.echo(
            "ERROR: every loader returned an empty list.\n"
            "Check the logs above; likely causes:\n"
            "  • No network access and cache not yet built\n"
            "  • 'datasets' package not installed (pip install datasets)\n"
            "  • HuggingFace Hub throttling — retry in a few minutes",
            err=True,
        )
        raise typer.Exit(code=1)

    # ── write manifest ───────────────────────────────────────────────────────

    manifest_path = processed_dir / "manifest.jsonl"
    _write_jsonl(all_docs, manifest_path)

    # ── deterministic eval split (≈20 %) ────────────────────────────────────
    # Using a separate RNG so eval membership doesn't depend on loader order.
    rng = random.Random(seed + 1)
    eval_docs = [d for d in all_docs if rng.random() < 0.2]
    eval_path = processed_dir / "eval.jsonl"
    _write_jsonl(eval_docs, eval_path)

    # ── summary ──────────────────────────────────────────────────────────────

    by_modality = {m: sum(1 for d in all_docs if d.modality == m) for m in ("text", "image", "audio")}
    summary = {
        "total": len(all_docs),
        "by_modality": by_modality,
        "eval_split": len(eval_docs),
        "manifest": str(manifest_path),
        "eval": str(eval_path),
    }

    typer.echo("\n" + "─" * 54)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"\n✓  manifest → {manifest_path}  ({len(all_docs)} docs)")
    typer.echo(f"✓  eval     → {eval_path}  ({len(eval_docs)} docs)")


if __name__ == "__main__":
    app()

"""Central configuration: pydantic-settings from .env + YAML loader for pipeline configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-level settings resolved from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM backend
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    # HuggingFace
    hf_home: Path = Field(default=Path(".cache/huggingface"))
    hf_token: str | None = Field(default=None)

    # Data paths
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    results_dir: Path = Field(default=Path("results"))

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("chroma_db"))

    # Whisper
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="base"
    )
    whisper_device: Literal["cpu", "cuda"] = Field(default="cpu")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    """Load a pipeline YAML config file and return it as a plain dict.

    Args:
        config_path: Path to a YAML file (e.g. ``configs/baseline.yaml``).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_baseline_config() -> dict[str, Any]:
    """Convenience wrapper that loads ``configs/baseline.yaml``."""
    return load_pipeline_config(Path(__file__).parents[2] / "configs" / "baseline.yaml")


def get_optimized_config() -> dict[str, Any]:
    """Convenience wrapper that loads ``configs/optimized.yaml``."""
    return load_pipeline_config(Path(__file__).parents[2] / "configs" / "optimized.yaml")


# Module-level singleton — import this everywhere
settings = Settings()

"""LLM client abstraction: Ollama (default) or any OpenAI-compatible endpoint.

Uses ``requests`` only — no external SDK required. Ollama must be running
locally (default: http://localhost:11434). OpenAI-compatible endpoints require
``base_url`` and, if authenticated, ``api_key``.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around an Ollama / OpenAI-compatible chat endpoint.

    Args:
        provider:    ``"ollama"`` (default) or ``"openai"`` for any
                     OpenAI-compatible API (vLLM, LM Studio, …).
        model:       Model name on the provider side (e.g. ``"llama3.2"``).
        base_url:    Override the API base URL.  Required for ``"openai"``
                     unless ``OPENAI_BASE_URL`` is set in the environment.
        api_key:     Bearer token.  Falls back to ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature (0 = near-deterministic).
        max_tokens:  Maximum tokens to generate.
        timeout:     HTTP timeout in seconds (generous default: 120 s for CPU).
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # ── token-usage instrumentation ───────────────────────────────────────
        # ``last_usage`` holds the provider-reported counts of the most recent
        # generate() call (or None when the provider returned none).
        # ``total_usage`` accumulates across all calls so an evaluation runner
        # can snapshot it before/after a query to attribute tokens per query
        # (see src.evaluation.system_metrics.token_delta). ``missing`` counts
        # calls that returned no usage, so a measurement gap is never mistaken
        # for genuine zero-token usage.
        self.last_usage: dict[str, int] | None = None
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "missing": 0,
        }

        if base_url:
            self._base_url = base_url.rstrip("/")
        elif provider == "ollama":
            self._base_url = settings.ollama_host.rstrip("/")
        elif provider == "openai" and settings.openai_base_url:
            self._base_url = settings.openai_base_url.rstrip("/")
        else:
            raise ValueError(
                f"base_url is required for provider={provider!r}. "
                "Set OPENAI_BASE_URL in .env or pass base_url= explicitly."
            )

        self._api_key: str | None = api_key or settings.openai_api_key

    # ── internals ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _chat_url(self) -> str:
        if self.provider == "ollama":
            return f"{self._base_url}/api/chat"
        return f"{self._base_url}/v1/chat/completions"

    def _build_messages(
        self, prompt: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _ollama_payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def _openai_payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt and return the complete generated response.

        Args:
            prompt:        User message / RAG prompt.
            system_prompt: Optional system instruction (sent as a ``system``
                           role message before the user message).

        Returns:
            Generated text as a plain string.

        Raises:
            RuntimeError: On HTTP error or unexpected response shape.
        """
        messages = self._build_messages(prompt, system_prompt)

        if self.provider == "ollama":
            payload = self._ollama_payload(messages, stream=False)
        else:
            payload = self._openai_payload(messages, stream=False)

        try:
            resp = requests.post(
                self._chat_url(),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"LLMClient ({self.provider}) request to {self._chat_url()} failed: {exc}"
            ) from exc

        data = resp.json()
        try:
            if self.provider == "ollama":
                content = str(data["message"]["content"])
            else:
                content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from {self.provider}: {data!r}"
            ) from exc

        self._record_usage(data)
        return content

    # ── usage accounting ──────────────────────────────────────────────────────

    def _record_usage(self, data: dict) -> None:
        """Extract provider-reported token counts and update usage counters.

        Ollama returns ``prompt_eval_count`` / ``eval_count``; OpenAI-compatible
        endpoints return a ``usage`` object.  When neither is present the call is
        recorded as ``missing`` (never fabricated as zero).
        """
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if self.provider == "ollama":
            p = data.get("prompt_eval_count")
            c = data.get("eval_count")
            if isinstance(p, int) and isinstance(c, int):
                prompt_tokens, completion_tokens = p, c
        else:
            usage = data.get("usage") or {}
            p = usage.get("prompt_tokens")
            c = usage.get("completion_tokens")
            if isinstance(p, int) and isinstance(c, int):
                prompt_tokens, completion_tokens = p, c

        self.total_usage["calls"] += 1
        if prompt_tokens is None or completion_tokens is None:
            self.total_usage["missing"] += 1
            self.last_usage = None
            logger.debug("LLMClient: provider %s returned no token usage", self.provider)
            return

        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.total_usage["prompt_tokens"] += prompt_tokens
        self.total_usage["completion_tokens"] += completion_tokens

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Stream a response chunk by chunk.

        Args:
            prompt:        User message / RAG prompt.
            system_prompt: Optional system instruction.

        Yields:
            String fragments as they arrive from the API.

        Raises:
            RuntimeError: On HTTP error.
        """
        messages = self._build_messages(prompt, system_prompt)

        if self.provider == "ollama":
            payload = self._ollama_payload(messages, stream=True)
            try:
                with requests.post(
                    self._chat_url(),
                    json=payload,
                    headers=self._headers(),
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        try:
                            chunk = json.loads(raw)
                            text = chunk.get("message", {}).get("content", "")
                            if text:
                                yield text
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            except requests.RequestException as exc:
                raise RuntimeError(f"LLMClient (ollama) stream failed: {exc}") from exc

        else:  # OpenAI SSE format
            payload = self._openai_payload(messages, stream=True)
            try:
                with requests.post(
                    self._chat_url(),
                    json=payload,
                    headers=self._headers(),
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        if line.startswith("data: "):
                            line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            delta = chunk["choices"][0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                yield text
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            except requests.RequestException as exc:
                raise RuntimeError(f"LLMClient (openai) stream failed: {exc}") from exc

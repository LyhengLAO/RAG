"""Prompt templates for RAG generation."""

from __future__ import annotations

from langchain_core.documents import Document

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    'If the answer is not in the context, say "I don\'t know."'
)

_RAG_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


def format_context(documents: list[Document], separator: str = "\n\n---\n\n") -> str:
    """Join document page contents into a single context string.

    Args:
        documents: List of retrieved documents.
        separator: String placed between consecutive documents.

    Returns:
        Concatenated context string.  Empty string when *documents* is empty.
    """
    return separator.join(doc.page_content for doc in documents)


def build_rag_prompt(
    question: str,
    context_docs: list[Document],
    template: str = _RAG_TEMPLATE,
    max_context_chars: int = 4_000,
) -> str:
    """Build a RAG prompt by formatting retrieved context + user question.

    Context is hard-capped at *max_context_chars* characters so the final
    prompt stays within the LLM's context window even when many chunks are
    retrieved.  Truncation is signalled by an inline marker, not silent.

    Args:
        question:         The user's question.
        context_docs:     Retrieved documents to inject as context.
        template:         Format string with ``{context}`` and ``{question}``
                          placeholders.
        max_context_chars: Hard cap on total context length.

    Returns:
        Formatted prompt string ready to pass to :class:`LLMClient`.
    """
    context = format_context(context_docs)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[... context truncated ...]"
    return template.format(context=context, question=question)

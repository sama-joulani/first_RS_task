from __future__ import annotations

from app.utils.text_chunker import Chunk


RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based strictly on the provided context documents.

Rules:
1. ONLY use information from the provided context to answer the question.
2. For every claim you make, add a citation footnote in the format [doc_title, page X].
3. If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question."
4. Be concise and accurate.
5. Support both Arabic and English questions and answers."""

RAG_USER_TEMPLATE = """Context Documents:
{context}

---

Question: {query}

Provide a detailed answer with inline citation footnotes [doc_title, page X] for every claim."""

SEARCH_REFINEMENT_TEMPLATE = """Based on the user query below, generate an improved search query that would better retrieve relevant documents.

Original query: {query}

Improved search query:"""


class PromptManager:
    """Manages structured prompt templates for RAG, search, and citations."""

    def __init__(self):
        self._templates: dict[str, str] = {
            "rag_system": RAG_SYSTEM_PROMPT,
            "rag_user": RAG_USER_TEMPLATE,
            "search_refinement": SEARCH_REFINEMENT_TEMPLATE,
        }

    def get_system_prompt(self) -> str:
        return self._templates["rag_system"]

    def format_rag_prompt(self, query: str, chunks: list[Chunk]) -> str:
        context = self._format_context(chunks)
        return self._templates["rag_user"].format(context=context, query=query)

    def format_search_refinement(self, query: str) -> str:
        return self._templates["search_refinement"].format(query=query)

    def register_template(self, name: str, template: str) -> None:
        self._templates[name] = template

    def get_template(self, name: str) -> str | None:
        return self._templates.get(name)

    def _format_context(self, chunks: list[Chunk]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            meta = chunk.metadata
            title = meta.get("title", "Unknown")
            page = meta.get("page", "?")
            section = meta.get("section", "")
            header = f"[Source: {title}, Page {page}"
            if section:
                header += f", Section: {section}"
            header += "]"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

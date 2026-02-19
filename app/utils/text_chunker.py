from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "。", "،", " ", ""],
        )

    def chunk(self, text: str, base_metadata: dict | None = None) -> list[Chunk]:
        base_metadata = base_metadata or {}
        splits = self._splitter.split_text(text)
        return [
            Chunk(text=s, chunk_index=i, metadata={**base_metadata})
            for i, s in enumerate(splits)
            if s.strip()
        ]

    def chunk_pages(self, pages: list[dict]) -> list[Chunk]:
        """Chunk a list of pages, preserving page-level metadata.

        Each page dict should have 'text', 'page_number', and optionally other keys.
        """
        all_chunks: list[Chunk] = []
        idx = 0
        for page in pages:
            text = page.get("text", "")
            page_meta = {k: v for k, v in page.items() if k != "text"}
            splits = self._splitter.split_text(text)
            for s in splits:
                if s.strip():
                    all_chunks.append(Chunk(text=s, chunk_index=idx, metadata=page_meta))
                    idx += 1
        return all_chunks

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

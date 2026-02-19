from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.utils.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int
    text: str
    is_ocr: bool = False


@dataclass
class ParsedDocument:
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    page_count: int = 0


class PDFParser:
    """Parses PDF files, extracting text and falling back to OCR for image-heavy pages."""

    MIN_TEXT_LENGTH = 50  # Threshold to consider a page as image-only

    def __init__(self, ocr_processor: OCRProcessor | None = None):
        self._ocr = ocr_processor or OCRProcessor()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        reader = PdfReader(str(file_path))
        doc_metadata = self._extract_metadata(reader)
        pages: list[PageContent] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            is_ocr = False

            if len(text.strip()) < self.MIN_TEXT_LENGTH:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    is_ocr = True

            if text.strip():
                pages.append(PageContent(page_number=i + 1, text=text.strip(), is_ocr=is_ocr))

        return ParsedDocument(pages=pages, metadata=doc_metadata, page_count=len(reader.pages))

    def _extract_metadata(self, reader: PdfReader) -> dict:
        meta = reader.metadata or {}
        return {
            "title": getattr(meta, "title", "") or "",
            "author": getattr(meta, "author", "") or "",
            "subject": getattr(meta, "subject", "") or "",
            "creator": getattr(meta, "creator", "") or "",
        }

    def _ocr_page(self, page) -> str:
        try:
            from PIL import Image
            import io

            for image_obj in page.images:
                data = image_obj.data
                image = Image.open(io.BytesIO(data))
                text = self._ocr.process_image(image)
                if text:
                    return text
        except Exception:
            logger.debug("OCR extraction failed for page", exc_info=True)
        return ""

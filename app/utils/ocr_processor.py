from __future__ import annotations

import logging
from pathlib import Path

import pytesseract
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Extracts text from images using Tesseract OCR with Arabic + English support."""

    def __init__(self, language: str = "ara+eng", tesseract_cmd: str | None = None):
        self._language = language
        cmd = tesseract_cmd or get_settings().tesseract_cmd
        if cmd and cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = cmd

    def process_image(self, image: Image.Image) -> str:
        try:
            text = pytesseract.image_to_string(image, lang=self._language)
            return text.strip()
        except pytesseract.TesseractError:
            logger.warning("Tesseract OCR failed for image, returning empty string")
            return ""

    def process_images(self, images: list[Image.Image]) -> list[str]:
        return [self.process_image(img) for img in images]

    @property
    def language(self) -> str:
        return self._language

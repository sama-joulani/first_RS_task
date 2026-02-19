from __future__ import annotations

import logging
from typing import Literal

import requests
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using local sentence-transformers or external API."""

    DIMENSION = 384

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self._service_type: Literal["local", "aragemma"] = settings.embedding_service_type
        self._api_url = settings.embedding_api_url
        self._api_key = settings.deepseek_api_key
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

        if self._service_type == "local":
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        else:
            logger.info("Using external embedding API: %s", self._api_url)

    def embed_text(self, text: str) -> list[float]:
        if self._service_type == "local":
            vector = self._model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        else:
            return self._embed_via_api([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if self._service_type == "local":
            vectors = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
            return vectors.tolist()
        else:
            return self._embed_via_api(texts)

    def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """Call external embedding API."""
        try:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = requests.post(
                self._api_url,
                json={"texts": texts},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"]
        except requests.RequestException as e:
            logger.error("Embedding API request failed: %s", e)
            raise RuntimeError(f"Embedding API request failed: {e}") from e
        except (KeyError, ValueError) as e:
            logger.error("Invalid response from embedding API: %s", e)
            raise RuntimeError(f"Invalid response from embedding API: {e}") from e

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def model_name(self) -> str:
        return self._model_name

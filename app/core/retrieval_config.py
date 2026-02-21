"""Centralized configuration for hybrid retrieval settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class RetrievalSettings(BaseSettings):
    """Configurable retrieval parameters for hybrid search."""
    
    # Dense retrieval settings
    top_k_dense: int = 10
    
    # Keyword/BM25 retrieval settings  
    top_k_keyword: int = 10
    
    # Fusion settings
    fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60  # RRF constant (typical range: 20-100)
    
    # Weighted fusion settings (only used if fusion_method="weighted")
    dense_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Final result count
    final_top_k: int = 5
    
    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()

"""BM25/TF-IDF keyword search implementation for hybrid retrieval."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass

from app.core.retrieval_config import get_retrieval_settings

logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Result from keyword search."""
    chunk_id: str
    chunk_text: str
    score: float
    document_id: str
    title: str
    author: str = ""
    page: int = 0
    section: str = ""
    chunk_index: int = 0
    tags: list[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class BM25Searcher:
    """BM25 keyword search implementation."""
    
    def __init__(self):
        settings = get_retrieval_settings()
        self.k1 = settings.bm25_k1
        self.b = settings.bm25_b
        
        # Corpus statistics
        self.documents: dict[str, dict] = {}  # chunk_id -> {text, metadata, term_freq}
        self.term_doc_freq: dict[str, int] = defaultdict(int)  # term -> document frequency
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.vocab_built: bool = False
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    def _compute_term_freq(self, tokens: list[str]) -> dict[str, int]:
        """Compute term frequency for a document."""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        return dict(tf)
    
    def add_document(self, chunk_id: str, text: str, metadata: dict | None = None):
        """Add a document chunk to the corpus."""
        tokens = self._tokenize(text)
        term_freq = self._compute_term_freq(tokens)
        
        self.documents[chunk_id] = {
            "text": text,
            "tokens": tokens,
            "term_freq": term_freq,
            "length": len(tokens),
            "metadata": metadata or {}
        }
        
        # Update document frequencies
        for term in term_freq.keys():
            self.term_doc_freq[term] += 1
        
        self.vocab_built = False
    
    def build_vocab(self):
        """Build corpus statistics after all documents are added."""
        if not self.documents:
            return
            
        total_length = sum(doc["length"] for doc in self.documents.values())
        self.avg_doc_length = total_length / len(self.documents)
        self.total_docs = len(self.documents)
        self.vocab_built = True
        
        logger.info("BM25 vocabulary built: %d documents, avg length %.2f", 
                   self.total_docs, self.avg_doc_length)
    
    def search(self, query: str, top_k: int = 10) -> list[KeywordResult]:
        """Search documents using BM25 scoring."""
        if not self.vocab_built:
            self.build_vocab()
        
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        scores = {}
        
        for chunk_id, doc in self.documents.items():
            score = self._bm25_score(query_tokens, doc)
            if score > 0:
                scores[chunk_id] = score
        
        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in ranked:
            doc = self.documents[chunk_id]
            meta = doc["metadata"]
            results.append(KeywordResult(
                chunk_id=chunk_id,
                chunk_text=doc["text"],
                score=score,
                document_id=meta.get("document_id", ""),
                title=meta.get("title", ""),
                author=meta.get("author", ""),
                page=meta.get("page", 0),
                section=meta.get("section", ""),
                chunk_index=meta.get("chunk_index", 0),
                tags=meta.get("tags", [])
            ))
        
        return results
    
    def _bm25_score(self, query_tokens: list[str], doc: dict) -> float:
        """Calculate BM25 score for a document given query tokens."""
        score = 0.0
        doc_length = doc["length"]
        term_freq = doc["term_freq"]
        
        for term in query_tokens:
            if term not in term_freq:
                continue
            
            # IDF calculation
            df = self.term_doc_freq.get(term, 0)
            if df == 0:
                continue
            
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # Term frequency component
            tf = term_freq[term]
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))
            
            score += idf * tf_component
        
        return score
    
    def clear(self):
        """Clear all indexed documents."""
        self.documents.clear()
        self.term_doc_freq.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.vocab_built = False

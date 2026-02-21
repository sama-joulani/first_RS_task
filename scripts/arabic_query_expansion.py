"""Arabic query expansion and enhancement for improved retrieval accuracy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """Query with expansions for better retrieval."""
    original: str
    expanded: str
    variations: List[str]
    expansion_method: str


class ArabicQueryExpander:
    """
    Arabic query expansion using multiple techniques:
    1. Normalization (remove tashkeel, normalize alef variants)
    2. Synonym expansion
    3. Root-based expansion
    4. Transliteration handling
    5. Common Arabic business terms
    """
    
    # Arabic normalization mappings
    ALEF_VARIANTS = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا'
    }
    
    TASHKEEL_CHARS = 'َُِّْٕٖٜٟٓٔٗ٘ٙٚٛٝٞ'
    
    # Business/technical synonym mappings (Arabic)
    SYNONYMS = {
        # Company related
        'شركة': ['مؤسسة', 'منظمة', 'فريق', 'عمل'],
        'حلول': ['منتجات', 'خدمات', 'أنظمة', 'تطبيقات'],
        'عملاء': ['زبائن', 'مستخدمين', 'مستفيدين'],
        'فريق': ['طاقم', 'كادر', 'موظفين'],
        
        # Technology related
        'ذكاء اصطناعي': ['AI', 'machine learning', 'تعلم آلي', 'تعلم الالة'],
        'تحول رقمي': ['digital transformation', 'رقمنة', 'أتمتة'],
        'تطبيق': ['برنامج', 'نظام', 'software', 'app'],
        'بيانات': ['data', 'معلومات', 'داتا'],
        
        # Services related
        'تدريب': ['تعليم', 'تأهيل', 'courses', 'دورات'],
        'دعم': ['مساعدة', 'مساندة', 'support', 'خدمة'],
        'استشارة': ['consulting', 'مشورة', 'استشارات'],
        
        # Quality/Standards
        'معايير': ['standards', 'مواصفات', 'قواعد'],
        'جودة': ['quality', 'تميز', 'كفاءة'],
        'اعتماد': ['accreditation', 'شهادة', 'ترخيص'],
        
        # Contact/Location
        'فرع': ['مكتب', 'موقع', 'office', 'branch'],
        'تواصل': ['اتصال', 'مراسلة', 'contact'],
        'عنوان': ['address', 'موقع', 'location'],
        
        # Products specific to RealSoft
        'اداء': ['adaa', 'performance', 'أداء'],
        'تعداد': ['census', 'إحصاء', 'احصاء'],
        'مصرف': ['bank', 'بنك', 'مصارف', 'بنوك'],
    }
    
    # Common Arabic-English mixed terms in business
    MIXED_TERMS = {
        'ريلسوفت': ['realsoft', 'ريل سوفت', 'real soft'],
        'الخوارزمي': ['alkhwarizmi', 'al-khwarizmi', 'خوارزمي'],
        'فالكون': ['falcon', 'falconmap'],
    }
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for efficiency."""
        return {
            'tashkeel': re.compile(f'[{self.TASHKEEL_CHARS}]'),
            'alef': re.compile('|'.join(self.ALEF_VARIANTS.keys())),
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text:
        - Remove tashkeel (diacritics)
        - Normalize alef variants
        - Convert to lowercase for any English
        """
        # Remove tashkeel
        text = self.compiled_patterns['tashkeel'].sub('', text)
        
        # Normalize alef variants
        for variant, standard in self.ALEF_VARIANTS.items():
            text = text.replace(variant, standard)
        
        return text.lower().strip()
    
    def expand_synonyms(self, query: str) -> List[str]:
        """Generate synonym expansions for query terms."""
        normalized = self.normalize(query)
        words = normalized.split()
        
        expansions = [query]  # Always include original
        
        for word in words:
            # Check for multi-word keys first
            for key, synonyms in self.SYNONYMS.items():
                if key in normalized or any(s in normalized for s in [key.replace(' ', '')]):
                    for syn in synonyms:
                        expanded = query.replace(key, syn) if key in query else query
                        if expanded != query:
                            expansions.append(expanded)
            
            # Single word synonyms
            if word in self.SYNONYMS:
                for syn in self.SYNONYMS[word]:
                    new_query = query.replace(word, syn)
                    if new_query != query:
                        expansions.append(new_query)
        
        return list(set(expansions))  # Remove duplicates
    
    def expand_mixed_terms(self, query: str) -> List[str]:
        """Handle mixed Arabic-English terms."""
        expansions = [query]
        normalized = self.normalize(query)
        
        for ar_term, variants in self.MIXED_TERMS.items():
            if ar_term in normalized or any(v in normalized for v in variants):
                # Add all variants
                for variant in variants:
                    if variant not in query.lower():
                        expansions.append(f"{query} {variant}")
        
        return list(set(expansions))
    
    def add_domain_context(self, query: str, domain: str = 'realsoft') -> str:
        """Add domain-specific context to query."""
        domain_boosters = {
            'realsoft': ['ريلسوفت', 'realsoft', 'شركة', 'company'],
            'tech': ['تقنية', 'technology', 'software', 'حلول'],
            'services': ['خدمات', 'services', 'حلول', 'solutions'],
        }
        
        boosters = domain_boosters.get(domain, [])
        if boosters and not any(b in query.lower() for b in boosters):
            # Add domain context if not already present
            return f"{query} {' '.join(boosters[:2])}"
        return query
    
    def expand(self, query: str, method: str = 'full') -> ExpandedQuery:
        """
        Expand Arabic query using specified method.
        
        Methods:
        - 'normalize': Just normalize the text
        - 'synonyms': Add synonym variations
        - 'mixed': Handle mixed Arabic-English terms
        - 'full': Apply all expansions
        """
        variations = [query]
        
        if method in ('normalize', 'full'):
            normalized = self.normalize(query)
            if normalized != query:
                variations.append(normalized)
        
        if method in ('synonyms', 'full'):
            synonyms = self.expand_synonyms(query)
            variations.extend(synonyms)
        
        if method in ('mixed', 'full'):
            mixed = self.expand_mixed_terms(query)
            variations.extend(mixed)
        
        if method == 'full':
            # Add domain context for business queries
            contextualized = self.add_domain_context(query)
            if contextualized != query:
                variations.append(contextualized)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_norm = self.normalize(v)
            if v_norm not in seen:
                seen.add(v_norm)
                unique_variations.append(v)
        
        # Create expanded query (combine unique terms)
        expanded = ' '.join(unique_variations[:3])  # Limit to avoid too long queries
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            variations=unique_variations,
            expansion_method=method
        )


class ArabicRetrievalEnhancer:
    """
    Enhances retrieval for Arabic queries by:
    1. Expanding queries
    2. Running multiple searches
    3. Merging and deduplicating results
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.expander = ArabicQueryExpander()
    
    def search_enhanced(
        self,
        query: str,
        filters=None,
        top_k: int = 5,
        expansion_method: str = 'full'
    ) -> Tuple[List, Dict]:
        """
        Enhanced search with Arabic query expansion.
        
        Returns:
            Tuple of (merged_results, debug_info)
        """
        # Expand query
        expanded = self.expander.expand(query, method=expansion_method)
        
        all_results = []
        search_metadata = []
        
        # Search with original and variations
        for variation in expanded.variations[:3]:  # Limit to top 3 variations
            results, debug = self.retriever.search(variation, filters=filters, use_cache=True)
            all_results.extend(results)
            search_metadata.append({
                'query': variation,
                'results_count': len(results)
            })
        
        # Deduplicate by document_id + page
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r.document_id, r.page, r.chunk_index)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        # Re-rank by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        final_results = unique_results[:top_k]
        
        debug_info = {
            'original_query': query,
            'expanded_query': expanded.expanded,
            'variations': expanded.variations,
            'expansion_method': expansion_method,
            'searches_performed': len(search_metadata),
            'total_results_before_dedup': len(all_results),
            'final_results': len(final_results),
            'search_metadata': search_metadata
        }
        
        return final_results, debug_info


def demonstrate_arabic_expansion():
    """Demonstrate Arabic query expansion."""
    expander = ArabicQueryExpander()
    
    test_queries = [
        "ما هي ريلسوفت؟",
        "من هم عملاؤكم؟",
        "ما هو دور الذكاء الاصطناعي؟",
        "هل تقدمون خدمات تدريب؟",
        "ما هو منتج أداء؟",
    ]
    
    print("=" * 70)
    print("ARABIC QUERY EXPANSION DEMONSTRATION")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        print("-" * 70)
        
        expanded = expander.expand(query, method='full')
        
        print(f"Normalized: {expander.normalize(query)}")
        print(f"Expanded: {expanded.expanded}")
        print(f"Variations ({len(expanded.variations)}):")
        for i, var in enumerate(expanded.variations[:5], 1):
            print(f"  {i}. {var}")


def test_enhanced_retrieval():
    """Test enhanced retrieval with Arabic queries."""
    from app.config import get_settings
    from app.core.hybrid_retrieval import HybridRetriever
    from app.core.embedding_service import EmbeddingService
    from qdrant_client import QdrantClient
    
    settings = get_settings()
    
    # Initialize services
    embedding_service = EmbeddingService()
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )
    
    enhancer = ArabicRetrievalEnhancer(retriever)
    
    test_queries = [
        "ما هي ريلسوفت؟",
        "من هم عملاؤكم؟",
    ]
    
    print("\n" + "=" * 70)
    print("ENHANCED ARABIC RETRIEVAL TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        # Standard search
        print("\n>>> Standard Search:")
        standard_results, _ = retriever.search(query)
        for i, r in enumerate(standard_results[:3], 1):
            print(f"  {i}. Page {r.page}: {r.chunk_text[:60]}...")
        
        # Enhanced search
        print("\n>>> Enhanced Search (with expansion):")
        enhanced_results, debug_info = enhancer.search_enhanced(query, top_k=5)
        print(f"    Expansions used: {debug_info['variations']}")
        for i, r in enumerate(enhanced_results[:3], 1):
            print(f"  {i}. Page {r.page}: {r.chunk_text[:60]}...")


if __name__ == "__main__":
    demonstrate_arabic_expansion()
    print("\n\n")
    test_enhanced_retrieval()

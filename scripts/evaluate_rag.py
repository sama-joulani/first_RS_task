"""Evaluate RAG system against evaluation set."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.core.llm_service import LLMService
from app.core.prompt_manager import PromptManager
from app.core.rag_pipeline import RAGPipeline
from qdrant_client import QdrantClient


evaluation_set = [
    {"id": 1, "question": "What is RealSoft's main mission?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 7},
    {"id": 2, "question": "When was RealSoft founded?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 2},
    {"id": 3, "question": "What products does RealSoft offer?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 4, "question": "What is Al-Khwarizmi platform?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 5, "question": "How many years of experience does RealSoft have?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 2},
    {"id": 6, "question": "What statistical solutions does RealSoft provide?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 1},
    {"id": 7, "question": "What is RealSoft's vision?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 7},
    {"id": 8, "question": "Who is the Director of Jordan Meteorological Department?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 9, "question": "What countries does RealSoft operate in?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 8},
    {"id": 10, "question": "What is RealData Hub?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 11, "question": "What technology partners does RealSoft work with?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 10},
    {"id": 12, "question": "What is Microsoft partnership about?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 10},
    {"id": 13, "question": "What awards has RealSoft received from Union of Arab Statisticians?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 13},
    {"id": 14, "question": "What is RealSoft's contact number in Jordan?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 5},
    {"id": 15, "question": "What is the FalconMap platform?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 16, "question": "What is RealSoft's approach to digital transformation?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 1},
    {"id": 17, "question": "Who is Dr. Diaa Awad Kazem?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 4},
    {"id": 18, "question": "What is the Ada'a product?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 19, "question": "What is Esri and what do they do?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 10},
    {"id": 20, "question": "What is Mendix platform?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 10},
    {"id": 21, "question": "What values does RealSoft have?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 7},
    {"id": 22, "question": "What is RealSoft's talent outsourcing service?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 1},
    {"id": 23, "question": "What is the 8th International Conference of the Union of Arab Statisticians?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 13},
    {"id": 24, "question": "What is RealData Flow?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 3},
    {"id": 25, "question": "What is SBM company?", "expected_source": "The Content of RealSoft (1).pdf", "expected_page": 11},
]


def main():
    settings = get_settings()
    
    print("Initializing RAG pipeline...")
    embedding = EmbeddingService()
    llm = LLMService()
    prompt = PromptManager()
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    pipeline = RAGPipeline(
        embedding_service=embedding,
        llm_service=llm,
        prompt_manager=prompt,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )
    
    results = []
    correct_source = 0
    correct_page = 0
    has_answer = 0
    
    print(f"\nRunning evaluation on {len(evaluation_set)} questions...\n")
    
    for item in evaluation_set:
        print(f"Q{item['id']}: {item['question']}")
        
        # Get RAG response
        response = pipeline.rag(item['question'], top_k=5)
        
        # Check if we got citations
        has_citations = len(response.citations) > 0
        if has_citations:
            has_answer += 1
            first_citation = response.citations[0]
            retrieved_source = first_citation.document_title
            retrieved_page = first_citation.page
            
            # Check if source matches
            source_match = item['expected_source'] in retrieved_source or retrieved_source in item['expected_source']
            page_match = retrieved_page == item['expected_page']
            
            if source_match:
                correct_source += 1
            if page_match:
                correct_page += 1
            
            print(f"  ✓ Retrieved: {retrieved_source}, Page {retrieved_page}")
            print(f"  ✓ Expected:  {item['expected_source']}, Page {item['expected_page']}")
            print(f"  ✓ Source match: {source_match}, Page match: {page_match}")
        else:
            print(f"  ✗ No citations returned")
            print(f"  ✗ Expected: {item['expected_source']}, Page {item['expected_page']}")
        
        print(f"  Answer: {response.answer[:100]}...\n")
        
        results.append({
            'id': item['id'],
            'question': item['question'],
            'has_citations': has_citations,
            'source_match': source_match if has_citations else False,
            'page_match': page_match if has_citations else False,
        })
    
    # Summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(evaluation_set)}")
    print(f"Has citations: {has_answer} ({has_answer/len(evaluation_set)*100:.1f}%)")
    print(f"Correct source: {correct_source} ({correct_source/len(evaluation_set)*100:.1f}%)")
    print(f"Correct page: {correct_page} ({correct_page/len(evaluation_set)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

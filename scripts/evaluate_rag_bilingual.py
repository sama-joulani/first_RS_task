"""Evaluate RAG system against bilingual evaluation set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.core.llm_service import LLMService
from app.core.prompt_manager import PromptManager
from app.core.rag_pipeline import RAGPipeline
from qdrant_client import QdrantClient


evaluation_set = [
    {"id": 1, "question_ar": "ما هو الخوارزمي؟", "question_en": "What is Al-Khwarizmi?", "expected_pages": [30, 31, 32, 33, 34, 35], "category": "products"},
    {"id": 2, "question_ar": "هل لديكم فرع في المملكة العربية السعودية؟", "question_en": "Do you have a branch in KSA?", "expected_pages": [5], "category": "contact"},
    {"id": 3, "question_ar": "لماذا حلول ريلسوفت؟", "question_en": "Why RealSoft solutions?", "expected_pages": [9], "category": "company_info"},
    {"id": 4, "question_ar": "من هم عملاؤكم؟", "question_en": "Who are your clients?", "expected_pages": [2, 39, 86, 90, 97], "category": "clients"},
    {"id": 5, "question_ar": "أرسل لي شهادات العملاء.", "question_en": "Send me the Testimonials.", "expected_pages": [3, 4, 5], "category": "testimonials"},
    {"id": 6, "question_ar": "ما هي ريلسوفت؟", "question_en": "What is RealSoft?", "expected_pages": [2, 6], "category": "company_info"},
    {"id": 7, "question_ar": "كيف يمكنني التواصل معكم؟", "question_en": "How to contact you?", "expected_pages": [5, 6], "category": "contact"},
    {"id": 8, "question_ar": "ما هي قيم ريلسوفت؟", "question_en": "What are RealSoft's values?", "expected_pages": [7, 8], "category": "company_info"},
    {"id": 9, "question_ar": "حدثني عن فريق ريلسوفت.", "question_en": "Tell me about RealSoft's team.", "expected_pages": list(range(16, 30)), "category": "team"},
    {"id": 10, "question_ar": "ما هي المعايير الدولية للعمليات الإحصائية التي تتبعونها؟", "question_en": "What International standards for statistical operations do you follow?", "expected_pages": [66, 67], "category": "standards"},
    {"id": 11, "question_ar": "كيف يمكن لشركتكم مساعدتي؟", "question_en": "How can your company help me?", "expected_pages": [74, 75], "category": "services"},
    {"id": 12, "question_ar": "ما هي رؤية ريلسوفت ورسالتها؟", "question_en": "What is RealSoft's vision and mission?", "expected_pages": [7], "category": "company_info"},
    {"id": 13, "question_ar": 'ما هو منتج "أداء"؟', "question_en": 'What is the "Ada\'a" product?', "expected_pages": [36, 37], "category": "products"},
    {"id": 14, "question_ar": "من هو رئيس الشركة؟", "question_en": "Who is the President of the company?", "expected_pages": [17], "category": "team"},
    {"id": 15, "question_ar": 'ما هي منتجات "RealData Hub"؟', "question_en": 'What are the "RealData Hub" products?', "expected_pages": list(range(38, 44)), "category": "products"},
    {"id": 16, "question_ar": "ما هو دور الذكاء الاصطناعي في ريلسوفت؟", "question_en": "What is the role of AI at RealSoft?", "expected_pages": list(range(47, 51)), "category": "technology"},
    {"id": 17, "question_ar": "هل تقدمون حلولاً للتحول الرقمي؟", "question_en": "Do you provide Digital Transformation solutions?", "expected_pages": [1, 78], "category": "services"},
    {"id": 18, "question_ar": "ما هي الجوائز التي حصلت عليها الشركة؟", "question_en": "What awards has the company received?", "expected_pages": list(range(12, 17)), "category": "awards"},
    {"id": 19, "question_ar": "من هم شركاء التكنولوجيا لريلسوفت؟", "question_en": "Who are RealSoft's technology partners?", "expected_pages": [10, 11], "category": "partners"},
    {"id": 20, "question_ar": 'ما هي خدمات "بعد البيع" التي تقدمونها؟', "question_en": "What after-sales services do you offer?", "expected_pages": [100, 101], "category": "services"},
    {"id": 21, "question_ar": 'ما هو تطبيق "FalconMap"؟', "question_en": 'What is the "FalconMap" application?', "expected_pages": [44, 45], "category": "products"},
    {"id": 22, "question_ar": "هل تقدمون خدمات تدريب؟", "question_en": "Do you provide training services?", "expected_pages": [98, 99], "category": "services"},
    {"id": 23, "question_ar": "ما هي خبرتكم في تعداد العراق 2024؟", "question_en": "What is your experience with the Iraq Census 2024?", "expected_pages": list(range(104, 108)), "category": "experience"},
    {"id": 24, "question_ar": "ما هي خدمات التعهيد (Outsourcing) لديكم؟", "question_en": "What are your outsourcing services?", "expected_pages": [95, 96], "category": "services"},
    {"id": 25, "question_ar": "تحدث عن حلول ريلسوفت للقطاع المصرفي.", "question_en": "Tell me about RealSoft's solutions for the banking sector.", "expected_pages": list(range(88, 91)), "category": "solutions"},
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
    correct_page = 0
    has_answer = 0
    
    print(f"\nRunning evaluation on {len(evaluation_set)} questions...\n")
    
    for item in evaluation_set:
        # Test with Arabic question
        question_ar = item['question_ar']
        print(f"Q{item['id']} (AR): {question_ar}")
        
        response_ar = pipeline.rag(question_ar, top_k=5)
        has_citations_ar = len(response_ar.citations) > 0
        
        if has_citations_ar:
            has_answer += 1
            first_citation = response_ar.citations[0]
            retrieved_page_ar = first_citation.page
            page_match_ar = retrieved_page_ar in item['expected_pages']
            if page_match_ar:
                correct_page += 1
            print(f"  ✓ Retrieved: Page {retrieved_page_ar} | Match: {page_match_ar}")
        else:
            print(f"  ✗ No citations returned")
            page_match_ar = False
        
        print(f"  Answer (AR): {response_ar.answer[:80]}...\n")
        
        # Test with English question
        question_en = item['question_en']
        print(f"Q{item['id']} (EN): {question_en}")
        
        response_en = pipeline.rag(question_en, top_k=5)
        has_citations_en = len(response_en.citations) > 0
        
        if has_citations_en:
            has_answer += 1
            first_citation = response_en.citations[0]
            retrieved_page_en = first_citation.page
            page_match_en = retrieved_page_en in item['expected_pages']
            if page_match_en:
                correct_page += 1
            print(f"  ✓ Retrieved: Page {retrieved_page_en} | Match: {page_match_en}")
        else:
            print(f"  ✗ No citations returned")
            page_match_en = False
        
        print(f"  Answer (EN): {response_en.answer[:80]}...\n")
        print("-" * 60)
        
        results.append({
            'id': item['id'],
            'question_ar': question_ar,
            'question_en': question_en,
            'has_citations_ar': has_citations_ar,
            'has_citations_en': has_citations_en,
            'page_match_ar': page_match_ar,
            'page_match_en': page_match_en,
        })
    
    # Summary
    print("=" * 60)
    print("BILINGUAL EVALUATION SUMMARY")
    print("=" * 60)
    total_tests = len(evaluation_set) * 2  # AR + EN
    print(f"Total questions: {len(evaluation_set)} (AR + EN = {total_tests} tests)")
    print(f"Has citations: {has_answer}/{total_tests} ({has_answer/total_tests*100:.1f}%)")
    print(f"Correct page: {correct_page}/{total_tests} ({correct_page/total_tests*100:.1f}%)")
    print("=" * 60)
    
    # Save results to JSON
    import json
    from datetime import datetime
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "configuration": "dense_only_no_hybrid",
        "summary": {
            "total_questions": len(evaluation_set),
            "total_tests": total_tests,
            "has_citations": has_answer,
            "has_citations_pct": round(has_answer/total_tests*100, 1),
            "correct_page": correct_page,
            "correct_page_pct": round(correct_page/total_tests*100, 1),
        },
        "results": results,
    }
    
    output_file = Path(__file__).resolve().parent / "evaluation_results_dense_only.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

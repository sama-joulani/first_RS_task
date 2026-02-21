"""Mini evaluation dataset for RealSoft - 25 bilingual questions with expected sources."""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuestion:
    """Single evaluation question with expected source pages."""
    query_ar: str  # Arabic query
    query_en: str  # English query
    expected_pages: List[int]  # Expected page numbers
    category: str  # Question category for analysis


# RealSoft Evaluation Dataset - 25 Questions
REALSOFT_EVAL_DATASET: List[EvalQuestion] = [
    # Company Overview
    EvalQuestion(
        query_ar="ما هي ريلسوفت؟",
        query_en="What is RealSoft?",
        expected_pages=[2, 6],
        category="company_overview"
    ),
    EvalQuestion(
        query_ar="ما هي رؤية ريلسوفت ورسالتها؟",
        query_en="What is RealSoft's vision and mission?",
        expected_pages=[7],
        category="company_overview"
    ),
    EvalQuestion(
        query_ar="ما هي قيم ريلسوفت؟",
        query_en="What are RealSoft's values?",
        expected_pages=[7, 8],
        category="company_overview"
    ),
    EvalQuestion(
        query_ar="لماذا حلول ريلسوفت؟",
        query_en="Why RealSoft solutions?",
        expected_pages=[9],
        category="company_overview"
    ),
    
    # Leadership & Team
    EvalQuestion(
        query_ar="من هو رئيس الشركة؟",
        query_en="Who is the President of the company?",
        expected_pages=[17],
        category="leadership"
    ),
    EvalQuestion(
        query_ar="حدثني عن فريق ريلسوفت.",
        query_en="Tell me about RealSoft's team.",
        expected_pages=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        category="leadership"
    ),
    
    # Clients & Partners
    EvalQuestion(
        query_ar="من هم عملاؤكم؟",
        query_en="Who are your clients?",
        expected_pages=[2, 39, 86, 90, 97],
        category="clients_partners"
    ),
    EvalQuestion(
        query_ar="من هم شركاء التكنولوجيا لريلسوفت؟",
        query_en="Who are RealSoft's technology partners?",
        expected_pages=[10, 11],
        category="clients_partners"
    ),
    EvalQuestion(
        query_ar="أرسل لي شهادات العملاء.",
        query_en="Send me the Testimonials.",
        expected_pages=[3, 4, 5],
        category="clients_partners"
    ),
    
    # Contact & Locations
    EvalQuestion(
        query_ar="كيف يمكنني التواصل معكم؟",
        query_en="How to contact you?",
        expected_pages=[5, 6],
        category="contact"
    ),
    EvalQuestion(
        query_ar="هل لديكم فرع في المملكة العربية السعودية؟",
        query_en="Do you have a branch in KSA?",
        expected_pages=[5],
        category="contact"
    ),
    
    # Awards & Recognition
    EvalQuestion(
        query_ar="ما هي الجوائز التي حصلت عليها الشركة؟",
        query_en="What awards has the company received?",
        expected_pages=[12, 13, 14, 15, 16],
        category="awards"
    ),
    EvalQuestion(
        query_ar="ما هو الخوارزمي؟",
        query_en="What is Al-Khwarizmi?",
        expected_pages=[30, 31, 32, 33, 34, 35],
        category="awards"
    ),
    
    # Products
    EvalQuestion(
        query_ar='ما هو منتج "أداء"؟',
        query_en='What is the "Ada\'a" product?',
        expected_pages=[36, 37],
        category="products"
    ),
    EvalQuestion(
        query_ar='ما هي منتجات "RealData Hub"؟',
        query_en='What are the "RealData Hub" products?',
        expected_pages=[38, 39, 40, 41, 42, 43],
        category="products"
    ),
    EvalQuestion(
        query_ar='ما هو تطبيق "FalconMap"؟',
        query_en='What is the "FalconMap" application?',
        expected_pages=[44, 45],
        category="products"
    ),
    
    # Technology & AI
    EvalQuestion(
        query_ar="ما هو دور الذكاء الاصطناعي في ريلسوفت؟",
        query_en="What is the role of AI at RealSoft?",
        expected_pages=[47, 48, 49, 50],
        category="technology"
    ),
    EvalQuestion(
        query_ar="هل تقدمون حلولاً للتحول الرقمي؟",
        query_en="Do you provide Digital Transformation solutions?",
        expected_pages=[1, 78],
        category="technology"
    ),
    
    # Standards & Quality
    EvalQuestion(
        query_ar="ما هي المعايير الدولية للعمليات الإحصائية التي تتبعونها؟",
        query_en="What International standards for statistical operations do you follow?",
        expected_pages=[66, 67],
        category="standards"
    ),
    
    # Services
    EvalQuestion(
        query_ar="كيف يمكن لشركتكم مساعدتي؟",
        query_en="How can your company help me?",
        expected_pages=[74, 75],
        category="services"
    ),
    EvalQuestion(
        query_ar="تحدث عن حلول ريلسوفت للقطاع المصرفي.",
        query_en="Tell me about RealSoft's solutions for the banking sector.",
        expected_pages=[88, 89, 90],
        category="services"
    ),
    EvalQuestion(
        query_ar="ما هي خدمات التعهيد (Outsourcing) لديكم؟",
        query_en="What are your outsourcing services?",
        expected_pages=[95, 96],
        category="services"
    ),
    EvalQuestion(
        query_ar="هل تقدمون خدمات تدريب؟",
        query_en="Do you provide training services?",
        expected_pages=[98, 99],
        category="services"
    ),
    EvalQuestion(
        query_ar='ما هي خدمات "بعد البيع" التي تقدمونها؟',
        query_en="What after-sales services do you offer?",
        expected_pages=[100, 101],
        category="services"
    ),
    
    # Experience & Projects
    EvalQuestion(
        query_ar="ما هي خبرتكم في تعداد العراق 2024؟",
        query_en="What is your experience with the Iraq Census 2024?",
        expected_pages=[104, 105, 106, 107],
        category="experience"
    ),
]


def get_dataset_stats():
    """Print dataset statistics."""
    print("=" * 60)
    print("RealSoft Evaluation Dataset Statistics")
    print("=" * 60)
    print(f"Total Questions: {len(REALSOFT_EVAL_DATASET)}")
    
    # Count by category
    from collections import Counter
    categories = Counter(q.category for q in REALSOFT_EVAL_DATASET)
    print(f"\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    
    # Language distribution
    print(f"\nLanguages: Arabic + English (Bilingual)")
    print(f"Total Queries: {len(REALSOFT_EVAL_DATASET) * 2} (AR + EN pairs)")

"""
ماژول رتبه‌بندی اسناد با TF-IDF
روش: وزن‌دهی TF-IDF با نرمال‌سازی کسینوسی
"""
import math
from typing import List, Dict, Set, Tuple
from modules.preprocessor import preprocess
from modules.index_builder import InvertedIndex


def compute_tf_weight(tf: int) -> float:
    """
    وزن TF لگاریتمی (Log TF Weighting):
    tf_weight = 1 + log(tf) اگر tf > 0 وگرنه 0
    این روش تأثیر تکرار بیش از حد یک کلمه را کاهش می‌دهد.
    """
    if tf == 0:
        return 0.0
    return 1.0 + math.log(tf)


def compute_tfidf_score(
    query_terms: List[str],
    doc_id: int,
    index: InvertedIndex,
    use_cosine_norm: bool = True
) -> float:
    """
    محاسبه امتیاز TF-IDF برای یک سند نسبت به پرس‌وجو.
    فرمول: score(q, d) = Σ [ tf_weight(t, d) × idf(t) ]
    با نرمال‌سازی کسینوسی: score_norm = score / |d|
    """
    score = 0.0
    for term in query_terms:
        normalized = preprocess(term, apply_stop_words=False)
        if not normalized:
            continue
        t = normalized[0]
        tf_val = index.tf(t, doc_id)
        idf_val = index.idf(t)
        tf_w = compute_tf_weight(tf_val)
        score += tf_w * idf_val

    # نرمال‌سازی بر اساس طول سند
    if use_cosine_norm and score > 0:
        doc_len = index.doc_lengths.get(doc_id, 1)
        if doc_len > 0:
            score = score / math.sqrt(doc_len)
    return score


def compute_phrase_boost(
    phrases: List[str],
    doc_id: int,
    index: InvertedIndex,
    boost_factor: float = 2.0
) -> float:
    """
    افزودن امتیاز اضافی برای اسنادی که عبارات نقل‌قول را
    به صورت دقیق و متوالی دارند.
    """
    total_boost = 0.0
    for phrase in phrases:
        phrase_tokens = preprocess(phrase, apply_stop_words=False)
        if index.find_phrase(phrase_tokens, doc_id):
            total_boost += boost_factor
    return total_boost


def rank_documents(
    candidate_doc_ids: Set[int],
    query_info: dict,
    index: InvertedIndex,
    top_k: int = 10
) -> List[Tuple[int, float, dict]]:
    """
    رتبه‌بندی اسناد بر اساس امتیاز TF-IDF.

    ورودی‌ها:
        candidate_doc_ids: مجموعه شناسه اسنادی که از مرحله پردازش پرس‌وجو آمده‌اند
        query_info: اطلاعات پرس‌وجوی تجزیه‌شده
        index: نمایه وارونه
        top_k: تعداد نتایج برتر

    خروجی: لیست مرتب‌شده از (doc_id, امتیاز, اطلاعات سند)
    """
    if not candidate_doc_ids:
        return []

    # استخراج تمام ترم‌های پرس‌وجو
    all_query_terms = list(query_info.get('keywords', []))

    # اضافه کردن ترم‌های عبارات
    for phrase in query_info.get('phrases', []):
        phrase_tokens = preprocess(phrase, apply_stop_words=False)
        all_query_terms.extend(phrase_tokens)

    # اضافه کردن ترم‌های پرس‌وجوی بولی
    if query_info.get('boolean_query'):
        bool_tokens = preprocess(query_info['boolean_query'], apply_stop_words=False)
        bool_tokens = [t for t in bool_tokens if t not in ('and', 'or', 'not')]
        all_query_terms.extend(bool_tokens)

    # ساخت مجموعه doc_id -> اطلاعات سند
    doc_map = {doc['doc_id']: doc for doc in index.documents}

    scored = []
    for doc_id in candidate_doc_ids:
        if doc_id not in doc_map:
            continue

        # امتیاز TF-IDF
        tfidf_score = compute_tfidf_score(all_query_terms, doc_id, index)

        # امتیاز اضافی برای تطابق عبارت
        phrase_boost = compute_phrase_boost(
            query_info.get('phrases', []), doc_id, index
        )

        final_score = tfidf_score + phrase_boost
        scored.append((doc_id, final_score, doc_map[doc_id]))

    # مرتب‌سازی نزولی بر اساس امتیاز
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def format_results(
    ranked_results: List[Tuple[int, float, dict]],
    query: str
) -> str:
    """
    قالب‌بندی خروجی نتایج جستجو برای نمایش به کاربر.
    """
    if not ranked_results:
        return f"\nجستجو برای «{query}» نتیجه‌ای نداشت.\n"

    lines = []
    lines.append(f"\nنتایج جستجو برای: «{query}»")
    lines.append(f"تعداد نتایج: {len(ranked_results)}")
    lines.append("=" * 60)

    for rank, (doc_id, score, doc_info) in enumerate(ranked_results, start=1):
        home = doc_info.get('home_team', '')
        away = doc_info.get('away_team', '')
        stage = doc_info.get('stage', '')
        score_str = doc_info.get('score', '')
        venue = doc_info.get('venue', '')
        referee = doc_info.get('referee', '')
        notes = doc_info.get('notes', '')

        lines.append(f"\nرتبه {rank} | DocID: {doc_id} | امتیاز: {score:.4f}")
        lines.append(f"  مسابقه: {home} vs {away}")
        lines.append(f"  مرحله: {stage}  |  نتیجه: {score_str}")
        lines.append(f"  ورزشگاه: {venue}")
        lines.append(f"  داور: {referee}")
        if notes and notes != 'nan':
            lines.append(f"  یادداشت: {notes}")
        lines.append("-" * 60)

    return '\n'.join(lines)

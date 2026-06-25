import math
from typing import List, Dict, Set, Tuple
from DataPreprocessing.preprocessor import preprocess
from Indexing.index_builder import InvertedIndex


def compute_tf_weight(tf: int) -> float:
    if tf == 0:
        return 0.0
    return 1.0 + math.log(tf)


def compute_tfidf_score(
    query_terms: List[str],
    doc_id: int,
    index: InvertedIndex,
    use_cosine_norm: bool = True
) -> float:
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
    if not candidate_doc_ids:
        return []

    all_query_terms = list(query_info.get('keywords', []))
    for phrase in query_info.get('phrases', []):
        phrase_tokens = preprocess(phrase, apply_stop_words=False)
        all_query_terms.extend(phrase_tokens)

    if query_info.get('boolean_query'):
        bool_tokens = preprocess(query_info['boolean_query'], apply_stop_words=False)
        bool_tokens = [t for t in bool_tokens if t not in ('and', 'or', 'not')]
        all_query_terms.extend(bool_tokens)

    doc_map = {doc['doc_id']: doc for doc in index.documents}

    scored = []
    for doc_id in candidate_doc_ids:
        if doc_id not in doc_map:
            continue

        tfidf_score = compute_tfidf_score(all_query_terms, doc_id, index)
        phrase_boost = compute_phrase_boost(
            query_info.get('phrases', []), doc_id, index
        )

        final_score = tfidf_score + phrase_boost
        scored.append((doc_id, final_score, doc_map[doc_id]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def format_results(
    ranked_results: List[Tuple[int, float, dict]],
    query: str
) -> str:
    query_clean = " ".join(query.split())
    if not ranked_results:
        return f"\nSearch for '{query_clean}' returned 0 results.\n"

    lines = []
    lines.append(f"Search Results for: '{query_clean}'")
    lines.append(f"Total Retrieved Results: {len(ranked_results)}")
    lines.append("=" * 60)

    for rank, (doc_id, score, doc_info) in enumerate(ranked_results, start=1):
        home = " ".join(str(doc_info.get('home_team', '')).split())
        away = " ".join(str(doc_info.get('away_team', '')).split())
        stage = " ".join(str(doc_info.get('stage', '')).split())
        score_str = " ".join(str(doc_info.get('score', '')).split())
        venue = " ".join(str(doc_info.get('venue', '')).split())
        referee = " ".join(str(doc_info.get('referee', '')).split())
        notes = " ".join(str(doc_info.get('notes', '')).split())

        lines.append(f"Rank {rank} | DocID: {doc_id} | Score: {score:.4f}")
        lines.append(f"  Match: {home} vs {away}")
        lines.append(f"  Stage: {stage} | Score: {score_str}")
        lines.append(f"  Stadium/Venue: {venue}")
        lines.append(f"  Referee: {referee}")
        if notes and notes.lower() != 'nan':
            lines.append(f"  Notes: {notes}")
        lines.append("-" * 60)

    return '\n'.join(lines)

import math
from typing import List, Set, Dict, Tuple



RELEVANCE_JUDGMENTS: Dict[str, List[int]] = {
    "messi": [1, 4, 8, 16, 28, 44, 57],
    "mbappe goal": [1, 13, 43, 60],
    "final penalty": [1],
    "own goal": [16, 21, 23],
    "extra time goal": [1, 7, 8, 9, 11],
    "yellow cards argentina": [1, 4, 8, 28, 44],
    "substitutions france": [1, 3, 6, 13, 26, 43, 60],
    "matches in lusail": [1, 4, 8, 10, 19, 27, 36, 44, 52, 57],
    "referee marciniak": [1, 16, 43],
    "captain luka modric": [2, 4, 7, 11, 22, 39, 53],
    "penalty miss": [6, 18, 28, 42, 56, 59],
    "red card": [34, 45],
    "0-0 draw": [9, 22, 48, 50, 53, 58, 59],
    'team:Portugal player:Goncalo Ramos': [10],
    'round:Semi-finals team:Morocco': [3],
}

QUERY_DESCRIPTIONS = {
    "messi": "مسابقاتی که مسی در آن‌ها حضور داشت",
    "mbappe goal": "مسابقاتی که امباپه گل زد",
    "final penalty": "مسابقه فینال که به پنالتی کشیده شد",
    "own goal": "مسابقاتی که گل به خودی داشتند",
    "extra time goal": "مسابقاتی که گل در وقت اضافه زده شد",
    "yellow cards argentina": "مسابقات آرژانتین با کارت زرد",
    "substitutions france": "مسابقات فرانسه با تعویض",
    "matches in lusail": "مسابقات در استادیوم لوسیل",
    "referee marciniak": "مسابقاتی که مارکینیاک داور بود",
    "captain luka modric": "مسابقاتی که لوکا مودریچ کاپیتان بود",
    "penalty miss": "مسابقاتی که پنالتی از دست رفت",
    "red card": "مسابقاتی با کارت قرمز",
    "0-0 draw": "مسابقاتی که با تساوی 0-0 تمام شدند",
    'team:Portugal player:Goncalo Ramos': "مسابقات پرتغال با گونسالو راموس",
    'round:Semi-finals team:Morocco': "مسابقات نیمه‌نهایی مراکش",
}



def precision(retrieved: List[int], relevant: Set[int]) -> float:

    if not retrieved:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
    return relevant_retrieved / len(retrieved)


def recall(retrieved: List[int], relevant: Set[int]) -> float:

    if not relevant:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def f1_score(prec: float, rec: float) -> float:

    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:

    top_k = retrieved[:k]
    return precision(top_k, relevant)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:

    if not relevant or not retrieved:
        return 0.0

    hits = 0
    ap = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            ap += hits / i

    return ap / len(relevant)


def mean_average_precision(results_per_query: Dict[str, List[int]],
                           judgments: Dict[str, List[int]]) -> float:

    aps = []
    for query, retrieved in results_per_query.items():
        relevant = set(judgments.get(query, []))
        ap = average_precision(retrieved, relevant)
        aps.append(ap)
    if not aps:
        return 0.0
    return sum(aps) / len(aps)


def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_system(
    search_fn,
    judgments: Dict[str, List[int]] = None,
    top_k: int = 10
) -> dict:

    if judgments is None:
        judgments = RELEVANCE_JUDGMENTS

    all_results = {}
    per_query_metrics = []

    for query, relevant_ids in judgments.items():
        relevant = set(relevant_ids)
        retrieved = search_fn(query, top_k=top_k)
        all_results[query] = retrieved

        prec = precision(retrieved, relevant)
        rec = recall(retrieved, relevant)
        f1 = f1_score(prec, rec)
        p_at_5 = precision_at_k(retrieved, relevant, 5)
        ap = average_precision(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)

        per_query_metrics.append({
            'query': query,
            'description': QUERY_DESCRIPTIONS.get(query, query),
            'relevant_count': len(relevant),
            'retrieved_count': len(retrieved),
            'relevant_retrieved': sum(1 for d in retrieved if d in relevant),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'p@5': p_at_5,
            'ap': ap,
            'rr': rr,
        })

    def mean(key):
        vals = [m[key] for m in per_query_metrics]
        return sum(vals) / len(vals) if vals else 0.0

    map_score = mean_average_precision(all_results, judgments)

    summary = {
        'per_query': per_query_metrics,
        'mean_precision': mean('precision'),
        'mean_recall': mean('recall'),
        'mean_f1': mean('f1'),
        'mean_p@5': mean('p@5'),
        'MAP': map_score,
        'MRR': mean('rr'),
        'num_queries': len(per_query_metrics),
    }
    return summary


def format_evaluation_report(eval_results: dict) -> str:

    lines = []
    lines.append("=" * 70)
    lines.append("            گزارش ارزیابی سامانه بازیابی اطلاعات جام جهانی")
    lines.append("=" * 70)
    lines.append(f"\nتعداد پرس‌وجوهای ارزیابی: {eval_results['num_queries']}")
    lines.append("\n--- معیارهای کلی سامانه ---")
    lines.append(f"  Mean Precision       : {eval_results['mean_precision']:.4f}  ({eval_results['mean_precision']*100:.1f}%)")
    lines.append(f"  Mean Recall          : {eval_results['mean_recall']:.4f}  ({eval_results['mean_recall']*100:.1f}%)")
    lines.append(f"  Mean F1-Score        : {eval_results['mean_f1']:.4f}")
    lines.append(f"  Mean Precision@5     : {eval_results['mean_p@5']:.4f}")
    lines.append(f"  MAP (Mean Avg Prec)  : {eval_results['MAP']:.4f}")
    lines.append(f"  MRR (Mean Recip Rank): {eval_results['MRR']:.4f}")
    lines.append("\n" + "-" * 70)
    lines.append("--- نتایج ارزیابی به تفکیک پرس‌وجو ---\n")

    header = f"{'#':<3} {'پرس‌وجو':<35} {'P':<7} {'R':<7} {'F1':<7} {'AP':<7} {'P@5':<6}"
    lines.append(header)
    lines.append("-" * 70)

    for i, m in enumerate(eval_results['per_query'], start=1):
        q_short = m['query'][:33] + '..' if len(m['query']) > 35 else m['query']
        row = (
            f"{i:<3} {q_short:<35} "
            f"{m['precision']:<7.3f} {m['recall']:<7.3f} "
            f"{m['f1']:<7.3f} {m['ap']:<7.3f} {m['p@5']:<6.3f}"
        )
        lines.append(row)
        lines.append(f"     توضیح: {m['description']}")
        lines.append(
            f"     مرتبط: {m['relevant_count']}  |  بازیابی‌شده: {m['retrieved_count']}"
            f"  |  مرتبط بازیابی‌شده: {m['relevant_retrieved']}"
        )
        lines.append("")

    lines.append("=" * 70)
    return '\n'.join(lines)

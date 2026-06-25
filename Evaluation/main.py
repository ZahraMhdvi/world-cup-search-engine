

import os
import sys
import json
from DataPreprocessing.data_loader import load_dataset
from DataPreprocessing.document_builder import build_all_documents
from Indexing.index_builder import InvertedIndex
from QueryProcessing.query_processor import process_query
from QueryProcessing.ranking import rank_documents, format_results
from Evaluation.evaluator import (
    evaluate_system,
    format_evaluation_report,
    RELEVANCE_JUDGMENTS,
)


OUTPUT_DIR = "output"
DATA_PATH = "Evaluation/matches_1930_2022.csv"


def build_system(data_path: str, year: int = 2022):

    print("\n[۱] بارگذاری دیتاست جام جهانی...")
    df = load_dataset(data_path, year=year)
    print(f"    تعداد مسابقات: {len(df)} مسابقه از جام جهانی {year}")

    print("\n[۲] ساخت اسناد از رکوردهای مسابقات...")
    documents = build_all_documents(df)
    print(f"    {len(documents)} سند ساخته شد.")

    print("\n[۳] ساخت نمایه وارونه...")
    index = InvertedIndex()
    index.build(documents)

    stats = index.get_vocabulary_stats()
    print(f"    واژگان منحصربه‌فرد: {stats['vocabulary_size']}")
    print(f"    کل پستینگ‌ها: {stats['total_postings']}")
    print(f"    میانگین طول سند: {stats['avg_doc_length']:.1f} توکن")

    return index, documents, df


def search(query: str, index: InvertedIndex, top_k: int = 10):

    candidate_docs, query_info = process_query(query, index)
    ranked = rank_documents(candidate_docs, query_info, index, top_k=top_k)
    output = format_results(ranked, query)
    print(output)
    return ranked


def search_for_eval(query: str, index: InvertedIndex, top_k: int = 10) -> list:

    candidate_docs, query_info = process_query(query, index)
    ranked = rank_documents(candidate_docs, query_info, index, top_k=top_k)
    return [doc_id for doc_id, _, _ in ranked]


def run_sample_queries(index: InvertedIndex):

    sample_queries = [
        "messi",
        "mbappe goal",
        "final penalty",
        "own goal",
        "extra time goal",
        "yellow cards argentina",
        "substitutions france",
        "matches in lusail",
        "referee marciniak",
        "captain luka modric",
        "penalty miss",
        "red card",
        "0-0 draw",

        "mbappe AND goal",
        "penalty AND quarter-finals",
        "messi AND NOT penalty",
        "argentina AND final",
        "morocco AND semi",
        "team:Argentina round:Final",
        "round:Semi-finals team:Morocco",
        'team:Portugal player:Goncalo Ramos',
        'referee:"Szymon Marciniak"',
        '"extra time goal"',
        '"penalty shootout"',
        '"Lionel Messi"',
    ]

    results_log = []
    print("\n" + "=" * 70)
    print("         اجرای پرس‌وجوهای نمونه")
    print("=" * 70)

    for query in sample_queries:
        candidate_docs, query_info = process_query(query, index)
        ranked = rank_documents(candidate_docs, query_info, index, top_k=5)
        output = format_results(ranked, query)
        print(output)

        results_log.append({
            'query': query,
            'result_count': len(ranked),
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'score': round(score, 4),
                    'match': f"{info['home_team']} vs {info['away_team']}",
                    'stage': info['stage'],
                    'result': info['score'],
                }
                for i, (doc_id, score, info) in enumerate(ranked)
            ]
        })

    return results_log


def run_evaluation(index: InvertedIndex):

    print("\n" + "=" * 70)
    print("               ارزیابی سامانه")
    print("=" * 70)

    def search_fn(query, top_k=10):
        return search_for_eval(query, index, top_k=top_k)

    eval_results = evaluate_system(search_fn, RELEVANCE_JUDGMENTS, top_k=10)
    report = format_evaluation_report(eval_results)
    print(report)
    return eval_results, report


def save_index_info(index: InvertedIndex, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    stats = index.get_vocabulary_stats()

    sorted_terms = sorted(
        index.doc_freq.items(), key=lambda x: x[1], reverse=True
    )[:50]

    index_summary = {
        'stats': stats,
        'top_50_terms': [
            {'term': t, 'doc_freq': df} for t, df in sorted_terms
        ],
        'sample_postings': {}
    }


    sample_terms = ['messi', 'goal', 'penalty', 'referee', 'final']
    for term in sample_terms:
        postings = index.get_postings(term)
        index_summary['sample_postings'][term] = {
            str(doc_id): {'tf': info['tf'], 'positions': info['positions'][:5]}
            for doc_id, info in list(postings.items())[:5]
        }

    path = os.path.join(output_dir, 'index_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[نمایه] اطلاعات نمایه در {path} ذخیره شد.")


def save_documents(documents: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'documents.json')
    save_docs = [
        {
            'doc_id': d['doc_id'],
            'home_team': d['home_team'],
            'away_team': d['away_team'],
            'stage': d['stage'],
            'date': d['date'],
            'score': d['score'],
            'venue': d['venue'],
            'referee': d['referee'],
            'text_preview': d['text'][:300] + '...' if len(d['text']) > 300 else d['text'],
        }
        for d in documents
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_docs, f, ensure_ascii=False, indent=2)
    print(f"[اسناد] اسناد در {path} ذخیره شدند.")


def interactive_mode(index: InvertedIndex):
    print("\n" + "=" * 70)
    print("         حالت تعاملی - برای خروج 'exit' تایپ کنید")
    print("=" * 70)
    print("راهنما:")
    print("  جستجوی ساده  : messi goal")
    print("  بولی         : mbappe AND goal | penalty AND NOT shootout")
    print("  عبارت         : \"extra time goal\"")
    print("  فیلد-محور    : team:Argentina round:Final")
    print("-" * 70)

    while True:
        try:
            query = input("\nپرس‌وجو: ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit', 'خروج'):
                print("خروج از سامانه.")
                break
            search(query, index, top_k=10)
        except (KeyboardInterrupt, EOFError):
            print("\nخروج از سامانه.")
            break


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index, documents, df = build_system(DATA_PATH, year=2022)
    save_documents(documents, OUTPUT_DIR)
    save_index_info(index, OUTPUT_DIR)

    results_log = run_sample_queries(index)
    results_path = os.path.join(OUTPUT_DIR, 'sample_query_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\n[خروجی] نتایج نمونه در {results_path} ذخیره شدند.")
    eval_results, report = run_evaluation(index)
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[ارزیابی] گزارش ارزیابی در {report_path} ذخیره شد.")

    eval_json_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    eval_save = {
        'mean_precision': eval_results['mean_precision'],
        'mean_recall': eval_results['mean_recall'],
        'mean_f1': eval_results['mean_f1'],
        'mean_p@5': eval_results['mean_p@5'],
        'MAP': eval_results['MAP'],
        'MRR': eval_results['MRR'],
        'per_query': [
            {k: v for k, v in m.items()} for m in eval_results['per_query']
        ]
    }
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_save, f, ensure_ascii=False, indent=2)

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode(index)


if __name__ == '__main__':
    main()

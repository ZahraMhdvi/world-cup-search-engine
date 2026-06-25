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


def clean_spaces(text: str) -> str:
    return " ".join(text.split())


def build_system(data_path: str, year: int = 2022):
    print(clean_spaces("\n[1] Loading World Cup dataset..."))
    df = load_dataset(data_path, year=year)
    print(clean_spaces(f" Total Matches: {len(df)} matches loaded from World Cup {year}"))

    print(clean_spaces("\n[2] Building documents from match records..."))
    documents = build_all_documents(df)
    print(clean_spaces(f" {len(documents)} documents successfully constructed."))

    print(clean_spaces("\n[3] Constructing Inverted Index..."))
    index = InvertedIndex()
    index.build(documents)

    stats = index.get_vocabulary_stats()
    print(clean_spaces(f" Unique Terms (Vocabulary Size): {stats['vocabulary_size']}"))
    print(clean_spaces(f" Total Postings: {stats['total_postings']}"))
    print(clean_spaces(f" Average Document Length: {stats['avg_doc_length']:.1f} tokens"))

    return index, documents, df


def search(query: str, index: InvertedIndex, top_k: int = 10):
    query = clean_spaces(query)
    candidate_docs, query_info = process_query(query, index)
    ranked = rank_documents(candidate_docs, query_info, index, top_k=top_k)
    output = format_results(ranked, query)
    # Splitting and joining lines to ensure format_results has no weird extra lines or spaces
    clean_output = "\n".join(clean_spaces(line) for line in output.splitlines())
    print(clean_output)
    return ranked


def search_for_eval(query: str, index: InvertedIndex, top_k: int = 10) -> list:
    query = clean_spaces(query)
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
    print(clean_spaces(" Executing Sample Benchmark Queries "))
    print("=" * 70)

    for query in sample_queries:
        query = clean_spaces(query)
        candidate_docs, query_info = process_query(query, index)
        ranked = rank_documents(candidate_docs, query_info, index, top_k=5)
        output = format_results(ranked, query)
        clean_output = "\n".join(clean_spaces(line) for line in output.splitlines())
        print(clean_output)

        results_log.append({
            'query': query,
            'result_count': len(ranked),
            'top_results': [
                {
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'score': round(score, 4),
                    'match': clean_spaces(f"{info['home_team']} vs {info['away_team']}"),
                    'stage': clean_spaces(info['stage']),
                    'result': clean_spaces(str(info['score'])),
                }
                for i, (doc_id, score, info) in enumerate(ranked)
            ]
        })

    return results_log


def run_evaluation(index: InvertedIndex):
    print("\n" + "=" * 70)
    print(clean_spaces(" Evaluating System Performance "))
    print("=" * 70)

    def search_fn(query, top_k=10):
        return search_for_eval(query, index, top_k=top_k)

    eval_results = evaluate_system(search_fn, RELEVANCE_JUDGMENTS, top_k=10)
    report = format_evaluation_report(eval_results)
    clean_report = "\n".join(clean_spaces(line) for line in report.splitlines())
    print(clean_report)
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
            {'term': clean_spaces(t), 'doc_freq': df} for t, df in sorted_terms
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
    print(clean_spaces(f"\n[Index] Index analytics metadata exported to {path}"))


def save_documents(documents: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'documents.json')
    save_docs = [
        {
            'doc_id': d['doc_id'],
            'home_team': clean_spaces(d['home_team']),
            'away_team': clean_spaces(d['away_team']),
            'stage': clean_spaces(d['stage']),
            'date': clean_spaces(d['date']),
            'score': clean_spaces(d['score']),
            'venue': clean_spaces(d['venue']),
            'referee': clean_spaces(d['referee']),
            'text_preview': clean_spaces(d['text'][:300]) + '...' if len(d['text']) > 300 else clean_spaces(d['text']),
        }
        for d in documents
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_docs, f, ensure_ascii=False, indent=2)
    print(clean_spaces(f"[Documents] Processed documents collection saved to {path}"))


def interactive_mode(index: InvertedIndex):
    print("\n" + "=" * 70)
    print(clean_spaces(" Interactive Mode Enabled - Type 'exit' to quit "))
    print("=" * 70)
    print("Query Syntax Examples:")
    print(clean_spaces(" Free-text Keyword Search : messi goal"))
    print(clean_spaces(" Boolean Constraints : mbappe AND goal | penalty AND NOT shootout"))
    print(clean_spaces(" Exact Phrase Search : \"extra time goal\""))
    print(clean_spaces(" Field-specific Filters : team:Argentina round:Final"))
    print("-" * 70)

    while True:
        try:
            query = input("\nSearch Query: ").strip()
            if not query:
                continue
            query = clean_spaces(query)
            if query.lower() in ('exit', 'quit'):
                print(clean_spaces("Exiting search engine platform."))
                break
            search(query, index, top_k=10)
        except (KeyboardInterrupt, EOFError):
            print(clean_spaces("\nExiting search engine platform."))
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
    print(clean_spaces(f"\nTest execution query outcomes recorded in {results_path}"))

    eval_results, report = run_evaluation(index)
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(clean_spaces(f"\nPerformance metric report successfully generated at {report_path}"))

    eval_json_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    eval_save = {
        'mean_precision': eval_results['mean_precision'],
        'mean_recall': eval_results['mean_recall'],
        'mean_f1': eval_results['mean_f1'],
        'mean_p@5': eval_results['mean_p@5'],
        'MAP': eval_results['MAP'],
        'MRR': eval_results['MRR'],
        'per_query': [
            {k: clean_spaces(str(v)) if isinstance(v, str) else v for k, v in m.items()}
            for m in eval_results['per_query']
        ]
    }
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_save, f, ensure_ascii=False, indent=2)

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode(index)


if __name__ == '__main__':
    main()
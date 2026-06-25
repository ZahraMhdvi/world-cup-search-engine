# World Cup 2022 Information Retrieval Engine

A high-performance, modular Information Retrieval (IR) system and text search engine built from scratch in Python. This system processes, indexes, searches, and evaluates structured and semi-structured data from the FIFA World Cup 2022 dataset without relying on external search libraries.

## Features

* **Custom Text Preprocessing Pipeline:** Unicode normalization (accent removal), case folding, punctuation cleaning, tokenization, football-specific stop-word filtering, and custom token normalization.
* **Positional Inverted Index:** Built completely from scratch using standard Python dictionaries, storing Term Frequency (TF) and exact token positions.
* **Parametric / Field-Based Indexing:** Supports fast structured filtering across specific document zones (`team`, `round`, `referee`, `venue`, `year`).
* **Vector Space Model (VSM) Ranking:** Scores and ranks documents utilizing logarithmic TF weighting, smoothed IDF calculations, and **Cosine Normalization** to balance document lengths.
* **Advanced Query Processing:**
    * *Free-text Keywords:* Rank-ordered keyword matching.
    * *Boolean Queries:* Full support for `AND`, `OR`, and `NOT` operators.
    * *Exact Phrase Search:* Positional proximity verification for queries inside double quotes (`"..."`).
    * *Field Filters:* Structured queries such as `team:Argentina round:Final`.
* **Automated Evaluation Suite:** Computes classic IR evaluation metrics including Precision, Recall, F1-Score, Precision@5, Mean Average Precision (MAP), and Mean Reciprocal Rank (MRR).



## Project Directory Structure

```text
world-cup-search-engine/
│
├── main.py                          # Main entry point managing execution workflow
│
├── DataPreprocessing/               # Data ingestion & text normalization package
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading, HTML entity cleaning & list parsing
│   ├── document_builder.py          # Concat fields into a unified text document per match
│   └── preprocessor.py              # Case folding, tokenization, and stop-word filtering
│
├── Indexing/                        # Positional index architecture package
│   ├── __init__.py
│   └── index_builder.py             # Vocabulary stats and main Inverted Index class
│
├── QueryProcessing/                 # Search core & mathematical ranking package
│   ├── __init__.py
│   ├── query_processor.py           # Logic parser for keyword, boolean, phrase, and zone search
│   └── ranking.py                   # VSM scoring using TF-IDF with Cosine Normalization
│
├── Evaluation/                      # System validation benchmark package
│   ├── __init__.py
│   ├── matches_1930_2022.csv        # World Cup match dataset (CSV)
│   └── evaluator.py                 # Benchmarking framework and ground truth judgments
│
└── output/                          # Auto-generated project output artifacts
    ├── documents.json               # Refactored textual document representations
    ├── index_summary.json           # Inverted Index analytics and posting list samples
    ├── sample_query_results.json    # Ranking outcomes for benchmark test queries
    ├── evaluation_metrics.json      # Quantitative evaluation metrics in JSON format
    └── evaluation_report.txt        # Formatted text table report of system performance

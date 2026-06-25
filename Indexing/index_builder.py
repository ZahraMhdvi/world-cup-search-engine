import math
from collections import defaultdict
from typing import List, Dict, Tuple
from DataPreprocessing.preprocessor import preprocess


class InvertedIndex:

    def __init__(self):
        self.index: Dict[str, Dict[int, Dict]] = defaultdict(dict)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.documents: List[dict] = []
        self.field_index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_lengths: Dict[int, int] = {}

    def add_document(self, doc_id: int, text: str, fields: dict = None):
        tokens = preprocess(text, apply_stop_words=True)
        self.doc_lengths[doc_id] = len(tokens)

        term_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        for term, positions in term_positions.items():
            self.index[term][doc_id] = {
                'tf': len(positions),
                'positions': positions,
            }
            self.doc_freq[term] += 1

        if fields:
            for field, value in fields.items():
                if isinstance(value, list):
                    for v in value:
                        norm_v = " ".join(str(v).lower().split())
                        if norm_v:
                            self.field_index[field][norm_v].append(doc_id)
                else:
                    norm_v = " ".join(str(value).lower().split())
                    if norm_v:
                        self.field_index[field][norm_v].append(doc_id)

    def build(self, documents: list):
        self.total_docs = len(documents)
        self.documents = documents
        for doc in documents:
            self.add_document(doc['doc_id'], doc['text'], doc.get('fields'))
        print(f"[Index] Indexed {self.total_docs} documents with {len(self.index)} unique terms.")

    def get_postings(self, term: str) -> Dict[int, Dict]:
        normalized = preprocess(term, apply_stop_words=False)
        if not normalized:
            return {}
        return self.index.get(normalized[0], {})

    def get_doc_ids(self, term: str) -> set:
        return set(self.get_postings(term).keys())

    def idf(self, term: str) -> float:
        normalized = preprocess(term, apply_stop_words=False)
        if not normalized:
            return 0.0
        t = normalized[0]
        df = self.doc_freq.get(t, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def tf(self, term: str, doc_id: int) -> int:
        normalized = preprocess(term, apply_stop_words=False)
        if not normalized:
            return 0
        t = normalized[0]
        return self.index.get(t, {}).get(doc_id, {}).get('tf', 0)

    def get_positions(self, term: str, doc_id: int) -> List[int]:
        normalized = preprocess(term, apply_stop_words=False)
        if not normalized:
            return []
        t = normalized[0]
        return self.index.get(t, {}).get(doc_id, {}).get('positions', [])

    def get_field_docs(self, field: str, value: str) -> set:
        norm_value = " ".join(str(value).lower().split())
        results = set()
        field_dict = self.field_index.get(field, {})
        for key, doc_ids in field_dict.items():
            if norm_value in key or key in norm_value:
                results.update(doc_ids)
        return results

    def get_vocabulary_stats(self) -> dict:
        return {
            'total_docs': self.total_docs,
            'vocabulary_size': len(self.index),
            'total_postings': sum(len(v) for v in self.index.values()),
            'avg_doc_length': (
                sum(self.doc_lengths.values()) / len(self.doc_lengths)
                if self.doc_lengths else 0
            ),
        }

    def find_phrase(self, phrase_terms: List[str], doc_id: int) -> bool:
        if not phrase_terms:
            return False

        norm_terms = []
        for t in phrase_terms:
            n = preprocess(t, apply_stop_words=False)
            if n:
                norm_terms.append(n[0])
        if not norm_terms:
            return False

        for term in norm_terms:
            if doc_id not in self.index.get(term, {}):
                return False

        first_positions = self.index[norm_terms[0]][doc_id]['positions']
        for start_pos in first_positions:
            match = True
            for offset, term in enumerate(norm_terms[1:], start=1):
                term_positions = self.index[term][doc_id]['positions']
                if (start_pos + offset) not in term_positions:
                    match = False
                    break
            if match:
                return True
        return False
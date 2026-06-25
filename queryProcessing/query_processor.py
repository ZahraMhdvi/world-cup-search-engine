
import re
from typing import List, Set, Tuple
from dataPreprocessing.preprocessor import preprocess, preprocess_query
from indexing.index_builder import InvertedIndex



def parse_field_queries(query: str) -> Tuple[str, dict]:

    field_pattern = re.compile(r'(\w+):"([^"]+)"|(\w+):(\S+)')
    fields = {}
    remaining = query
    for match in field_pattern.finditer(query):
        if match.group(1):
            field = match.group(1).lower()
            value = match.group(2).lower()
        else:
            field = match.group(3).lower()
            value = match.group(4).lower()
        fields[field] = value
        remaining = remaining.replace(match.group(0), '').strip()
    return remaining.strip(), fields


def parse_phrase_queries(query: str) -> Tuple[str, List[str]]:

    phrase_pattern = re.compile(r'"([^"]+)"')
    phrases = phrase_pattern.findall(query)
    remaining = phrase_pattern.sub('', query).strip()
    return remaining, phrases


def tokenize_boolean(query: str) -> List[str]:

    tokens = re.split(r'\s+', query.strip())
    return [t for t in tokens if t]


def is_boolean_query(query: str) -> bool:

    boolean_ops = re.compile(r'\bAND\b|\bOR\b|\bNOT\b')
    return bool(boolean_ops.search(query))



def execute_boolean(query: str, index: InvertedIndex) -> Set[int]:

    all_doc_ids = set(doc['doc_id'] for doc in index.documents)
    tokens = tokenize_boolean(query)

    if not tokens:
        return set()

    result = None
    current_op = 'OR'

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == 'AND':
            current_op = 'AND'
            i += 1
            continue
        elif token == 'OR':
            current_op = 'OR'
            i += 1
            continue
        elif token == 'NOT':

            i += 1
            if i < len(tokens):
                next_term = tokens[i]
                term_docs = _get_term_docs(next_term, index)
                not_docs = all_doc_ids - term_docs
                result = _apply_op(result, not_docs, current_op, all_doc_ids)
            i += 1
            continue
        else:
            term_docs = _get_term_docs(token, index)
            result = _apply_op(result, term_docs, current_op, all_doc_ids)
            current_op = 'AND'
            i += 1

    return result if result is not None else set()


def _get_term_docs(term: str, index: InvertedIndex) -> Set[int]:

    normalized = preprocess(term, apply_stop_words=False)
    result = set()
    for t in normalized:
        result |= index.get_doc_ids(t)
    return result


def _apply_op(current: Set[int], new: Set[int], op: str, all_docs: Set[int]) -> Set[int]:

    if current is None:
        return new
    if op == 'AND':
        return current & new
    elif op == 'OR':
        return current | new
    return current



def execute_phrase_query(phrase: str, index: InvertedIndex) -> Set[int]:

    phrase_tokens = preprocess(phrase, apply_stop_words=False)
    if not phrase_tokens:
        return set()

    candidate_docs = None
    for token in phrase_tokens:
        token_docs = index.get_doc_ids(token)
        if candidate_docs is None:
            candidate_docs = token_docs
        else:
            candidate_docs &= token_docs

    if not candidate_docs:
        return set()

    result = set()
    for doc_id in candidate_docs:
        if index.find_phrase(phrase_tokens, doc_id):
            result.add(doc_id)
    return result



def process_query(query: str, index: InvertedIndex) -> Tuple[Set[int], dict]:

    query_info = {
        'original': query,
        'fields': {},
        'phrases': [],
        'keywords': [],
        'is_boolean': False,
        'boolean_query': '',
    }


    remaining, field_filters = parse_field_queries(query)
    query_info['fields'] = field_filters

    remaining, phrases = parse_phrase_queries(remaining)
    query_info['phrases'] = phrases

    candidate_docs = None

    if phrases:
        phrase_result = set()
        for phrase in phrases:
            phrase_docs = execute_phrase_query(phrase, index)
            if phrase_result:
                phrase_result &= phrase_docs
            else:
                phrase_result = phrase_docs
        candidate_docs = phrase_result

    if remaining:
        if is_boolean_query(remaining):
            query_info['is_boolean'] = True
            query_info['boolean_query'] = remaining
            bool_docs = execute_boolean(remaining, index)
            if candidate_docs is None:
                candidate_docs = bool_docs
            else:
                candidate_docs &= bool_docs
        else:

            tokens = preprocess_query(remaining, apply_stop_words=False)
            query_info['keywords'] = tokens
            keyword_docs = set()
            for token in tokens:
                keyword_docs |= index.get_doc_ids(token)
            if candidate_docs is None:
                candidate_docs = keyword_docs
            else:
                candidate_docs &= keyword_docs

    if candidate_docs is None:
        candidate_docs = set(doc['doc_id'] for doc in index.documents)

    structural_fields = {'team', 'round', 'stage', 'referee', 'venue', 'year', 'home_team', 'away_team'}
    for field, value in field_filters.items():
        if field in structural_fields:
            field_docs = index.get_field_docs(field, value)
            candidate_docs &= field_docs
        else:
            kw_tokens = preprocess_query(value, apply_stop_words=False)
            kw_docs = set()
            for token in kw_tokens:
                kw_docs |= index.get_doc_ids(token)
            if kw_docs:
                candidate_docs &= kw_docs

    return candidate_docs, query_info

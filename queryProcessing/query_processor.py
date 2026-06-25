"""
ماژول پردازش پرس‌وجو
پشتیبانی از:
1. جستجوی ساده کلمات کلیدی
2. جستجوی بولی (AND, OR, NOT)
3. جستجوی عبارت (phrase query با دابل کوتیشن)
4. جستجوی فیلد-محور (team:Argentina round:Final)
"""
import re
from typing import List, Set, Tuple
from modules.preprocessor import preprocess, preprocess_query
from modules.index_builder import InvertedIndex


# ---- تجزیه پرس‌وجو ----

def parse_field_queries(query: str) -> Tuple[str, dict]:
    """
    استخراج فیلترهای فیلد-محور از پرس‌وجو.
    مثال: 'team:Argentina round:Final messi' -> ('messi', {'team': 'argentina', 'round': 'final'})
    """
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
    """
    استخراج عبارت‌های داخل دابل کوتیشن.
    مثال: '"Szymon Marciniak" penalty' -> ('penalty', ['Szymon Marciniak'])
    """
    phrase_pattern = re.compile(r'"([^"]+)"')
    phrases = phrase_pattern.findall(query)
    remaining = phrase_pattern.sub('', query).strip()
    return remaining, phrases


def tokenize_boolean(query: str) -> List[str]:
    """
    توکن‌سازی پرس‌وجوی بولی با حفظ عملگرهای AND، OR، NOT.
    مثال: 'messi AND goal NOT penalty' -> ['messi', 'AND', 'goal', 'NOT', 'penalty']
    """
    tokens = re.split(r'\s+', query.strip())
    return [t for t in tokens if t]


def is_boolean_query(query: str) -> bool:
    """تشخیص اینکه آیا پرس‌وجو شامل عملگرهای بولی است"""
    boolean_ops = re.compile(r'\bAND\b|\bOR\b|\bNOT\b')
    return bool(boolean_ops.search(query))


# ---- اجرای پرس‌وجوهای بولی ----

def execute_boolean(query: str, index: InvertedIndex) -> Set[int]:
    """
    اجرای پرس‌وجوی بولی با عملگرهای AND، OR، NOT.
    پشتیبانی از: AND، OR، NOT (پیشوندی)
    مثال: 'mbappe AND goal', 'penalty AND quarter-finals', 'messi AND NOT penalty'
    """
    all_doc_ids = set(doc['doc_id'] for doc in index.documents)
    tokens = tokenize_boolean(query)

    if not tokens:
        return set()

    result = None
    current_op = 'OR'   # عملگر پیش‌فرض

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
            # NOT unary: نقیض ترم بعدی
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
            current_op = 'AND'  # بعد از اولین ترم عملگر پیش‌فرض AND می‌شود
            i += 1

    return result if result is not None else set()


def _get_term_docs(term: str, index: InvertedIndex) -> Set[int]:
    """بازیابی مجموعه اسناد حاوی یک ترم"""
    normalized = preprocess(term, apply_stop_words=False)
    result = set()
    for t in normalized:
        result |= index.get_doc_ids(t)
    return result


def _apply_op(current: Set[int], new: Set[int], op: str, all_docs: Set[int]) -> Set[int]:
    """اعمال عملگر بولی روی دو مجموعه"""
    if current is None:
        return new
    if op == 'AND':
        return current & new
    elif op == 'OR':
        return current | new
    return current


# ---- اجرای پرس‌وجوی عبارت ----

def execute_phrase_query(phrase: str, index: InvertedIndex) -> Set[int]:
    """
    جستجوی عبارت با استفاده از نمایه پوزیشنی.
    مثال: "extra time goal", "Szymon Marciniak"
    """
    phrase_tokens = preprocess(phrase, apply_stop_words=False)
    if not phrase_tokens:
        return set()

    # ابتدا اسنادی که همه ترم‌ها را دارند پیدا می‌کنیم
    candidate_docs = None
    for token in phrase_tokens:
        token_docs = index.get_doc_ids(token)
        if candidate_docs is None:
            candidate_docs = token_docs
        else:
            candidate_docs &= token_docs

    if not candidate_docs:
        return set()

    # سپس بررسی مجاورت ترم‌ها
    result = set()
    for doc_id in candidate_docs:
        if index.find_phrase(phrase_tokens, doc_id):
            result.add(doc_id)
    return result


# ---- پردازش کلی پرس‌وجو ----

def process_query(query: str, index: InvertedIndex) -> Tuple[Set[int], dict]:
    """
    پردازش کامل پرس‌وجوی کاربر.
    ترتیب پردازش:
    1. استخراج فیلترهای فیلد-محور (team:X round:Y)
    2. استخراج عبارت‌های نقل‌قول ("phrase")
    3. تشخیص و اجرای بولی (AND/OR/NOT)
    4. جستجوی ساده کلمات کلیدی
    5. اعمال فیلترهای فیلد

    برمی‌گرداند: (مجموعه doc_id‌ها, اطلاعات پرس‌وجو برای رتبه‌بندی)
    """
    query_info = {
        'original': query,
        'fields': {},
        'phrases': [],
        'keywords': [],
        'is_boolean': False,
        'boolean_query': '',
    }

    # مرحله ۱: استخراج فیلتر فیلدها
    remaining, field_filters = parse_field_queries(query)
    query_info['fields'] = field_filters

    # مرحله ۲: استخراج عبارات نقل‌قول
    remaining, phrases = parse_phrase_queries(remaining)
    query_info['phrases'] = phrases

    # مرحله ۳: تشخیص نوع پرس‌وجو و اجرا
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
            # جستجوی ساده: هر توکن به صورت OR ترکیب می‌شود سپس رتبه‌بندی
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

    # مرحله ۴: اعمال فیلتر فیلدها
    # فیلدهای ساختاری مستقیم جستجو می‌شوند؛ بقیه به عنوان کلمه کلیدی
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

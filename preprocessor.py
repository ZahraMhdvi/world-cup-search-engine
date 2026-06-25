
import re
import unicodedata
from typing import List



STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she',
    'they', 'them', 'their', 'our', 'your', 'his', 'her', 'which', 'who',
    'what', 'when', 'where', 'how', 'all', 'each', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'just', 'about', 'as', 'into',
    'through', 'during', 'up', 'down', 'out', 'off', 'over', 'under',
    'then', 'once', 'here', 'there', 'between', 'after', 'before',
    'while', 'also', 'if', 'because', 'although', 'however',
}

FOOTBALL_IMPORTANT_WORDS = {
    'goal', 'goals', 'penalty', 'penalties', 'red', 'yellow', 'card', 'cards',
    'own', 'extra', 'time', 'final', 'semi', 'quarter', 'round', 'group',
    'shootout', 'draw', 'won', 'lost', 'win', 'miss', 'scored', 'save',
    'substitution', 'substitute', 'referee', 'captain', 'manager', 'coach',
    'match', 'game', 'versus', 'vs', 'score', 'stadium', 'venue', 'host',
    'attendance', 'var', 'xg', 'assist',
}

EFFECTIVE_STOP_WORDS = STOP_WORDS - FOOTBALL_IMPORTANT_WORDS


def normalize_unicode(text: str) -> str:

    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def normalize_case(text: str) -> str:

    return text.lower()


def remove_punctuation(text: str) -> str:

    text = re.sub(r'-', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r"'", '', text)
    return text


def tokenize(text: str) -> List[str]:

    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def remove_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in EFFECTIVE_STOP_WORDS]


def normalize_token(token: str) -> str:

    token_map = {
        'penaltys': 'penalty',
        'penalties': 'penalty',
        'goals': 'goal',
        'cards': 'card',
        'matches': 'match',
        'games': 'game',
        'substitutions': 'substitution',
        'substitutes': 'substitute',
        'referees': 'referee',
        'managers': 'manager',
        'coaches': 'coach',
        'captains': 'captain',
        'scorers': 'scorer',
    }
    return token_map.get(token, token)


def preprocess(text: str, apply_stop_words: bool = True) -> List[str]:

    text = normalize_unicode(text)
    text = normalize_case(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    if apply_stop_words:
        tokens = remove_stop_words(tokens)
    tokens = [normalize_token(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def preprocess_query(query: str, apply_stop_words: bool = False) -> List[str]:
    return preprocess(query, apply_stop_words=apply_stop_words)

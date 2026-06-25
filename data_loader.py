
import pandas as pd
import ast
import re


def load_dataset(filepath: str, year: int = 2022) -> pd.DataFrame:

    df = pd.read_csv(filepath)
    df = df[df['Year'] == year].reset_index(drop=True)
    df['doc_id'] = df.index + 1
    return df


def clean_html_entities(text: str) -> str:

    if not isinstance(text, str):
        return ''
    text = re.sub(r'&rsquor;', "'", text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    return text


def parse_list_field(value) -> list:

    if pd.isna(value) or value == '':
        return []
    value = str(value)
    value = clean_html_entities(value)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [v.strip() for v in value.split('|') if v.strip()]


def extract_player_names_from_event_list(event_list: list) -> list:

    names = []
    for event in event_list:
        event = clean_html_entities(str(event))

        parts = re.split(r'\|', event)
        for i, part in enumerate(parts):
            part = part.strip()

            if part and not re.match(r'^\d', part) and ':' not in part and 'Assist' not in part and 'for' not in part:
                names.append(part)
    return names


def extract_players_from_substitution(event_list: list) -> list:

    names = []
    for event in event_list:
        event = clean_html_entities(str(event))

        parts = re.split(r'\|', event)
        for i, part in enumerate(parts):
            part = part.strip()
            if part and not re.match(r'^\d', part) and ':' not in part and part.lower() != 'for':
                names.append(part)
    return names


def extract_names_from_simple_field(value) -> list:

    if pd.isna(value) or value == '':
        return []
    value = clean_html_entities(str(value))

    parts = re.split(r'\|', value)
    names = []
    for part in parts:
        name = re.sub(r'\d.*$', '', part).strip().rstrip('·').strip()
        if name:
            names.append(name)
    return names


def extract_officials(officials_str) -> dict:

    result = {'referee': '', 'assistants': [], 'var': ''}
    if pd.isna(officials_str) or officials_str == '':
        return result
    officials_str = str(officials_str)

    parts = re.split(r'·', officials_str)
    for part in parts:
        part = part.strip()
        if '(Referee)' in part:
            result['referee'] = re.sub(r'\(Referee\)', '', part).strip()
        elif '(VAR)' in part:
            result['var'] = re.sub(r'\(VAR\)', '', part).strip()
        elif re.search(r'\(AR\d\)', part):
            name = re.sub(r'\(AR\d\)', '', part).strip()
            result['assistants'].append(name)
    return result

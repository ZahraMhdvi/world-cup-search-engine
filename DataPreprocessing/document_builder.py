
import pandas as pd
from DataPreprocessing.data_loader import (
    parse_list_field,
    extract_player_names_from_event_list,
    extract_players_from_substitution,
    extract_names_from_simple_field,
    extract_officials,
    clean_html_entities
)


def build_document_text(row: pd.Series) -> str:

    parts = []
    home = str(row.get('home_team', ''))
    away = str(row.get('away_team', ''))
    stage = str(row.get('Round', ''))
    venue_raw = str(row.get('Venue', ''))
    venue = clean_html_entities(venue_raw)
    date = str(row.get('Date', ''))
    score = str(row.get('Score', ''))
    year = str(row.get('Year', ''))
    host = str(row.get('Host', ''))

    parts.append(f"{home} vs {away}")
    parts.append(f"round {stage}")
    parts.append(f"stage {stage}")
    parts.append(f"venue {venue} stadium")
    parts.append(f"date {date} year {year} host {host}")
    parts.append(f"score {score}")


    try:
        h_score = int(row.get('home_score', -1))
        a_score = int(row.get('away_score', -1))
        if h_score == a_score:
            parts.append(f"draw tied equal {h_score}-{a_score} {h_score} {a_score}")
            if h_score == 0:
                parts.append("0-0 zero zero goalless")
        parts.append(f"score {h_score} {a_score}")
    except Exception:
        pass

    notes = clean_html_entities(str(row.get('Notes', '')))
    if notes and notes != 'nan':
        parts.append(notes)


    home_manager = clean_html_entities(str(row.get('home_manager', '')))
    away_manager = clean_html_entities(str(row.get('away_manager', '')))
    home_captain = clean_html_entities(str(row.get('home_captain', '')))
    away_captain = clean_html_entities(str(row.get('away_captain', '')))

    if home_manager and home_manager != 'nan':
        parts.append(f"manager coach {home_manager} {home}")
    if away_manager and away_manager != 'nan':
        parts.append(f"manager coach {away_manager} {away}")
    if home_captain and home_captain != 'nan':
        parts.append(f"captain {home_captain} {home}")
    if away_captain and away_captain != 'nan':
        parts.append(f"captain {away_captain} {away}")


    referee = clean_html_entities(str(row.get('Referee', '')))
    if referee and referee != 'nan':
        parts.append(f"referee {referee}")
    officials_info = extract_officials(str(row.get('Officials', '')))
    if officials_info['var']:
        parts.append(f"var referee {officials_info['var']}")


    home_goals = extract_names_from_simple_field(row.get('home_goal', ''))
    away_goals = extract_names_from_simple_field(row.get('away_goal', ''))
    home_goals_long = parse_list_field(row.get('home_goal_long', ''))
    away_goals_long = parse_list_field(row.get('away_goal_long', ''))

    all_scorers = home_goals + away_goals
    if all_scorers:
        parts.append("goal scorer " + ' '.join(all_scorers))

    home_scorer_names = extract_player_names_from_event_list(home_goals_long)
    away_scorer_names = extract_player_names_from_event_list(away_goals_long)
    if home_scorer_names:
        parts.append(f"goal {home} " + ' '.join(home_scorer_names))
    if away_scorer_names:
        parts.append(f"goal {away} " + ' '.join(away_scorer_names))


    home_own = clean_html_entities(str(row.get('home_own_goal', '')))
    away_own = clean_html_entities(str(row.get('away_own_goal', '')))
    if home_own and home_own != 'nan':
        parts.append(f"own goal {home} {home_own}")
    if away_own and away_own != 'nan':
        parts.append(f"own goal {away} {away_own}")


    home_pen_goal = extract_names_from_simple_field(row.get('home_penalty_goal', ''))
    away_pen_goal = extract_names_from_simple_field(row.get('away_penalty_goal', ''))
    if home_pen_goal:
        parts.append("penalty goal " + ' '.join(home_pen_goal))
    if away_pen_goal:
        parts.append("penalty goal " + ' '.join(away_pen_goal))

    home_pen_miss = parse_list_field(row.get('home_penalty_miss_long', ''))
    away_pen_miss = parse_list_field(row.get('away_penalty_miss_long', ''))
    if home_pen_miss:
        miss_names = extract_player_names_from_event_list(home_pen_miss)
        if miss_names:
            parts.append("penalty miss " + ' '.join(miss_names))
    if away_pen_miss:
        miss_names = extract_player_names_from_event_list(away_pen_miss)
        if miss_names:
            parts.append("penalty miss " + ' '.join(miss_names))



    home_shootout_goal = parse_list_field(row.get('home_penalty_shootout_goal_long', ''))
    away_shootout_goal = parse_list_field(row.get('away_penalty_shootout_goal_long', ''))
    home_shootout_miss = parse_list_field(row.get('home_penalty_shootout_miss_long', ''))
    away_shootout_miss = parse_list_field(row.get('away_penalty_shootout_miss_long', ''))

    if home_shootout_goal or away_shootout_goal:
        parts.append("penalty shootout penalties")
        shootout_names = (
            extract_player_names_from_event_list(home_shootout_goal) +
            extract_player_names_from_event_list(away_shootout_goal) +
            extract_player_names_from_event_list(home_shootout_miss) +
            extract_player_names_from_event_list(away_shootout_miss)
        )
        if shootout_names:
            parts.append("shootout " + ' '.join(shootout_names))


    home_red = clean_html_entities(str(row.get('home_red_card', '')))
    away_red = clean_html_entities(str(row.get('away_red_card', '')))
    if home_red and home_red != 'nan':
        parts.append(f"red card {home} {home_red}")
    if away_red and away_red != 'nan':
        parts.append(f"red card {away} {away_red}")


    home_yr = clean_html_entities(str(row.get('home_yellow_red_card', '')))
    away_yr = clean_html_entities(str(row.get('away_yellow_red_card', '')))
    if home_yr and home_yr != 'nan':
        parts.append(f"yellow red card {home} {home_yr}")
    if away_yr and away_yr != 'nan':
        parts.append(f"yellow red card {away} {away_yr}")


    home_yellow = parse_list_field(row.get('home_yellow_card_long', ''))
    away_yellow = parse_list_field(row.get('away_yellow_card_long', ''))
    if home_yellow:
        yellow_names = extract_player_names_from_event_list(home_yellow)
        if yellow_names:
            parts.append(f"yellow card {home} " + ' '.join(yellow_names))
    if away_yellow:
        yellow_names = extract_player_names_from_event_list(away_yellow)
        if yellow_names:
            parts.append(f"yellow card {away} " + ' '.join(yellow_names))


    home_subs = parse_list_field(row.get('home_substitute_in_long', ''))
    away_subs = parse_list_field(row.get('away_substitute_in_long', ''))
    if home_subs:
        sub_names = extract_players_from_substitution(home_subs)
        if sub_names:
            parts.append(f"substitution substitute {home} " + ' '.join(sub_names))
    if away_subs:
        sub_names = extract_players_from_substitution(away_subs)
        if sub_names:
            parts.append(f"substitution substitute {away} " + ' '.join(sub_names))


    home_xg = str(row.get('home_xg', ''))
    away_xg = str(row.get('away_xg', ''))
    if home_xg and home_xg != 'nan':
        parts.append(f"xg expected goals {home} {home_xg}")
    if away_xg and away_xg != 'nan':
        parts.append(f"xg expected goals {away} {away_xg}")


    attendance = str(row.get('Attendance', ''))
    if attendance and attendance != 'nan':
        parts.append(f"attendance {attendance}")

    return ' '.join(filter(None, parts))


def build_document_fields(row: pd.Series) -> dict:

    home = str(row.get('home_team', ''))
    away = str(row.get('away_team', ''))
    return {
        'team': [home.lower(), away.lower()],
        'round': str(row.get('Round', '')).lower(),
        'stage': str(row.get('Round', '')).lower(),
        'referee': clean_html_entities(str(row.get('Referee', ''))).lower(),
        'venue': clean_html_entities(str(row.get('Venue', ''))).lower(),
        'year': str(row.get('Year', '')),
        'home_team': home.lower(),
        'away_team': away.lower(),
    }


def build_all_documents(df: pd.DataFrame) -> list:

    documents = []
    for _, row in df.iterrows():
        doc_text = build_document_text(row)
        doc_fields = build_document_fields(row)
        doc = {
            'doc_id': int(row['doc_id']),
            'text': doc_text,
            'fields': doc_fields,
            'home_team': str(row.get('home_team', '')),
            'away_team': str(row.get('away_team', '')),
            'stage': str(row.get('Round', '')),
            'date': str(row.get('Date', '')),
            'score': str(row.get('Score', '')),
            'venue': clean_html_entities(str(row.get('Venue', ''))),
            'referee': clean_html_entities(str(row.get('Referee', ''))),
            'notes': clean_html_entities(str(row.get('Notes', ''))),
        }
        documents.append(doc)
    return documents

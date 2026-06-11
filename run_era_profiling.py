"""
Era-based bias profiling runner.

Runs the full bias profiling pipeline twice:
  Era 2 — Ranil/UNP government  (Dec 2023 – Sep 22 2024)
  Era 3 — NPP/AKD government    (Sep 23 2024 – present)

Each era uses its own politician classification (who was government at the time)
so that the resulting bias scores are calibrated to the political context of
that period, not the current one.

Outputs:
  data/bias_profiles_era2.json
  data/bias_profiles_era3.json
  data/bias_profiles_era_comparison.json  (side-by-side summary)
"""

import json
import sys
import numpy as np
import pandas as pd

sys.path.append('src')

from bias_profiling.article_grouper import ArticleGrouper
from bias_profiling.bias_analyzer  import BiasAnalyzer
from bias_profiling.bias_scorer    import BiasScorer

# ---------------------------------------------------------------------------
# Era definitions
# ---------------------------------------------------------------------------
ERAS = {
    'era2': {
        'label':      'Ranil/UNP Government',
        'start_date': '2023-12-10',   # start of available data
        'end_date':   '2024-09-22',   # day before NPP election result
        'politician_file': 'data/sri_lankan_politicians_era2.json',
        'output_file':     'data/bias_profiles_era2.json',
        'groups_file':     'data/article_groups_era2.json',
        'matrix_file':     'data/bias_matrix_era2.json',
        'political_csv':   'data/political_articles_era2.csv',
    },
    'era3': {
        'label':      'NPP/AKD Government',
        'start_date': '2024-09-23',   # NPP election win
        'end_date':   '2030-01-01',   # far future = all remaining data
        'politician_file': 'data/sri_lankan_politicians_era3.json',
        'output_file':     'data/bias_profiles_era3.json',
        'groups_file':     'data/article_groups_era3.json',
        'matrix_file':     'data/bias_matrix_era3.json',
        'political_csv':   'data/political_articles_era3.csv',
    },
}

# ---------------------------------------------------------------------------
# Era-specific politician classifications
# ---------------------------------------------------------------------------
ERA_POLITICIANS = {
    'era2': {
        # Ranil Wickremesinghe became president Jul 2022.
        # Government: UNP core + SLPP ministers who backed Ranil.
        'government': [
            'ranil wickremesinghe', 'ranil', 'rw',
            'vajira abeywardena', 'vajira',
            'ruwan wijewardene', 'ruwan wijewardene',
            'malik samarawickrama', 'malik',
            'dinesh gunawardena', 'dinesh gunawardena',   # PM under Ranil
            'nimal siripala de silva', 'nimal siripala',  # backed Ranil
            'sarath fonseka', 'sarath',
        ],
        # Opposition: SJB, NPP/JVP, and SLPP factions that opposed Ranil.
        'opposition': [
            # SJB
            'sajith premadasa', 'sajith', 'premadasa',
            'eran wickramaratne', 'eran',
            'harsha de silva', 'harsha',
            'patali champika ranawaka', 'champika ranawaka', 'patali champika',
            'kabir hashim', 'kabir',
            'nalin bandara', 'nalin bandara',
            # NPP / JVP
            'anura kumara dissanayake', 'akd', 'anura dissanayake',
            'harini amarasuriya', 'harini',
            'vijitha herath', 'vijitha',
            'tilvin silva', 'tilvin',
            'bimal ratnayake', 'bimal',
            'sunil handunnetti', 'sunil handunnetti',
            'wasantha mudalige', 'wasantha',
            # SLPP opposition faction
            'mahinda rajapaksa', 'mahinda', 'mr',
            'gotabaya rajapaksa', 'gotabaya', 'gota', 'gr',
            'basil rajapaksa', 'basil',
            'namal rajapaksa', 'namal',
            'chamal rajapaksa', 'chamal',
            'g.l. peiris', 'gl peiris', 'professor peiris',
            'dullas alahapperuma', 'dullas',
            'johnston fernando', 'johnston',
            'wimal weerawansa', 'wimal',
            'udaya gammanpila', 'udaya gammanpila',
            # Other
            'maithripala sirisena', 'maithripala', 'ms',
            'chandrika bandaranaike kumaratunga', 'chandrika', 'cbk',
            'karu jayasuriya', 'karu',
        ],
    },
    'era3': {
        # NPP/AKD won election Sep 23 2024.
        'government': [
            'anura kumara dissanayake', 'akd', 'anura dissanayake',
            'harini amarasuriya', 'harini',
            'vijitha herath', 'vijitha',
            'tilvin silva', 'tilvin',
            'bimal ratnayake', 'bimal',
            'sunil handunnetti', 'sunil handunnetti',
            'nalaka godahewa', 'nalaka',
            'wasantha mudalige', 'wasantha',
        ],
        'opposition': [
            # SJB
            'sajith premadasa', 'sajith', 'premadasa',
            'eran wickramaratne', 'eran',
            'harsha de silva', 'harsha',
            'patali champika ranawaka', 'champika ranawaka', 'patali champika',
            'kabir hashim', 'kabir',
            # SLPP
            'mahinda rajapaksa', 'mahinda', 'mr',
            'gotabaya rajapaksa', 'gotabaya', 'gota', 'gr',
            'basil rajapaksa', 'basil',
            'namal rajapaksa', 'namal',
            'chamal rajapaksa', 'chamal',
            'g.l. peiris', 'gl peiris', 'professor peiris',
            'dullas alahapperuma', 'dullas',
            'johnston fernando', 'johnston',
            'wimal weerawansa', 'wimal',
            'udaya gammanpila', 'udaya gammanpila',
            # UNP (now opposition)
            'ranil wickremesinghe', 'ranil', 'rw',
            'vajira abeywardena', 'vajira',
            'ruwan wijewardene',
            'malik samarawickrama', 'malik',
            # Other
            'maithripala sirisena', 'maithripala', 'ms',
            'chandrika bandaranaike kumaratunga', 'chandrika', 'cbk',
            'sarath fonseka', 'sarath',
            'diana gamage', 'diana',
        ],
    },
}


def save_politician_file(era_key: str, filepath: str):
    pol = ERA_POLITICIANS[era_key]
    politicians = {}
    for name in pol['government']:
        politicians[name] = 'government'
    for name in pol['opposition']:
        politicians[name] = 'opposition'

    data = {
        'politicians': politicians,
        'government': pol['government'],
        'opposition': pol['opposition'],
        'total_count': len(politicians),
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved politician file -> {filepath}")
    print(f"  Government: {len(pol['government'])} names, Opposition: {len(pol['opposition'])} names")


def filter_and_save_political_articles(era_cfg: dict, pol_file: str) -> str:
    """Filter main CSV by date range, then filter for political articles."""
    df = pd.read_csv('data/filtered_english_articles.csv')
    df['date_str'] = pd.to_datetime(df['date_str'])

    mask = (
        (df['date_str'] >= era_cfg['start_date']) &
        (df['date_str'] <= era_cfg['end_date'])
    )
    df_era = df[mask].copy()
    print(f"  Date-filtered: {len(df_era)} articles "
          f"({df_era['date_str'].min().date()} -> {df_era['date_str'].max().date()})")
    print(f"  Per source:\n{df_era['source'].value_counts().to_string()}")

    # Load politician names for this era
    with open(pol_file, 'r', encoding='utf-8') as f:
        pol_data = json.load(f)
    politician_names = list(pol_data['politicians'].keys())

    def contains_politician(text):
        if pd.isna(text):
            return False
        t = str(text).lower()
        return any(p in t for p in politician_names)

    mask_pol = df_era['description'].apply(contains_politician)
    df_political = df_era[mask_pol].copy()
    print(f"  Political articles: {len(df_political)} "
          f"({len(df_political)/len(df_era)*100:.1f}% of era total)")

    df_political.to_csv(era_cfg['political_csv'], index=False)
    print(f"  Saved -> {era_cfg['political_csv']}")
    return era_cfg['political_csv']


def run_era(era_key: str):
    era_cfg = ERAS[era_key]
    print(f"\n{'='*60}")
    print(f"ERA: {era_cfg['label']}  ({era_cfg['start_date']} to {era_cfg['end_date']})")
    print(f"{'='*60}")

    # Step 1 — politician file
    print("\n[1/4] Building era-specific politician file...")
    save_politician_file(era_key, era_cfg['politician_file'])

    # Step 2 — filter articles
    print("\n[2/4] Filtering political articles for this era...")
    filter_and_save_political_articles(era_cfg, era_cfg['politician_file'])

    # Step 3 — group articles
    print("\n[3/4] Grouping related articles...")
    df_pol = pd.read_csv(era_cfg['political_csv'])
    grouper = ArticleGrouper(similarity_threshold=0.25)
    article_groups = grouper.group_related_articles(df_pol)
    if not article_groups:
        print("  No article groups formed — skipping era.")
        return None
    grouper.save_groups(article_groups, era_cfg['groups_file'])

    # Step 4 — ABSA bias matrix
    print("\n[4/4] Running ABSA bias analysis (this takes a while)...")
    analyzer = BiasAnalyzer(politician_file=era_cfg['politician_file'])
    bias_matrix, sources = analyzer.calculate_pairwise_bias(article_groups)
    if bias_matrix is None:
        print("  Bias matrix failed — skipping era.")
        return None
    analyzer.save_bias_matrix(bias_matrix, sources, era_cfg['matrix_file'])

    # Step 5 — score and save
    scorer = BiasScorer()
    scorer.bias_matrix = bias_matrix
    scorer.sources = sources
    graph = scorer.build_bias_graph()
    acyclic = scorer.remove_cycles(graph)
    scores = scorer.calculate_bias_scores(acyclic)
    interpreted = scorer.interpret_bias_scores(scores)
    scorer.save_bias_profiles(interpreted, era_cfg['output_file'],
                              political_csv=era_cfg['political_csv'])
    scorer.print_results(interpreted)
    return interpreted


def compare_eras(results: dict):
    """Print and save a side-by-side comparison of all eras."""
    print(f"\n{'='*70}")
    print("ERA COMPARISON — Bias score shift per source")
    print(f"{'='*70}")

    all_sources = sorted({s for r in results.values() for s in r.keys()})
    header = f"{'Source':<22}"
    for era_key, era_cfg in ERAS.items():
        if era_key in results:
            header += f"  {era_cfg['label'][:20]:<20}"
    print(header)
    print("-" * 70)

    comparison = {}
    for source in all_sources:
        row = f"{source:<22}"
        comparison[source] = {}
        for era_key in ERAS:
            if era_key in results and source in results[era_key]:
                score = results[era_key][source]['bias_score']
                row += f"  {score:>+7.1f}             "
                comparison[source][era_key] = score
            else:
                row += f"  {'N/A':>7}             "
                comparison[source][era_key] = None
        print(row)

    # Save comparison
    out = {
        'eras': {k: v['label'] for k, v in ERAS.items()},
        'sources': comparison,
    }
    with open('data/bias_profiles_era_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> data/bias_profiles_era_comparison.json")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--era', choices=['era2', 'era3', 'all'], default='all',
                        help='Which era(s) to run (default: all)')
    args = parser.parse_args()

    eras_to_run = list(ERAS.keys()) if args.era == 'all' else [args.era]

    results = {}
    for era_key in eras_to_run:
        result = run_era(era_key)
        if result:
            results[era_key] = result

    if len(results) > 1:
        compare_eras(results)

    print("\nDone.")

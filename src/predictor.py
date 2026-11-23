"""Prediction generation, pool export, and automated verification pipeline for SSQ lottery."""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import features as feat_module
from src import game_theory, model, storage
from collections import Counter

logger = logging.getLogger('ssq.predictor')

# Hetu Wuxing mapping for Blue balls (01-16)
HETU_BLUE_BUCKETS = {
    '水': [1, 6, 11, 16],
    '火': [2, 7, 12],
    '木': [3, 8, 13],
    '金': [4, 9, 14],
    '土': [5, 10, 15],
}


def get_next_draw_dates(n: int = 1, from_date: Optional[datetime] = None) -> List[datetime]:
    """Calculate the next n official SSQ draw dates (Tuesdays, Thursdays, Sundays)."""
    if from_date is None:
        from_date = datetime.now()
    dates: List[datetime] = []
    current = from_date
    while len(dates) < n:
        if current.date() > from_date.date() and current.weekday() in (1, 3, 6):
            dates.append(current)
        current += timedelta(days=1)
    return dates


def estimate_issue_number(
    draw_date: datetime,
    latest_issue: Optional[str] = None,
    latest_date: Optional[str] = None,
) -> str:
    """Estimate upcoming 5-digit SSQ issue number."""
    year_code = draw_date.year % 100
    if not latest_issue or not latest_date:
        return f"{year_code:02d}001"
    
    try:
        latest_year = int(str(latest_issue)[:2])
        latest_num = int(str(latest_issue)[2:])
        if year_code != latest_year:
            return f"{year_code:02d}001"
        else:
            return f"{year_code:02d}{latest_num + 1:03d}"
    except (ValueError, IndexError):
        return f"{year_code:02d}001"


def generate_predictions(
    n_combinations: int = 5,
    n_draws: int = 1,
    from_date: Optional[datetime] = None,
    mode: str = 'balanced',
    beta: float = config.GAME_THEORY_BETA,
    max_ball_freq: int = config.MAX_BALL_FREQ,
) -> List[Dict[str, Any]]:
    """Train ensemble model on latest historical data and generate game-theoretic predictions."""
    storage.init_db()
    df = storage.get_all_draws()
    if df.empty or len(df) < 50:
        logger.error('Insufficient draw data for predictions')
        return []

    latest_issue = storage.get_latest_issue()
    latest_row = df[df['issue'] == latest_issue].iloc[0] if latest_issue else df.iloc[-1]
    latest_date_str = latest_row['draw_date']

    target_dates = get_next_draw_dates(n_draws, from_date)
    ensemble = model.EnsembleModel()
    ensemble.fit(df)

    filter_obj = game_theory.AntiPopularityFilter(history_df=df)
    scorer_obj = game_theory.UniquenessScorer(history_df=df)

    all_predictions = []
    current_issue = latest_issue
    current_date_str = latest_date_str

    for d in target_dates:
        t_issue = estimate_issue_number(d, current_issue, current_date_str)
        combos, stats = game_theory.select_game_theory_combinations(
            ensemble,
            n_combinations=n_combinations,
            mode=mode,
            beta=beta,
            max_ball_freq=max_ball_freq,
            pool_size=300,
            df=df,
            filter_obj=filter_obj,
            scorer_obj=scorer_obj,
        )

        db_records = []
        for c in combos:
            reds = c['reds']
            odd, even = feat_module.compute_odd_even_ratio(reds)
            db_records.append({
                'predict_date': datetime.now().strftime('%Y-%m-%d'),
                'target_issue': t_issue,
                'target_date': d.strftime('%Y-%m-%d'),
                'red_1': reds[0], 'red_2': reds[1], 'red_3': reds[2],
                'red_4': reds[3], 'red_5': reds[4], 'red_6': reds[5],
                'blue': c['blue'],
                'score': c.get('final_score', c.get('model_score', 0.5)),
                'strategy': f"ensemble_gt_{mode}",
                'sum_value': int(sum(reds)),
                'odd_even_ratio': f"{odd}:{even}",
            })

        storage.save_predictions(db_records)
        sync_predictions_csv()

        all_predictions.append({
            'target_issue': t_issue,
            'target_date': d.strftime('%Y-%m-%d'),
            'combinations': combos,
            'stats': stats,
        })
        current_issue = t_issue
        current_date_str = d.strftime('%Y-%m-%d')

    return all_predictions


def sync_predictions_csv(output_path: Optional[str] = None):
    """Sync full predictions table to reports/predictions.csv with UTF-8-SIG."""
    if output_path is None:
        output_path = os.path.join(config.REPORTS_DIR, 'predictions.csv')
    df = storage.get_all_predictions()
    if not df.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_predictions_csv(predictions: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Save formatted predictions to CSV."""
    sync_predictions_csv(output_path)


def print_predictions(predictions_by_draw: List[Dict[str, Any]]):
    """Print formatted prediction tables."""
    for p in predictions_by_draw:
        print(f"\n🔮 双色球预测目标期号: {p['target_issue']} (预计开奖: {p['target_date']})")
        table_data = []
        for i, c in enumerate(p['combinations']):
            reds_str = ' '.join(f"{r:02d}" for r in c['reds'])
            tags_str = ' / '.join(c.get('tags', []))
            table_data.append([
                f"第 {i+1} 注",
                reds_str,
                f"{c['blue']:02d}",
                f"{c.get('model_score', 0.0):.4f}",
                f"{c.get('uniqueness_score', 0.0):.4f}",
                f"{c.get('final_score', 0.0):.4f}",
                tags_str
            ])
        print(tabulate(
            table_data,
            headers=['序号', '红球推荐 (6码)', '蓝球', '模型分', '独享分', '综合分', '反大众标签'],
            tablefmt='grid'
        ))


def run_check_and_report() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Check unverified predictions against latest draws and return results and summary."""
    storage.init_db()
    newly_checked = storage.check_and_update_predictions()
    summary = storage.get_checked_predictions_summary()
    sync_predictions_csv()

    if newly_checked:
        print(f"\n🎯 自动核对完成！新核对 {len(newly_checked)} 注预测：")
        check_table = []
        for c in newly_checked:
            check_table.append([
                c['target_issue'],
                f"红{c['red_hit']}+蓝{c['blue_hit']}",
                c['prize_level'],
                f"¥{c['prize_amount']}"
            ])
        print(tabulate(check_table, headers=['期号', '命中情况', '中奖等级', '中奖金额'], tablefmt='grid'))

    print(f"\n📊 实盘累计总投入: ¥{summary['total_investment']} | 累计总中奖: ¥{summary['total_winnings']} | 净收益: ¥{summary['net_return']} (ROI: {summary['roi_pct']}%)")
    return newly_checked, summary


def check_past_predictions() -> List[Dict[str, Any]]:
    """Alias for automated scheduler backward compatibility."""
    newly_checked, _ = run_check_and_report()
    return newly_checked


def export_frozen_candidate_pool(
    output_path: Optional[str] = None,
    backup_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Export top-20 candidate pool (balanced) and 5-element bucket pool (jackpot) to JSON.
    
    Balanced pool: 20 tickets strictly covering all 16 blues (1-16), mainstream parity >= 75%,
                   single ball freq <= 5, overlap <= 3.
    Jackpot pool: 5 Hetu Wuxing buckets x 8 tickets = 40 frozen tickets with single ball freq <= 3.
    """
    config.ensure_dirs()
    if random_seed is not None:
        np.random.seed(random_seed)

    if output_path is None:
        output_path = os.path.join(config.DOCS_DATA_DIR, 'pool.json')
    if backup_path is None:
        backup_path = os.path.join(config.REPORTS_DIR, 'latest_pool.json')

    storage.init_db()
    df = storage.get_all_draws()
    if df.empty or len(df) < 50:
        raise RuntimeError('数据库历史开奖记录不足，无法导出候选池')

    latest_issue = storage.get_latest_issue()
    latest_row = df[df['issue'] == latest_issue].iloc[0]
    target_dates = get_next_draw_dates(1)
    target_date = target_dates[0]
    target_issue = estimate_issue_number(target_date, latest_issue, latest_row['draw_date'])
    weekday_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][target_date.weekday()]

    # Train ensemble model
    ensemble = model.EnsembleModel()
    ensemble.fit(df)

    filter_obj = game_theory.AntiPopularityFilter(history_df=df)
    scorer_obj = game_theory.UniquenessScorer(history_df=df)

    # 1. Sample 1,500 valid candidate combinations uniformly across all 16 blues
    raw_candidates: List[Dict[str, Any]] = []
    seen_combos: Set[Tuple[Tuple[int, ...], int]] = set()
    target_candidates = 1500
    attempts = 0
    max_attempts = 15000

    red_nums = list(range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1))
    red_probs = ensemble.get_probabilities()['red']
    red_weights = np.array([red_probs.get(n, 1.0 / 33.0) for n in red_nums], dtype=float)
    red_weights = np.maximum(red_weights, 1e-6)
    red_weights /= red_weights.sum()

    while len(raw_candidates) < target_candidates and attempts < max_attempts:
        attempts += 1
        # Cycle through 1..16 so every blue gets equal representation (~94 candidates)
        b = (len(raw_candidates) % 16) + 1

        avail_nums = red_nums.copy()
        avail_w = red_weights.copy()
        reds = []
        for _ in range(6):
            w = avail_w / avail_w.sum()
            idx = int(np.random.choice(len(avail_nums), p=w))
            reds.append(avail_nums[idx])
            avail_nums.pop(idx)
            avail_w = np.delete(avail_w, idx)
        reds.sort()

        combo_key = (tuple(reds), b)
        if combo_key in seen_combos:
            continue

        is_pass, _ = filter_obj.is_valid(reds, b)
        if not is_pass:
            continue

        model_s = ensemble.score_combination(reds, b)
        uniq_b = scorer_obj.score(reds, b, mode='balanced')
        uniq_j = scorer_obj.score(reds, b, mode='jackpot')
        odd, even = feat_module.compute_odd_even_ratio(reds)

        raw_candidates.append({
            'reds': reds,
            'blue': b,
            'model_score': round(model_s, 4),
            'u_score_balanced': round(uniq_b['uniqueness_score'], 4),
            'tags_balanced': uniq_b['tags'],
            'u_score_jackpot': round(uniq_j['uniqueness_score'], 4),
            'tags_jackpot': uniq_j['tags'],
            'odd_count': odd,
            'even_count': even,
            'sum_val': int(sum(reds)),
            'features': {
                'sum_value': int(sum(reds)),
                'odd_even_ratio': f'{odd}:{even}',
                'ac_value': int(feat_module.compute_ac_value(reds)),
                'consecutive': int(feat_module.compute_consecutive_count(reds)),
            }
        })
        seen_combos.add(combo_key)

    # Normalize scores across candidates for both balanced and jackpot
    m_scores = [c['model_score'] for c in raw_candidates]
    ub_scores = [c['u_score_balanced'] for c in raw_candidates]
    uj_scores = [c['u_score_jackpot'] for c in raw_candidates]

    min_m, max_m = min(m_scores), max(m_scores)
    min_ub, max_ub = min(ub_scores), max(ub_scores)
    min_uj, max_uj = min(uj_scores), max(uj_scores)

    denom_m = (max_m - min_m) + 1e-9
    denom_ub = (max_ub - min_ub) + 1e-9
    denom_uj = (max_uj - min_uj) + 1e-9

    for c in raw_candidates:
        norm_m = (c['model_score'] - min_m) / denom_m
        norm_ub = (c['u_score_balanced'] - min_ub) / denom_ub
        norm_uj = (c['u_score_jackpot'] - min_uj) / denom_uj
        c['score_balanced'] = round(float((1.0 - config.GAME_THEORY_BETA) * norm_m + config.GAME_THEORY_BETA * norm_ub), 4)
        c['score_jackpot'] = round(float((1.0 - config.GAME_THEORY_BETA) * norm_m + config.GAME_THEORY_BETA * norm_uj), 4)

    # -------------------------------------------------------------
    # 2. Balanced 20-ticket selection (16-blue full coverage guarantee)
    # -------------------------------------------------------------
    cands_by_blue = {b: [c for c in raw_candidates if c['blue'] == b] for b in range(1, 17)}
    for b in range(1, 17):
        cands_by_blue[b].sort(key=lambda x: x['score_balanced'], reverse=True)

    sorted_blues = sorted(range(1, 17), key=lambda b: cands_by_blue[b][0]['score_balanced'] if cands_by_blue[b] else 0, reverse=True)

    balanced_selected: List[Dict[str, Any]] = []
    for max_ov in [3, 4, 5]:
        for max_red_f in [4, 5]:
            for enforce_parity in [True, False]:
                selected: List[Dict[str, Any]] = []
                seen_red_tuples: Set[Tuple[int, ...]] = set()
                red_counts: Counter = Counter()
                blue_counts: Counter = Counter()

                # Phase 1: Pick 1 best ticket for each of the 16 blue balls
                success_phase1 = True
                for b in sorted_blues:
                    found = False
                    for c in cands_by_blue[b]:
                        red_tup = tuple(c['reds'])
                        if red_tup in seen_red_tuples:
                            continue
                        if enforce_parity and c['odd_count'] not in (2, 3, 4) and sum(1 for x in selected if x['odd_count'] in (2, 3, 4)) < 12:
                            continue
                        if any(red_counts[r] >= max_red_f for r in c['reds']):
                            continue
                        if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in selected):
                            continue
                        selected.append(c)
                        seen_red_tuples.add(red_tup)
                        blue_counts[b] += 1
                        for r in c['reds']:
                            red_counts[r] += 1
                        found = True
                        break
                    if not found:
                        for c in cands_by_blue[b]:
                            red_tup = tuple(c['reds'])
                            if red_tup in seen_red_tuples:
                                continue
                            if any(red_counts[r] >= max_red_f for r in c['reds']):
                                continue
                            if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in selected):
                                continue
                            selected.append(c)
                            seen_red_tuples.add(red_tup)
                            blue_counts[b] += 1
                            for r in c['reds']:
                                red_counts[r] += 1
                            found = True
                            break
                    if not found:
                        success_phase1 = False
                        break

                if not success_phase1 or len(selected) != 16:
                    continue

                # Phase 2: Pick 4 remaining tickets from top candidates (blue_counts < 2)
                sorted_all_b = sorted(raw_candidates, key=lambda x: x['score_balanced'], reverse=True)
                for c in sorted_all_b:
                    if len(selected) == 20:
                        break
                    red_tup = tuple(c['reds'])
                    if red_tup in seen_red_tuples:
                        continue
                    if blue_counts[c['blue']] >= 2:
                        continue
                    if any(red_counts[r] >= max_red_f for r in c['reds']):
                        continue
                    if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in selected):
                        continue
                    if enforce_parity and c['odd_count'] not in (2, 3, 4):
                        mainstream_count = sum(1 for x in selected if x['odd_count'] in (2, 3, 4))
                        if mainstream_count < 15 and (20 - len(selected) <= 15 - mainstream_count):
                            continue

                    selected.append(c)
                    seen_red_tuples.add(red_tup)
                    blue_counts[c['blue']] += 1
                    for r in c['reds']:
                        red_counts[r] += 1

                if len(selected) == 20:
                    mainstream_pct = sum(1 for x in selected if x['odd_count'] in (2, 3, 4)) / 20.0
                    unique_blues = len(set(x['blue'] for x in selected))
                    if unique_blues == 16 and (mainstream_pct >= 0.75 or not enforce_parity):
                        balanced_selected = selected
                        break
            if len(balanced_selected) == 20:
                break
        if len(balanced_selected) == 20:
            break

    # Fallback if balanced pool still has fewer than 20
    if len(balanced_selected) < 20:
        seen_reds = {tuple(x['reds']) for x in balanced_selected}
        sorted_all_b = sorted(raw_candidates, key=lambda x: x['score_balanced'], reverse=True)
        for c in sorted_all_b:
            if tuple(c['reds']) not in seen_reds:
                balanced_selected.append(c)
                seen_reds.add(tuple(c['reds']))
                if len(balanced_selected) == 20:
                    break

    # Sort balanced Top 20 by score_balanced descending
    balanced_selected.sort(key=lambda x: x['score_balanced'], reverse=True)

    # -------------------------------------------------------------
    # 3. Jackpot 5-Element Blue Buckets
    # -------------------------------------------------------------
    blue_model_probs = [ensemble.get_probabilities()['blue'].get(b, 1.0 / 16.0) for b in range(1, 17)]
    blue_uniqueness = []
    for b in range(1, 17):
        if b in game_theory.POPULAR_BLUES:
            bu = 0.20
        elif b in game_theory.AVOIDED_BLUES:
            bu = 0.85
        else:
            bu = 0.50
        blue_uniqueness.append(bu)
    min_bm, max_bm = min(blue_model_probs), max(blue_model_probs)
    min_bu, max_bu = min(blue_uniqueness), max(blue_uniqueness)
    denom_bm = (max_bm - min_bm) + 1e-9
    denom_bu = (max_bu - min_bu) + 1e-9
    blue_composite = {
        b: round(float((1.0 - config.GAME_THEORY_BETA) * ((blue_model_probs[b - 1] - min_bm) / denom_bm) +
                       config.GAME_THEORY_BETA * ((blue_uniqueness[b - 1] - min_bu) / denom_bu)), 4)
        for b in range(1, 17)
    }

    def format_pool_ticket(rank: int, item: Dict[str, Any], mode: str = 'balanced') -> Dict[str, Any]:
        reds = sorted(item['reds'])
        odd, even = feat_module.compute_odd_even_ratio(reds)
        u_s = item['u_score_balanced'] if mode == 'balanced' else item['u_score_jackpot']
        c_s = item['score_balanced'] if mode == 'balanced' else item['score_jackpot']
        t_tags = item['tags_balanced'] if mode == 'balanced' else item['tags_jackpot']
        return {
            'rank': rank,
            'reds': reds,
            'blue': int(item['blue']),
            'model_score': round(float(item.get('model_score', 0.5)), 4),
            'uniqueness_score': round(float(u_s), 4),
            'composite_score': round(float(c_s), 4),
            'sum_val': int(sum(reds)),
            'odd_even': f"{odd}:{even}",
            'ac_val': int(feat_module.compute_ac_value(reds)),
            'consecutive': int(feat_module.compute_consecutive_count(reds)),
            'tags': t_tags,
        }

    jackpot_pool: Dict[str, Any] = {}
    for elem, elem_blues in HETU_BLUE_BUCKETS.items():
        # Representative blue for this element
        rep_blue = max(elem_blues, key=lambda b: blue_composite[b])
        rep_blue_score = blue_composite[rep_blue]

        blue_cands = [c for c in raw_candidates if c['blue'] == rep_blue]
        blue_cands.sort(key=lambda x: x['score_jackpot'], reverse=True)

        bucket_selected: List[Dict[str, Any]] = []
        # Prefer low pairwise red overlap (1 -> 2 -> 3 -> 4) to maximize pairwise overlap <= 1 for Meihua DFS
        for max_ov in [1, 2, 3, 4]:
            for max_red_f in [2, 3, 4]:
                bucket_selected = []
                seen_bucket_reds: Set[Tuple[int, ...]] = set()
                red_counts = Counter()
                for c in blue_cands:
                    red_tup = tuple(c['reds'])
                    if red_tup in seen_bucket_reds:
                        continue
                    if any(red_counts[r] >= max_red_f for r in c['reds']):
                        continue
                    if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in bucket_selected):
                        continue
                    bucket_selected.append(c)
                    seen_bucket_reds.add(red_tup)
                    for r in c['reds']:
                        red_counts[r] += 1
                    if len(bucket_selected) == 8:
                        break
                if len(bucket_selected) == 8:
                    break
            if len(bucket_selected) == 8:
                break

        if len(bucket_selected) < 8:
            seen_bucket_reds = {tuple(x['reds']) for x in bucket_selected}
            for c in blue_cands:
                red_tup = tuple(c['reds'])
                if red_tup not in seen_bucket_reds:
                    bucket_selected.append(c)
                    seen_bucket_reds.add(red_tup)
                    if len(bucket_selected) == 8:
                        break

        bucket_tickets = [
            format_pool_ticket(i + 1, item, mode='jackpot')
            for i, item in enumerate(bucket_selected[:8])
        ]

        jackpot_pool[elem] = {
            'element': elem,
            'blue': int(rep_blue),
            'blue_score': rep_blue_score,
            'tickets': bucket_tickets,
        }

    balanced_pool = [
        format_pool_ticket(i + 1, item, mode='balanced')
        for i, item in enumerate(balanced_selected[:20])
    ]

    # Baseline validation before export
    from src.meihua import validate_selected_tickets
    val_b = validate_selected_tickets(balanced_pool, mode='balanced', expected_count=20, max_ball_freq=5, max_pairwise_overlap=5)
    if not val_b['is_baseline_valid']:
        raise ValueError(f"Balanced candidate pool failed baseline validation: {val_b['baseline_issues']}")
    for elem, bdata in jackpot_pool.items():
        val_j = validate_selected_tickets(bdata['tickets'], mode='jackpot', expected_count=8, max_ball_freq=4, max_pairwise_overlap=5)
        if not val_j['is_baseline_valid']:
            raise ValueError(f"Jackpot bucket {elem} failed baseline validation: {val_j['baseline_issues']}")

    # Extract recent 10 draws for client-side auto-reconciliation
    recent_df = df.tail(10).iloc[::-1]
    recent_draws = []
    for _, r in recent_df.iterrows():
        recent_draws.append({
            'issue': str(r['issue']),
            'draw_date': str(r['draw_date']),
            'reds': [int(x) for x in feat_module.get_red_balls_from_row(r)],
            'blue': int(r['blue']),
        })

    # Note: redundant meta.recent_draws is removed, keeping only top-level recent_draws
    pool_data = {
        'meta': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_issue': target_issue,
            'target_draw_date': f"{target_date.strftime('%Y-%m-%d')} ({weekday_cn})",
            'latest_draw': {
                'issue': str(latest_issue),
                'date': latest_row['draw_date'],
                'reds': feat_module.get_red_balls_from_row(latest_row),
                'blue': int(latest_row['blue']),
            },
            'ev_timing': None,
            'consensus': None,
        },
        'recent_draws': recent_draws,
        'pools': {
            'balanced': balanced_pool,
            'jackpot': jackpot_pool,
        }
    }

    # Save to docs/data/pool.json, reports/latest_pool.json, and archive pool_{target_issue}.json
    archive_path = os.path.join(config.REPORTS_DIR, f'pool_{target_issue}.json')
    save_paths = [output_path, backup_path, archive_path]
    for p in set(save_paths):
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)
            logger.info(f'Exported candidate pool to {p}')

    return pool_data

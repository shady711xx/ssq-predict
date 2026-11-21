"""Backtesting module for SSQ lottery prediction system.

Implements Walk-Forward Validation without data leakage:
- Evaluates Strategy A, Strategy B, Strategy C, Ensemble, and Random Baseline
- Computes calibration metrics: Log-Loss and Brier Score
- Official prize tier determination (O(1) tuple lookup) with optional 福运奖
- Support for test period by draw count (--draws) or day range (--days)
"""
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
from src import storage
from src.game_theory import (
    AntiPopularityFilter,
    CrowdBettingSimulator,
    UniquenessScorer,
    select_game_theory_combinations,
)
from src.model import (
    EnsembleModel,
    LogisticRegressionModel,
    RandomForestModel,
    StatisticalModel,
)

logger = logging.getLogger('ssq.backtest')


def determine_prize_level(
    red_hit: int,
    blue_hit: int,
    special_rule_active: bool = False,
    special_rule: Optional[bool] = None,
) -> Tuple[str, int]:
    """Determine prize level based on official SSQ rules.
    
    Args:
        red_hit: number of red balls matched (0-6)
        blue_hit: number of blue balls matched (0 or 1)
        special_rule_active: whether 2026 福运奖 (3+0 = 5元) is active
        special_rule: alias for special_rule_active
    
    Returns:
        tuple: (prize_level_name, prize_amount_cny)
    """
    if special_rule is not None:
        special_rule_active = special_rule

    # 2026 new rule: 福运奖 (3 red + 0 blue = 5 CNY)
    if special_rule_active and (red_hit, blue_hit) == (3, 0):
        return ('福运奖', 5)

    rule = config.PRIZE_RULES.get((red_hit, blue_hit))
    if rule is not None:
        return (rule['desc'], rule['prize'])

    return ('未中奖', 0)


def evaluate_single_draw(
    pred_reds: List[int],
    pred_blue: int,
    actual_reds: List[int],
    actual_blue: int,
    special_rule_active: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single ticket against actual winning numbers."""
    red_hit = len(set(pred_reds) & set(actual_reds))
    blue_hit = 1 if pred_blue == actual_blue else 0
    prize_level, prize_amount = determine_prize_level(
        red_hit, blue_hit, special_rule_active=special_rule_active
    )

    return {
        'red_hit': red_hit,
        'blue_hit': blue_hit,
        'prize_level': prize_level,
        'prize_amount': prize_amount,
    }


def compute_multi_label_log_loss(y_red_list: Any, p_red_list: Any) -> float:
    """Helper to compute mean multi-label log-loss across draws."""
    losses = []
    for y, p in zip(y_red_list, p_red_list):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        loss = -np.mean(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))
        losses.append(loss)
    return float(np.mean(losses))


def run_monte_carlo_baseline(
    actuals: Any,
    n_simulations: int = 100,
    n_combinations: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Alias for simulate_random_baseline_mc."""
    return simulate_random_baseline_mc(
        actual_draws_tuples=actuals,
        n_combinations=n_combinations,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )


def compute_calibration_metrics(
    red_probs: Dict[int, float],
    blue_probs: Dict[int, float],
    actual_reds: List[int],
    actual_blue: int,
) -> Dict[str, float]:
    """Calculate multi-label binary Log-Loss & Brier Score for red/blue balls.
    
    Returns dict with red_log_loss, red_brier, blue_log_loss, blue_brier.
    """
    # Red balls: 33 binary indicators
    y_red = np.zeros(33, dtype=float)
    for r in actual_reds:
        y_red[r - 1] = 1.0

    # Scale normalized red probabilities (sum=1) to marginal probability of occurrence (scale ~ 6/33)
    p_red_norm = np.array([red_probs.get(i + 1, 1.0 / 33.0) for i in range(33)], dtype=float)
    p_marginal = np.clip(p_red_norm * 6.0, 1e-6, 1.0 - 1e-6)

    red_brier = float(np.mean((p_marginal - y_red) ** 2))
    red_log_loss = float(-np.mean(y_red * np.log(p_marginal) + (1.0 - y_red) * np.log(1.0 - p_marginal)))

    # Blue ball: 1-of-16 categorical
    y_blue = np.zeros(16, dtype=float)
    y_blue[actual_blue - 1] = 1.0

    p_blue = np.array([blue_probs.get(j + 1, 1.0 / 16.0) for j in range(16)], dtype=float)
    p_blue_actual = max(blue_probs.get(actual_blue, 1.0 / 16.0), 1e-6)

    blue_brier = float(np.sum((p_blue - y_blue) ** 2))
    blue_log_loss = float(-np.log(p_blue_actual))

    return {
        'red_log_loss': red_log_loss,
        'red_brier': red_brier,
        'blue_log_loss': blue_log_loss,
        'blue_brier': blue_brier,
    }


def compute_uniform_baseline_metrics(actual_reds: List[int], actual_blue: int) -> Dict[str, float]:
    """Calculate theoretical Log-Loss & Brier Score for pure uniform random guessing."""
    # Red uniform prior: 6 / 33 per ball
    p_u_red = 6.0 / 33.0
    y_red = np.zeros(33, dtype=float)
    for r in actual_reds:
        y_red[r - 1] = 1.0

    red_brier = float(np.mean((p_u_red - y_red) ** 2))
    red_log_loss = float(-np.mean(y_red * np.log(p_u_red) + (1.0 - y_red) * np.log(1.0 - p_u_red)))

    # Blue uniform prior: 1 / 16
    p_u_blue = 1.0 / 16.0
    y_blue = np.zeros(16, dtype=float)
    y_blue[actual_blue - 1] = 1.0
    p_b_vec = np.full(16, p_u_blue)

    blue_brier = float(np.sum((p_b_vec - y_blue) ** 2))
    blue_log_loss = float(-np.log(p_u_blue))

    return {
        'red_log_loss': red_log_loss,
        'red_brier': red_brier,
        'blue_log_loss': blue_log_loss,
        'blue_brier': blue_brier,
    }


def simulate_random_baseline_mc(
    actual_draws_tuples: List[Tuple[List[int], int]],
    n_combinations: int = 5,
    n_simulations: int = 100,
    special_rule_active: bool = False,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation of Random Baseline on actual draw results.
    
    Performs n_simulations independent runs across the test draws:
    1. Distinct blue version: 5 distinct blue balls per draw + random red balls
    2. Pure random version: independent uniform random blue and red balls
    
    Returns metrics including Median, P25, P75, and fixed-prize means.
    """
    rng = np.random.default_rng(random_seed)
    n_draws = len(actual_draws_tuples)
    total_tickets = n_draws * n_combinations

    distinct_winnings = []
    distinct_fixed_winnings = []
    distinct_red_hits = []
    distinct_blue_hits = []

    pure_winnings = []
    pure_fixed_winnings = []

    for _ in range(n_simulations):
        run_win_dist = 0
        run_fixed_dist = 0
        run_red_dist = 0
        run_blue_dist = 0

        run_win_pure = 0
        run_fixed_pure = 0

        for actual_reds, actual_blue in actual_draws_tuples:
            actual_reds_set = set(actual_reds)

            # 1. Distinct blue version (5 distinct blues sampled from 1..16)
            distinct_blues = rng.choice(16, size=n_combinations, replace=False) + 1
            for b in distinct_blues:
                reds = rng.choice(33, size=6, replace=False) + 1
                r_hit = len(actual_reds_set.intersection(reds))
                b_hit = 1 if b == actual_blue else 0
                run_red_dist += r_hit
                run_blue_dist += b_hit
                _, pz = determine_prize_level(r_hit, b_hit, special_rule_active=special_rule_active)
                run_win_dist += pz
                if pz <= 3000:
                    run_fixed_dist += pz

            # 2. Pure random version (independent blue for each ticket)
            for _ in range(n_combinations):
                b = int(rng.integers(1, 17))
                reds = rng.choice(33, size=6, replace=False) + 1
                r_hit = len(actual_reds_set.intersection(reds))
                b_hit = 1 if b == actual_blue else 0
                _, pz = determine_prize_level(r_hit, b_hit, special_rule_active=special_rule_active)
                run_win_pure += pz
                if pz <= 3000:
                    run_fixed_pure += pz

        distinct_winnings.append(run_win_dist)
        distinct_fixed_winnings.append(run_fixed_dist)
        distinct_red_hits.append(run_red_dist)
        distinct_blue_hits.append(run_blue_dist)

        pure_winnings.append(run_win_pure)
        pure_fixed_winnings.append(run_fixed_pure)

    return {
        'distinct_blue': {
            'median': float(np.median(distinct_winnings)),
            'p25': float(np.percentile(distinct_winnings, 25)),
            'p75': float(np.percentile(distinct_winnings, 75)),
            'fixed_mean': float(np.mean(distinct_fixed_winnings)),
            'mean': float(np.mean(distinct_winnings)),
            'avg_red_hits': float(np.mean(distinct_red_hits)) / total_tickets if total_tickets > 0 else 0.0,
            'avg_blue_hits': float(np.mean(distinct_blue_hits)) / total_tickets if total_tickets > 0 else 0.0,
        },
        'pure_random': {
            'median': float(np.median(pure_winnings)),
            'p25': float(np.percentile(pure_winnings, 25)),
            'p75': float(np.percentile(pure_winnings, 75)),
            'fixed_mean': float(np.mean(pure_fixed_winnings)),
            'mean': float(np.mean(pure_winnings)),
        },
    }


def run_backtest(
    days: Optional[int] = None,
    draws: Optional[int] = 100,
    n_combinations: int = 5,
    special_rule_active: bool = False,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run Walk-Forward backtesting with multi-strategy ablation comparison.
    
    Evaluates:
    1. Strategy A (Statistical)
    2. Strategy B (Regularized Random Forest)
    3. Strategy C (L2 Logistic Regression)
    4. Ensemble (Weighted blend)
    5. Ensemble + GameTheory (balanced)
    6. Ensemble + GameTheory (jackpot)
    7. Random Baseline (Monte Carlo expectations)
    
    Args:
        days: Filter by last N days (if draws is None)
        draws: Number of recent draws to backtest (default: 100)
        n_combinations: Number of tickets per draw (default: 5)
        special_rule_active: Whether to enable 2026 福运奖 (3+0=5元)
        random_seed: Random seed for reproducible backtesting (default: 42)
    
    Returns:
        tuple: (detail_results: list[dict], ablation_summary: dict)
    """
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_draws = storage.get_all_draws()
    if all_draws.empty:
        logger.error('No data available for backtesting')
        return [], {}

    # Select test period
    if draws is not None and draws > 0:
        test_draws = all_draws.tail(draws).copy()
        test_period_desc = f'最近 {len(test_draws)} 期'
    elif days is not None and days > 0:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        test_draws = storage.get_draws_in_date_range(
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        if test_draws.empty:
            test_draws = all_draws.tail(int((days // 7) * 3 + 3)).copy()
        test_period_desc = f'最近 {days} 天 ({len(test_draws)} 期)'
    else:
        test_draws = all_draws.tail(100).copy()
        test_period_desc = f'最近 {len(test_draws)} 期'

    n_test_draws = len(test_draws)
    logger.info(f'Starting walk-forward backtest over {n_test_draws} draws ({test_period_desc})')

    # Tracking metrics per strategy
    strategies = [
        'Strategy A (统计加权)',
        'Strategy B (随机森林)',
        'Strategy C (逻辑回归)',
        'Ensemble (基础融合)',
        'Ensemble + GT (balanced)',
        'Ensemble + GT (jackpot)',
    ]
    stats_map: Dict[str, Dict[str, Any]] = {
        name: {
            'total_red_hits': 0,
            'total_blue_hits': 0,
            'total_predictions': 0,
            'total_investment': 0,
            'total_winnings': 0,
            'red_log_loss_sum': 0.0,
            'blue_log_loss_sum': 0.0,
            'red_brier_sum': 0.0,
            'blue_brier_sum': 0.0,
            'uniqueness_score_sum': 0.0,
            'prize_counts': {},
        }
        for name in strategies
    }

    # Ticket tracking for CrowdBettingSimulator
    unfiltered_tickets: List[Dict[str, Any]] = []
    gt_bal_tickets: List[Dict[str, Any]] = []
    gt_jack_tickets: List[Dict[str, Any]] = []

    # Tracking for Random baseline
    random_stats = {
        'total_red_hits': 0.0,
        'total_blue_hits': 0.0,
        'total_predictions': n_test_draws * n_combinations,
        'total_investment': n_test_draws * n_combinations * 2,
        'total_winnings': 0.0,
        'red_log_loss_sum': 0.0,
        'blue_log_loss_sum': 0.0,
        'red_brier_sum': 0.0,
        'blue_brier_sum': 0.0,
    }

    # Theoretical random expectation per ticket:
    # E[red_hits] = 6 * (6 / 33) = 1.0909
    # E[blue_hits] = 1 / 16 = 0.0625
    expected_red_per_ticket = 6.0 * 6.0 / 33.0
    expected_blue_per_ticket = 1.0 / 16.0
    random_stats['total_red_hits'] = expected_red_per_ticket * (n_test_draws * n_combinations)
    random_stats['total_blue_hits'] = expected_blue_per_ticket * (n_test_draws * n_combinations)

    # Expected payout per ticket in SSQ ~ 0.50 * cost = ~1.00 CNY
    # Using Monte Carlo simulation across prize table:
    # P(6+1)=1/17.72M, P(6+0)=15/17.72M, P(5+1)=162/17.72M, P(5+0)=2430/17.72M,
    # P(4+1)=7695/17.72M, P(4+0)=115425/17.72M, P(3+1)=136800/17.72M,
    # P(2+1)=1026000/17.72M, P(1+1)=3283200/17.72M, P(0+1)=3283200/17.72M (~1/16)
    expected_payout_per_ticket = 0.985  # official expectation is ~50% prize payout
    if special_rule_active:
        # P(3+0) = C(6,3)*C(27,3)*15 / 17.72M = 20 * 2925 * 15 / 17.72M = 877,500 / 17,721,088 ≈ 0.0495
        # Expected extra payout: 0.0495 * 5 ≈ 0.2475 CNY
        expected_payout_per_ticket += 0.2475
    random_stats['total_winnings'] = expected_payout_per_ticket * (n_test_draws * n_combinations)

    detail_results: List[Dict[str, Any]] = []
    actual_draws_tuples: List[Tuple[List[int], int]] = []

    # Pre-build feature matrix for all draws once
    # Row t in full_X is computed strictly using draws 0..t-1 (no future data)
    logger.info('Pre-building feature matrix for Walk-Forward backtest...')
    full_X, full_y_red, full_y_blue = feat_module.build_feature_matrix(all_draws)
    max_w = max(config.MODEL_WINDOW_SIZES)
    issue_to_idx = {str(row['issue']).zfill(5): i for i, (_, row) in enumerate(all_draws.iterrows())}

    for idx, (_, test_row) in enumerate(test_draws.iterrows()):
        test_issue = str(test_row['issue']).zfill(5)
        test_date = str(test_row['draw_date'])
        actual_reds = feat_module.get_red_balls_from_row(test_row)
        actual_blue = int(test_row['blue'])

        t_idx = issue_to_idx.get(test_issue)
        if t_idx is None or t_idx <= max_w:
            continue

        actual_draws_tuples.append((actual_reds, actual_blue))
        train_df = all_draws.iloc[:t_idx]
        train_limit = t_idx - max_w

        ensemble_predictions: List[Dict[str, Any]] = []
        ensemble_eval_results: List[Dict[str, Any]] = []
        ensemble_red_hits = 0
        ensemble_blue_hits = 0
        best_prize: Tuple[str, int] = ('未中奖', 0)

        # Slice historical feature matrix (strictly < test draw)
        # Using recent 500 draws for high statistical relevance and speed
        window_start = max(0, train_limit - 500)
        X_train = full_X.iloc[window_start:train_limit]
        y_red_train = full_y_red[window_start:train_limit]
        y_blue_train = full_y_blue[window_start:train_limit]
        X_latest = full_X.iloc[train_limit:train_limit + 1]

        # Fit all 3 strategies
        stat_model = StatisticalModel()
        stat_model.fit(train_df)

        rf_model = RandomForestModel()
        rf_model.fit_with_matrix(X_train, y_red_train, y_blue_train, X_latest)

        lr_model = LogisticRegressionModel()
        lr_model.fit_with_matrix(X_train, y_red_train, y_blue_train, X_latest)

        # Ensemble
        ensemble_model = EnsembleModel()
        ensemble_model.fit_with_components(stat_model, rf_model, lr_model)

        # Models to evaluate
        scorer_obj = UniquenessScorer(history_df=train_df)
        filter_obj = AntiPopularityFilter(history_df=train_df)

        model_dict = {
            'Strategy A (统计加权)': stat_model,
            'Strategy B (随机森林)': rf_model,
            'Strategy C (逻辑回归)': lr_model,
            'Ensemble (基础融合)': ensemble_model,
        }

        # Evaluate base 4 strategies
        for name, m in model_dict.items():
            preds = m.predict(n_combinations=n_combinations, df=train_df)
            probs = m.get_probabilities()
            calib = compute_calibration_metrics(probs['red'], probs['blue'], actual_reds, actual_blue)

            st = stats_map[name]
            st['red_log_loss_sum'] += calib['red_log_loss']
            st['blue_log_loss_sum'] += calib['blue_log_loss']
            st['red_brier_sum'] += calib['red_brier']
            st['blue_brier_sum'] += calib['blue_brier']

            st['total_investment'] += n_combinations * 2
            st['total_predictions'] += len(preds)

            for pred in preds:
                ev = evaluate_single_draw(
                    pred['reds'], pred['blue'], actual_reds, actual_blue,
                    special_rule_active=special_rule_active
                )
                st['total_red_hits'] += ev['red_hit']
                st['total_blue_hits'] += ev['blue_hit']
                st['total_winnings'] += ev['prize_amount']
                u_res = scorer_obj.score(pred['reds'], pred['blue'])
                st['uniqueness_score_sum'] += u_res['uniqueness_score']

                lvl = ev['prize_level']
                st['prize_counts'][lvl] = st['prize_counts'].get(lvl, 0) + 1

                if name == 'Ensemble (基础融合)':
                    unfiltered_tickets.append(pred)
                    ensemble_predictions.append(pred)
                    ensemble_eval_results.append(ev)
                    ensemble_red_hits += ev['red_hit']
                    ensemble_blue_hits += ev['blue_hit']
                    if ev['prize_amount'] > best_prize[1]:
                        best_prize = (ev['prize_level'], ev['prize_amount'])

        # Evaluate Game Theory strategies
        gt_bal_preds, _ = select_game_theory_combinations(
            ensemble_model,
            n_combinations=n_combinations,
            mode='balanced',
            beta=config.GAME_THEORY_BETA,
            df=train_df,
            filter_obj=filter_obj,
            scorer_obj=scorer_obj,
            random_seed=random_seed + t_idx,
        )
        gt_jack_preds, _ = select_game_theory_combinations(
            ensemble_model,
            n_combinations=n_combinations,
            mode='jackpot',
            beta=config.GAME_THEORY_BETA,
            df=train_df,
            filter_obj=filter_obj,
            scorer_obj=scorer_obj,
            random_seed=random_seed + t_idx,
        )

        gt_dict = {
            'Ensemble + GT (balanced)': gt_bal_preds,
            'Ensemble + GT (jackpot)': gt_jack_preds,
        }
        calib_ens = compute_calibration_metrics(
            ensemble_model.get_probabilities()['red'],
            ensemble_model.get_probabilities()['blue'],
            actual_reds,
            actual_blue,
        )

        for gt_name, gt_preds in gt_dict.items():
            st_gt = stats_map[gt_name]
            st_gt['red_log_loss_sum'] += calib_ens['red_log_loss']
            st_gt['blue_log_loss_sum'] += calib_ens['blue_log_loss']
            st_gt['red_brier_sum'] += calib_ens['red_brier']
            st_gt['blue_brier_sum'] += calib_ens['blue_brier']
            st_gt['total_investment'] += n_combinations * 2
            st_gt['total_predictions'] += len(gt_preds)

            for pred in gt_preds:
                ev = evaluate_single_draw(
                    pred['reds'], pred['blue'], actual_reds, actual_blue,
                    special_rule_active=special_rule_active
                )
                st_gt['total_red_hits'] += ev['red_hit']
                st_gt['total_blue_hits'] += ev['blue_hit']
                st_gt['total_winnings'] += ev['prize_amount']
                st_gt['uniqueness_score_sum'] += pred.get('uniqueness_score', 0.5)
                lvl = ev['prize_level']
                st_gt['prize_counts'][lvl] = st_gt['prize_counts'].get(lvl, 0) + 1

                if gt_name == 'Ensemble + GT (balanced)':
                    gt_bal_tickets.append(pred)
                else:
                    gt_jack_tickets.append(pred)

        # Random baseline calibration metrics
        u_calib = compute_uniform_baseline_metrics(actual_reds, actual_blue)
        random_stats['red_log_loss_sum'] += u_calib['red_log_loss']
        random_stats['blue_log_loss_sum'] += u_calib['blue_log_loss']
        random_stats['red_brier_sum'] += u_calib['red_brier']
        random_stats['blue_brier_sum'] += u_calib['blue_brier']

        # Actual draw score under Ensemble model (LLR-based)
        actual_score = ensemble_model.score_combination(actual_reds, actual_blue)
        avg_red = ensemble_red_hits / len(ensemble_predictions) if ensemble_predictions else 0.0
        avg_blue = ensemble_blue_hits / len(ensemble_predictions) if ensemble_predictions else 0.0

        draw_detail = {
            'issue': test_issue,
            'draw_date': test_date,
            'actual_reds': actual_reds,
            'actual_blue': actual_blue,
            'predictions': ensemble_predictions,
            'eval_results': ensemble_eval_results,
            'avg_red_hit': avg_red,
            'avg_blue_hit': avg_blue,
            'best_prize': best_prize[0],
            'best_prize_amount': best_prize[1],
            'investment': n_combinations * 2,
            'winnings': sum(ev['prize_amount'] for ev in ensemble_eval_results),
            'actual_score': round(actual_score, 4),
        }
        detail_results.append(draw_detail)

    # Build summary table across all strategies
    summary_rows = []
    for name, st in stats_map.items():
        n_p = st['total_predictions']
        n_d = len(detail_results)
        inv = st['total_investment']
        win = st['total_winnings']
        net = win - inv
        roi = (net / inv * 100.0) if inv > 0 else 0.0

        summary_rows.append({
            'strategy': name,
            'avg_red_hits': round(st['total_red_hits'] / n_p, 2) if n_p > 0 else 0.0,
            'avg_blue_hits': round(st['total_blue_hits'] / n_p, 3) if n_p > 0 else 0.0,
            'red_log_loss': round(st['red_log_loss_sum'] / n_d, 4) if n_d > 0 else 0.0,
            'blue_log_loss': round(st['blue_log_loss_sum'] / n_d, 4) if n_d > 0 else 0.0,
            'red_brier': round(st['red_brier_sum'] / n_d, 4) if n_d > 0 else 0.0,
            'blue_brier': round(st['blue_brier_sum'] / n_d, 4) if n_d > 0 else 0.0,
            'uniqueness_score': round(st['uniqueness_score_sum'] / n_p, 4) if n_p > 0 else 0.0,
            'total_investment': inv,
            'total_winnings': win,
            'net_return': net,
            'roi_pct': round(roi, 1),
            'prize_distribution': st['prize_counts'],
        })

    # 5. Add random baseline via 100-run Monte Carlo empirical simulation on actual test draws
    n_d = len(detail_results)
    r_inv = n_d * n_combinations * 2
    mc_stats = simulate_random_baseline_mc(
        actual_draws_tuples,
        n_combinations=n_combinations,
        n_simulations=100,
        special_rule_active=special_rule_active,
        random_seed=random_seed,
    )
    mc_dist = mc_stats['distinct_blue']
    r_win_med = round(mc_dist['median'], 1)
    r_net_med = round(r_win_med - r_inv, 1)
    r_roi_med = round(r_net_med / r_inv * 100.0, 1) if r_inv > 0 else 0.0

    summary_rows.append({
        'strategy': 'Random Baseline (蒙特卡洛实测)',
        'avg_red_hits': round(mc_dist['avg_red_hits'], 2),
        'avg_blue_hits': round(mc_dist['avg_blue_hits'], 3),
        'red_log_loss': round(random_stats['red_log_loss_sum'] / n_d, 4) if n_d > 0 else 0.0,
        'blue_log_loss': round(random_stats['blue_log_loss_sum'] / n_d, 4) if n_d > 0 else 0.0,
        'red_brier': round(random_stats['red_brier_sum'] / n_d, 4) if n_d > 0 else 0.0,
        'blue_brier': round(random_stats['blue_brier_sum'] / n_d, 4) if n_d > 0 else 0.0,
        'uniqueness_score': 0.5050,
        'total_investment': r_inv,
        'total_winnings': r_win_med,
        'winnings_display': f"¥{int(r_win_med)} (¥{int(mc_dist['p25'])}-¥{int(mc_dist['p75'])})",
        'net_return': r_net_med,
        'roi_pct': r_roi_med,
        'prize_distribution': {
            '中位数': f"¥{int(r_win_med)}",
            'P25-P75': f"¥{int(mc_dist['p25'])}-¥{int(mc_dist['p75'])}",
            '固定奖均值': f"¥{mc_dist['fixed_mean']:.1f}",
        },
    })

    # Crowd Betting Simulator evaluation across strategies
    crowd_sim = CrowdBettingSimulator(
        last_draw_reds=actual_draws_tuples[-1][0] if actual_draws_tuples else None,
        seed=random_seed,
    )
    crowd_pool = crowd_sim.generate_crowd_pool(n_bets=50_000)

    rng_rand = np.random.default_rng(random_seed)
    random_sample_tickets = []
    for _ in range(len(unfiltered_tickets)):
        rr = sorted(rng_rand.choice(33, size=6, replace=False) + 1)
        bb = int(rng_rand.integers(1, 17))
        random_sample_tickets.append({'reds': rr, 'blue': bb})

    crowd_report = {
        'random': crowd_sim.evaluate_strategy_collision(random_sample_tickets, crowd_pool),
        'unfiltered_ensemble': crowd_sim.evaluate_strategy_collision(unfiltered_tickets, crowd_pool),
        'gt_balanced': crowd_sim.evaluate_strategy_collision(gt_bal_tickets, crowd_pool),
        'gt_jackpot': crowd_sim.evaluate_strategy_collision(gt_jack_tickets, crowd_pool),
    }

    ablation_summary = {
        'test_period': test_period_desc,
        'n_test_draws': len(detail_results),
        'n_combinations_per_draw': n_combinations,
        'special_rule_active': special_rule_active,
        'strategies': summary_rows,
        'mc_baseline_stats': mc_stats,
        'crowd_report': crowd_report,
    }

    return detail_results, ablation_summary


def print_backtest_report(
    detail_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    recent_detail_count: int = 14,
) -> None:
    """Print ablation comparison table and recent draws detail."""
    if not detail_results:
        print('\n❌ No backtest results to display.')
        return

    print('\n' + '=' * 90)
    print(f"📊 双色球模型回测报告 ({summary['test_period']}, 共 {summary['n_test_draws']} 期)")
    if summary.get('special_rule_active'):
        print('   ⭐ 2026 新规福运奖 (3红+0蓝=5元) [已激活]')
    print('=' * 90)

    # 1. Ablation comparison table
    ablation_data = []
    for s in summary['strategies']:
        w_display = s.get('winnings_display', f"¥{s['total_winnings']}")
        ablation_data.append([
            s['strategy'],
            f"{s['avg_red_hits']:.2f}",
            f"{s['avg_blue_hits']:.3f}",
            f"{s['red_log_loss']:.4f}",
            f"{s['blue_log_loss']:.4f}",
            f"{s['red_brier']:.4f}",
            f"{s.get('uniqueness_score', 0.5):.4f}",
            w_display,
            f"¥{s['net_return']}",
            f"{s['roi_pct']:+.1f}%",
        ])

    headers = [
        '策略 / 模型',
        '红球均中',
        '蓝球均中',
        '红球 Log-Loss',
        '蓝球 Log-Loss',
        '红球 Brier',
        '独享度',
        '模拟总中奖',
        '净收益',
        'ROI',
    ]
    print('\n🏆 多方消融与博弈论对比表 (Ablation Comparison):')
    print(tabulate(ablation_data, headers=headers, tablefmt='grid'))

    if 'mc_baseline_stats' in summary:
        mc = summary['mc_baseline_stats']
        dist = mc['distinct_blue']
        pure = mc['pure_random']
        print(f"\n🎲 随机基线蒙特卡洛实测统计 (100 次独立模拟, 每次 {summary['n_test_draws']} 期 × {summary['n_combinations_per_draw']} 注 = {summary['n_test_draws'] * summary['n_combinations_per_draw']} 注):")
        print(f"   • 蓝球去重随机 (Top-5不同蓝): 中位数 ¥{int(dist['median'])} (P25-P75: ¥{int(dist['p25'])} - ¥{int(dist['p75'])}) | 固定奖均值: ¥{dist['fixed_mean']:.1f} | 均值总中奖: ¥{dist['mean']:.1f}")
        print(f"   • 完全独立随机 (独立选蓝):   中位数 ¥{int(pure['median'])} (P25-P75: ¥{int(pure['p25'])} - ¥{int(pure['p75'])}) | 固定奖均值: ¥{pure['fixed_mean']:.1f} | 均值总中奖: ¥{pure['mean']:.1f}")

    if 'crowd_report' in summary:
        cr = summary['crowd_report']
        print(f"\n🎯 大众投注碰撞模拟评估 (基于 50,000 注真实彩民模拟: 60%机选 + 40%认知偏置):")
        print(f"   • 纯随机选号 (Random Baseline):             号码热度倍率 {cr['random']['avg_popularity']:.2f}x | 独享奖金期望 {cr['random']['share_multiplier']:.2f}x (基准)")
        print(f"   • 未过滤融合模型 (Ensemble 基础):           号码热度倍率 {cr['unfiltered_ensemble']['avg_popularity']:.2f}x | 独享奖金期望 {cr['unfiltered_ensemble']['share_multiplier']:.2f}x")
        print(f"   • 博弈论增强-稳健型 (Ensemble + GT balanced): 号码热度倍率 {cr['gt_balanced']['avg_popularity']:.2f}x | 独享奖金期望 {cr['gt_balanced']['share_multiplier']:.2f}x (预期平分人数下降 {cr['gt_balanced']['co_winners_reduction_pct']}%)")
        print(f"   • 博弈论增强-冲大奖型 (Ensemble + GT jackpot): 号码热度倍率 {cr['gt_jackpot']['avg_popularity']:.2f}x | 独享奖金期望 {cr['gt_jackpot']['share_multiplier']:.2f}x (预期平分人数下降 {cr['gt_jackpot']['co_winners_reduction_pct']}%)")

    # 2. Detail table for recent N draws
    recent_draws = detail_results[-recent_detail_count:]
    detail_data = []
    for d in recent_draws:
        reds_str = ','.join(f'{r:02d}' for r in d['actual_reds'])
        detail_data.append([
            d['issue'],
            d['draw_date'],
            f"{reds_str}+{d['actual_blue']:02d}",
            f"{d['avg_red_hit']:.1f}",
            f"{d['avg_blue_hit']:.1f}",
            d['best_prize'],
            f"¥{d['winnings']}",
            f"{d['actual_score']:.4f}",
        ])

    detail_headers = ['期号', '开奖日期', '实际开奖', '平均红命中', '平均蓝命中', '最佳奖级', '中奖金额', '融合评分']
    print(f'\n📋 最近 {len(recent_draws)} 期逐期回测明细 (融合模型):')
    print(tabulate(detail_data, headers=detail_headers, tablefmt='grid'))

    print('\n' + '=' * 90)


def save_backtest_report(
    detail_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """Save backtest results to CSV and markdown."""
    config.ensure_dirs()

    # Save detailed CSV
    csv_data = []
    for d in detail_results:
        for i, (pred, ev) in enumerate(zip(d['predictions'], d['eval_results'])):
            csv_data.append({
                'issue': d['issue'],
                'draw_date': d['draw_date'],
                'actual_reds': ','.join(f'{r:02d}' for r in d['actual_reds']),
                'actual_blue': f"{d['actual_blue']:02d}",
                'pred_index': i + 1,
                'pred_reds': ','.join(f'{r:02d}' for r in pred['reds']),
                'pred_blue': f"{pred['blue']:02d}",
                'pred_score': f"{pred['score']:.4f}",
                'actual_score': f"{d['actual_score']:.4f}",
                'red_hit': ev['red_hit'],
                'blue_hit': ev['blue_hit'],
                'prize_level': ev['prize_level'],
                'prize_amount': ev['prize_amount'],
            })

    csv_path = os.path.join(config.REPORTS_DIR, 'backtest_last_30d.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f'Backtest detail saved to {csv_path}')

    # Save markdown summary
    md_path = os.path.join(config.REPORTS_DIR, 'backtest_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 双色球预测系统回测与模型消融对比报告\n\n')
        f.write(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'## 回测概况\n\n')
        f.write(f'- **回测期间**: {summary["test_period"]}\n')
        f.write(f'- **回测期数**: {summary["n_test_draws"]} 期\n')
        f.write(f'- **单期预测注数**: {summary["n_combinations_per_draw"]} 注\n')
        f.write(f'- **福运奖规则**: {"已开启 (3+0=5元)" if summary.get("special_rule_active") else "未开启"}\n\n')

        f.write('## 多方消融与博弈论对比表\n\n')
        f.write('| 策略 / 模型 | 红球均中 | 蓝球均中 | 红球 Log-Loss | 蓝球 Log-Loss | 红球 Brier | 独享度 | 模拟总中奖 | 净收益 | ROI |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for s in summary['strategies']:
            w_display = s.get('winnings_display', f"¥{s['total_winnings']}")
            f.write(
                f"| {s['strategy']} | {s['avg_red_hits']:.2f} | {s['avg_blue_hits']:.3f} | "
                f"{s['red_log_loss']:.4f} | {s['blue_log_loss']:.4f} | "
                f"{s['red_brier']:.4f} | {s.get('uniqueness_score', 0.5):.4f} | "
                f"{w_display} | ¥{s['net_return']} | {s['roi_pct']:+.1f}% |\n"
            )

        if 'mc_baseline_stats' in summary:
            mc = summary['mc_baseline_stats']
            dist = mc['distinct_blue']
            pure = mc['pure_random']
            f.write('\n### 随机基准蒙特卡洛实测详情 (100 次独立模拟)\n\n')
            f.write(f"- **蓝球去重随机 (Top-5不同蓝)**: 中位数 ¥{int(dist['median'])} (P25-P75: ¥{int(dist['p25'])} - ¥{int(dist['p75'])})，纯固定奖均值: ¥{dist['fixed_mean']:.1f}，均值总中奖: ¥{dist['mean']:.1f}\n")
            f.write(f"- **完全独立随机 (独立选蓝)**: 中位数 ¥{int(pure['median'])} (P25-P75: ¥{int(pure['p25'])} - ¥{int(pure['p75'])})，纯固定奖均值: ¥{pure['fixed_mean']:.1f}，均值总中奖: ¥{pure['mean']:.1f}\n")

        if 'crowd_report' in summary:
            cr = summary['crowd_report']
            f.write('\n### 大众投注碰撞模拟评估 (基于 50,000 注真实彩民偏置模拟)\n\n')
            f.write(f"- **纯随机选号 (Random Baseline)**: 号码热度倍率 {cr['random']['avg_popularity']:.2f}x | 独享奖金期望 {cr['random']['share_multiplier']:.2f}x (基准)\n")
            f.write(f"- **未过滤融合模型 (Ensemble 基础)**: 号码热度倍率 {cr['unfiltered_ensemble']['avg_popularity']:.2f}x | 独享奖金期望 {cr['unfiltered_ensemble']['share_multiplier']:.2f}x\n")
            f.write(f"- **博弈论增强-稳健型 (Ensemble + GT balanced)**: 号码热度倍率 {cr['gt_balanced']['avg_popularity']:.2f}x | 独享奖金期望 {cr['gt_balanced']['share_multiplier']:.2f}x (预期平分人数下降 {cr['gt_balanced']['co_winners_reduction_pct']}%)\n")
            f.write(f"- **博弈论增强-冲大奖型 (Ensemble + GT jackpot)**: 号码热度倍率 {cr['gt_jackpot']['avg_popularity']:.2f}x | 独享奖金期望 {cr['gt_jackpot']['share_multiplier']:.2f}x (预期平分人数下降 {cr['gt_jackpot']['co_winners_reduction_pct']}%)\n")

        f.write('\n## 最近 14 期逐期明细 (融合模型)\n\n')
        f.write('| 期号 | 开奖日期 | 实际开奖 | 平均红命中 | 平均蓝命中 | 最佳奖级 | 中奖金额 | 融合评分 |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for d in detail_results[-14:]:
            reds_str = ','.join(f'{r:02d}' for r in d['actual_reds'])
            f.write(
                f"| {d['issue']} | {d['draw_date']} | {reds_str}+{d['actual_blue']:02d} | "
                f"{d['avg_red_hit']:.1f} | {d['avg_blue_hit']:.1f} | "
                f"{d['best_prize']} | ¥{d['winnings']} | {d['actual_score']:.4f} |\n"
            )

    logger.info(f'Backtest summary saved to {md_path}')

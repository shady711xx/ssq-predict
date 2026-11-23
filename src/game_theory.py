"""Game Theory and Anti-Popularity Module for SSQ Lottery.

Implements anti-popular betting strategies to maximize expected solo jackpot payout:
1. AntiPopularityFilter: Hard-filtering combinations with extreme crowd collisions
2. UniquenessScorer: Soft-scoring uniqueness [0.0, 1.0] across birthday, pattern, cultural, and history dimensions
3. CrowdBettingSimulator: Simulates 50,000 biased crowd bets to quantify collision reduction
4. select_game_theory_combinations: Re-ranks model candidates and supports 'balanced' and 'jackpot' modes
"""
import itertools
import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import config
from src import features as feat_module
from src import storage

logger = logging.getLogger('ssq.game_theory')

# Mirror number pairs frequently chosen by graphic pattern pickers
MIRROR_PAIRS = {(1, 10), (2, 20), (3, 30), (12, 21), (13, 31), (23, 32)}

# Cultural avoid/prefer sets
AVOIDED_REDS = {4, 14, 24}
AVOIDED_BLUES = {4, 13, 14, 15}
POPULAR_BLUES = {6, 7, 8, 9, 16}
LUCKY_TAILS = {6, 8, 9}


class AntiPopularityFilter:
    """Hard-filters out combinations prone to extreme collision with common player heuristics."""

    def __init__(self, history_df: Optional[pd.DataFrame] = None) -> None:
        self.history_red_tuples: Set[Tuple[int, ...]] = set()
        self.history_masks: List[int] = []
        self.latest_reds: Optional[List[int]] = None
        self._load_history(history_df)

    def _load_history(self, history_df: Optional[pd.DataFrame] = None) -> None:
        """Load 3,530+ historical draws for O(1) duplicate checks and bitmask matching."""
        if history_df is None:
            history_df = storage.get_all_draws()

        if history_df is not None and not history_df.empty:
            for _, row in history_df.iterrows():
                try:
                    reds = feat_module.get_red_balls_from_row(row)
                    if len(reds) == 6:
                        tup = tuple(sorted(reds))
                        self.history_red_tuples.add(tup)
                        mask = sum(1 << (r - 1) for r in tup)
                        self.history_masks.append(mask)
                except Exception:
                    continue

            # Store latest draw reds
            latest_row = history_df.iloc[-1]
            try:
                self.latest_reds = feat_module.get_red_balls_from_row(latest_row)
            except Exception:
                self.latest_reds = None

        logger.debug(f'AntiPopularityFilter loaded {len(self.history_red_tuples)} historical draws.')

    def is_valid(self, reds: List[int], blue: int) -> Tuple[bool, List[str]]:
        """Determine if a combination passes hard anti-popularity filtering.
        
        Rejection criteria:
        1. Exact duplicate of any historical draw in SQLite
        2. Strong arithmetic progression (6-ball AP or any 5-ball AP)
        3. Consecutive sequence (>= 3 consecutive numbers)
        4. Pure month zone crowding (all 6 reds <= 12)
        5. Extreme last-digit collision (>= 4 balls share the same last digit)
        6. Extreme odd/even ratio (0:6 all even or 6:0 all odd)
        7. Extreme sum value (< 70 or > 145)
        
        Returns:
            (is_valid: bool, reject_reasons: list[str])
        """
        sorted_reds = sorted(reds)
        reasons: List[str] = []

        # 1. Past draw duplicate check
        if tuple(sorted_reds) in self.history_red_tuples:
            reasons.append('命中历史开奖原号')

        # 2. Arithmetic progression check (6-ball or any 5-ball subset)
        diffs = [sorted_reds[i + 1] - sorted_reds[i] for i in range(5)]
        if len(set(diffs)) == 1 and diffs[0] > 0:
            reasons.append('6红球完整等差数列')
        else:
            for sub5 in itertools.combinations(sorted_reds, 5):
                d5 = [sub5[i + 1] - sub5[i] for i in range(4)]
                if len(set(d5)) == 1 and d5[0] > 0:
                    reasons.append('包含5项等差数列')
                    break

        # 3. Consecutive numbers check (>= 3)
        max_consec = 1
        curr_consec = 1
        for i in range(len(sorted_reds) - 1):
            if sorted_reds[i + 1] == sorted_reds[i] + 1:
                curr_consec += 1
                max_consec = max(max_consec, curr_consec)
            else:
                curr_consec = 1
        if max_consec >= 3:
            reasons.append(f'包含{max_consec}连号')

        # 4. Pure month zone crowding (all <= 12)
        if all(r <= 12 for r in sorted_reds):
            reasons.append('全部红球集中在月份区(01-12)')

        # 5. Extreme last-digit collision (>= 4)
        tails = [r % 10 for r in sorted_reds]
        tail_counts = Counter(tails)
        if any(c >= 4 for c in tail_counts.values()):
            reasons.append('4个及以上同尾号')

        # 6. Extreme odd/even ratio (0:6 or 6:0)
        odd_count = sum(1 for r in sorted_reds if r % 2 == 1)
        even_count = 6 - odd_count
        if odd_count == 0:
            reasons.append('全偶数(0:6)')
        elif even_count == 0:
            reasons.append('全奇数(6:0)')

        # 7. Extreme sum value
        s = sum(sorted_reds)
        if s < 70:
            reasons.append(f'极端低和值({s}<70)')
        elif s > 145:
            reasons.append(f'极端高和值({s}>145)')

        return len(reasons) == 0, reasons


class UniquenessScorer:
    """Soft-scores lottery combination uniqueness [0.0, 1.0] based on behavioral game theory."""

    def __init__(
        self,
        history_df: Optional[pd.DataFrame] = None,
        latest_reds: Optional[List[int]] = None,
    ) -> None:
        self.history_masks: List[int] = []
        self.latest_reds: Optional[List[int]] = latest_reds
        self._init_history(history_df)

    def _init_history(self, history_df: Optional[pd.DataFrame] = None) -> None:
        if history_df is None:
            history_df = storage.get_all_draws()
        if history_df is not None and not history_df.empty:
            for _, row in history_df.iterrows():
                try:
                    reds = feat_module.get_red_balls_from_row(row)
                    if len(reds) == 6:
                        mask = sum(1 << (r - 1) for r in reds)
                        self.history_masks.append(mask)
                except Exception:
                    continue
            if self.latest_reds is None and not history_df.empty:
                self.latest_reds = feat_module.get_red_balls_from_row(history_df.iloc[-1])

    def score(self, reds: List[int], blue: int, mode: str = 'balanced') -> Dict[str, Any]:
        """Calculate multi-dimensional uniqueness score.
        
        Weights:
        - 破除生日偏置 (30%): >=32 加分, 全<=31 扣分, 01-12扎堆扣分
        - 次级规律与视觉惩罚 (20%): 4项等差、镜像对、步长过于均匀扣分
        - 文化与心理反转 (20%): 6/8/9吉利尾数扣分, 4及回避蓝加分(jackpot强/balanced弱), 热门蓝扣分
        - 历史高重合度 (15%): 历史重合5红或与上期重合>=4扣分
        - 奇偶均衡度 (15%): 3:3/4:2/2:4不扣分, 1:5/5:1扣分
        
        Returns:
            dict containing uniqueness_score (0..1), breakdown, and tags
        """
        sorted_reds = sorted(reds)
        tags: List[str] = []

        # -------------------------------------------------------------
        # 1. 破除生日偏置 (30%)
        # -------------------------------------------------------------
        s_birthday = 0.50
        has_32 = 32 in sorted_reds
        has_33 = 33 in sorted_reds
        if has_32 and has_33:
            s_birthday += 0.40
            tags.append('含32/33冷号')
        elif has_32 or has_33:
            s_birthday += 0.25
            tags.append('含>31冷号')

        if all(r <= 31 for r in sorted_reds):
            s_birthday -= 0.25
        else:
            tags.append('破除生日偏置')

        month_count = sum(1 for r in sorted_reds if r <= 12)
        if month_count > 3:
            s_birthday -= 0.25
            tags.append('月份区密集')
        elif month_count <= 2:
            s_birthday += 0.10
            tags.append('避开生日扎堆')

        s_birthday = float(np.clip(s_birthday, 0.0, 1.0))

        # -------------------------------------------------------------
        # 2. 次级规律与视觉图形惩罚 (20%)
        # -------------------------------------------------------------
        s_pattern = 0.85

        # 4-item sub arithmetic progression
        has_4_ap = False
        for sub4 in itertools.combinations(sorted_reds, 4):
            d4 = [sub4[i + 1] - sub4[i] for i in range(3)]
            if len(set(d4)) == 1 and d4[0] > 0:
                has_4_ap = True
                break
        if has_4_ap:
            s_pattern -= 0.35
            tags.append('含4项次级等差')

        # 3-consecutive natural numbers
        has_3_consec = any(sorted_reds[i + 2] == sorted_reds[i] + 2 and sorted_reds[i + 1] == sorted_reds[i] + 1 for i in range(len(sorted_reds) - 2))
        if has_3_consec:
            s_pattern -= 0.50
            tags.append('含3连号')

        # Mirror number pairs
        red_set = set(sorted_reds)
        mirror_hits = sum(1 for p1, p2 in MIRROR_PAIRS if p1 in red_set and p2 in red_set)
        if mirror_hits > 0:
            s_pattern -= 0.15 * mirror_hits
            tags.append(f'含{mirror_hits}组镜像号')

        # Overly uniform step distribution
        diffs = np.diff(sorted_reds)
        if len(diffs) > 0 and np.std(diffs) < 1.0:
            s_pattern -= 0.15
            tags.append('步长过于均匀')

        s_pattern = float(np.clip(s_pattern, 0.0, 1.0))

        # -------------------------------------------------------------
        # 3. 文化与心理偏置反转 (20%)
        # -------------------------------------------------------------
        s_cultural = 0.50

        # Lucky endings 6, 8, 9
        lucky_count = sum(1 for r in sorted_reds if r % 10 in LUCKY_TAILS)
        if lucky_count >= 4:
            s_cultural -= 0.30
            tags.append('吉利尾数扎堆')
        elif lucky_count <= 1:
            s_cultural += 0.15
            tags.append('反吉利偏置')

        # Avoided numbers containing 4: capped bonus (1 hit = full bonus, no linear stacking)
        avoided_hits = sum(1 for r in sorted_reds if r in AVOIDED_REDS)
        if avoided_hits > 0:
            s_cultural += 0.10  # Fixed cap: 1 hit = 0.10, no more stacking
            tags.append('含反偏好冷号4')

        # Blue ball psychological traits (mode-aware)
        blue_weight_scale = 0.30 if mode == 'balanced' else 1.0
        if blue in AVOIDED_BLUES:
            s_cultural += 0.15 * blue_weight_scale
            if blue_weight_scale >= 0.5:
                tags.append('大众回避蓝')
        elif blue in POPULAR_BLUES:
            s_cultural -= 0.15 * blue_weight_scale

        s_cultural = float(np.clip(s_cultural, 0.0, 1.0))

        # -------------------------------------------------------------
        # 4. 历史高重合度惩罚 (15%)
        # -------------------------------------------------------------
        s_history = 0.85
        cand_mask = sum(1 << (r - 1) for r in sorted_reds)

        # Check maximum overlap with any past draw
        max_hist_overlap = 0
        if self.history_masks:
            max_hist_overlap = max((m & cand_mask).bit_count() for m in self.history_masks)

        if max_hist_overlap >= 5:
            s_history -= 0.50
            tags.insert(0, '历史高重合(5红)')
        elif self.history_masks:
            tags.append('无历史重号')

        # Overlap with the immediate previous draw
        if self.latest_reds:
            last_overlap = len(set(sorted_reds) & set(self.latest_reds))
            if last_overlap >= 4:
                s_history -= 0.40
                tags.insert(0, f'与上期重合{last_overlap}号')
            elif last_overlap == 0:
                s_history += 0.10
                tags.append('与上期完全隔离')

        s_history = float(np.clip(s_history, 0.0, 1.0))

        # -------------------------------------------------------------
        # 5. 奇偶均衡度 (15%)
        # -------------------------------------------------------------
        odd_count = sum(1 for r in sorted_reds if r % 2 == 1)
        even_count = 6 - odd_count
        s_parity = 0.70

        if odd_count == 3:  # 3:3 ideal
            s_parity += 0.25
            tags.append('奇偶均衡3:3')
        elif odd_count in (2, 4):  # 2:4 or 4:2 acceptable
            s_parity += 0.10
        elif odd_count in (1, 5):  # 1:5 or 5:1 mild penalty
            s_parity -= 0.25
            tags.append(f'奇偶偏斜{odd_count}:{even_count}')
        # 0:6 and 6:0 are already hard-filtered out

        s_parity = float(np.clip(s_parity, 0.0, 1.0))

        # Weighted final score
        final_score = (
            0.30 * s_birthday
            + 0.20 * s_pattern
            + 0.20 * s_cultural
            + 0.15 * s_history
            + 0.15 * s_parity
        )
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # De-duplicate tags preserving order
        seen_tags = set()
        clean_tags = []
        for t in tags:
            if t not in seen_tags:
                seen_tags.add(t)
                clean_tags.append(t)

        return {
            'uniqueness_score': round(final_score, 4),
            'breakdown': {
                'birthday_bias': round(s_birthday, 4),
                'pattern_penalty': round(s_pattern, 4),
                'cultural_bias': round(s_cultural, 4),
                'history_overlap': round(s_history, 4),
                'parity_balance': round(s_parity, 4),
            },
            'tags': clean_tags[:5],  # Top distinctive tags
        }


class CrowdBettingSimulator:
    """Simulates 50,000 biased casual bets to quantify collision rates and jackpot share gains."""

    def __init__(
        self,
        last_draw_reds: Optional[List[int]] = None,
        seed: int = 42,
        quick_pick_ratio: float = config.CROWD_QUICK_PICK_RATIO,
        birthday_ratio: float = config.CROWD_BIRTHDAY_RATIO,
        hot_follow_ratio: float = config.CROWD_HOT_FOLLOW_RATIO,
        pattern_ratio: float = config.CROWD_PATTERN_RATIO,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_draw_reds = last_draw_reds or []
        self.quick_pick_ratio = quick_pick_ratio
        self.birthday_ratio = birthday_ratio
        self.hot_follow_ratio = hot_follow_ratio
        self.pattern_ratio = pattern_ratio
        self._init_distributions()

    def _init_distributions(self) -> None:
        """Construct realistic empirical weight distributions for crowd bettors."""
        # Red ball weights reflecting human cognitive biases
        rw = np.ones(33, dtype=float)
        for i in range(1, 34):
            # 1. Birthday bias: heavily favor <= 31, penalize 32, 33
            if i <= 31:
                rw[i - 1] *= 1.35
            else:
                rw[i - 1] *= 0.35

            # 2. Lucky endings: 6, 8, 9
            if i % 10 in LUCKY_TAILS:
                rw[i - 1] *= 1.30

            # 3. Superstitious avoidance of 4
            if i in AVOIDED_REDS:
                rw[i - 1] *= 0.70

            # 4. Previous draw chasing
            if i in self.last_draw_reds:
                rw[i - 1] *= 1.35

        self.red_weights = rw / rw.sum()

        # Blue ball weights
        bw = np.ones(16, dtype=float)
        for b in POPULAR_BLUES:
            bw[b - 1] *= 1.75
        for b in AVOIDED_BLUES:
            bw[b - 1] *= 0.55
        self.blue_weights = bw / bw.sum()

        # Prior uniform probability of single ticket (1 in 17,721,088)
        self.p_uniform = 1.0 / (math.comb(33, 6) * 16)

    def compute_relative_crowd_popularity(self, reds: List[int], blue: int) -> float:
        """Calculate the theoretical probability ratio of the crowd choosing this ticket vs random.
        
        Relative Popularity = P_crowd(T) / P_uniform(T)
        With market structure (60% quick-pick, 40% biased human):
        P_crowd(T) = quick_pick_ratio * P_uniform(T) + (1 - quick_pick_ratio) * P_biased(T)
        Therefore:
        Relative Popularity = quick_pick_ratio + (1 - quick_pick_ratio) * (P_biased(T) / P_uniform(T))
        """
        sorted_reds = sorted(reds)
        rem_w = self.red_weights.copy()
        p_red = 1.0
        for r in sorted_reds:
            s = rem_w.sum()
            if s <= 0:
                p_red = 0.0
                break
            p_red *= (rem_w[r - 1] / s)
            rem_w[r - 1] = 0.0
        p_red *= math.factorial(6)

        p_blue = float(self.blue_weights[blue - 1])
        p_ticket_biased = p_red * p_blue
        biased_ratio = float(p_ticket_biased / self.p_uniform)

        # Blend with 60% uniform quick pick market baseline
        pop = self.quick_pick_ratio * 1.0 + (1.0 - self.quick_pick_ratio) * biased_ratio
        return float(pop)

    def generate_crowd_pool(self, n_bets: int = 50_000) -> List[Tuple[Tuple[int, ...], int]]:
        """Generate N simulated virtual crowd bets with realistic heuristics (60% quick pick + 40% biased)."""
        crowd_bets = []
        red_indices = np.arange(1, 34)

        for _ in range(n_bets):
            r_val = self.rng.random()
            if r_val < self.quick_pick_ratio:
                # 60% completely random uniform quick pick
                reds = tuple(sorted(self.rng.choice(red_indices, size=6, replace=False)))
                blue = int(self.rng.integers(1, 17))
            elif r_val < self.quick_pick_ratio + self.pattern_ratio:
                # Pattern bets (arithmetic, consecutive, or month zone)
                pat_type = self.rng.choice(3)
                if pat_type == 0:
                    start = int(self.rng.integers(1, 28))
                    reds = tuple(range(start, start + 6))
                elif pat_type == 1:
                    start = int(self.rng.integers(1, 4))
                    step = int(self.rng.choice([2, 3, 5]))
                    reds = tuple(start + i * step for i in range(6) if start + i * step <= 33)
                    if len(reds) < 6:
                        reds = tuple(sorted(self.rng.choice(red_indices, size=6, replace=False, p=self.red_weights)))
                else:
                    reds = tuple(sorted(self.rng.choice(np.arange(1, 13), size=6, replace=False)))
                blue = int(self.rng.choice(np.arange(1, 17), p=self.blue_weights))
            else:
                # Biased cognitive bets (birthday + lucky tails + hot chasing)
                reds = tuple(sorted(self.rng.choice(red_indices, size=6, replace=False, p=self.red_weights)))
                blue = int(self.rng.choice(np.arange(1, 17), p=self.blue_weights))

            crowd_bets.append((reds, blue))

        return crowd_bets

    def evaluate_strategy_collision(
        self,
        tickets: List[Dict[str, Any]],
        crowd_pool: Optional[List[Tuple[Tuple[int, ...], int]]] = None,
    ) -> Dict[str, float]:
        """Quantify how vulnerable a set of tickets is to crowd collision."""
        if not tickets:
            return {'avg_popularity': 1.0, 'share_multiplier': 1.0, 'co_winners_reduction_pct': 0.0}

        pops = [self.compute_relative_crowd_popularity(t['reds'], t['blue']) for t in tickets]
        avg_pop = float(np.mean(pops))
        share_mult = float(1.0 / avg_pop) if avg_pop > 0 else 1.0
        reduction_pct = float(max(0.0, (1.0 - avg_pop) * 100.0)) if avg_pop < 1.0 else 0.0

        return {
            'avg_popularity': round(avg_pop, 3),
            'share_multiplier': round(share_mult, 2),
            'co_winners_reduction_pct': round(reduction_pct, 1),
        }


def select_game_theory_combinations(
    model: Any,
    n_combinations: int = 5,
    mode: str = 'balanced',
    beta: float = config.GAME_THEORY_BETA,
    max_ball_freq: int = config.MAX_BALL_FREQ,
    pool_size: int = 300,
    df: Optional[pd.DataFrame] = None,
    filter_obj: Optional[AntiPopularityFilter] = None,
    scorer_obj: Optional[UniquenessScorer] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate candidates from model, filter high-collision, normalize & re-rank, and sample N tickets.
    
    Modes:
    - 'balanced' (default): 5 distinct blue balls, red overlap <= 3, max_ball_freq <= 2.
    - 'jackpot': Lock 1 blue ball via composite score, red overlap <= 2, max_ball_freq <= 2.
    
    Returns:
        (selected_combinations, stats: {'pool_size', 'rejected_count', 'passed_count', 'mode', 'beta'})
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    probs = model.get_probabilities()
    red_probs = probs['red']
    blue_probs = probs['blue']

    if filter_obj is None:
        filter_obj = AntiPopularityFilter(history_df=df)
    if scorer_obj is None:
        scorer_obj = UniquenessScorer(history_df=df)

    red_nums = list(range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1))
    red_weights = np.array([red_probs.get(n, 1.0 / 33.0) for n in red_nums], dtype=float)
    red_weights = np.maximum(red_weights, 1e-6)
    red_weights /= red_weights.sum()

    sorted_blues = sorted(blue_probs.keys(), key=lambda b: blue_probs[b], reverse=True)

    # 1. Sample raw candidate pool from model
    candidates: List[Dict[str, Any]] = []
    rejected_count = 0

    # Ensure sufficient candidates per blue ball
    attempts = max(pool_size * 3, 1000)
    for _ in range(attempts):
        if len(candidates) >= pool_size:
            break

        # Distribute blue balls across top options
        b_idx = len(candidates) % min(10, len(sorted_blues))
        b = sorted_blues[b_idx]

        # Sample 6 distinct red balls
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

        # Hard filtering
        is_pass, reasons = filter_obj.is_valid(reds, b)
        if not is_pass:
            rejected_count += 1
            continue

        # Soft uniqueness scoring
        uniq_res = scorer_obj.score(reds, b, mode=mode)
        model_score = model.score_combination(reds, b)
        u_score = uniq_res['uniqueness_score']

        odd, even = feat_module.compute_odd_even_ratio(reds)
        candidates.append({
            'reds': reds,
            'blue': b,
            'model_score': round(model_score, 4),
            'uniqueness_score': round(u_score, 4),
            'tags': uniq_res['tags'],
            'features': {
                'sum_value': int(sum(reds)),
                'odd_even_ratio': f'{odd}:{even}',
                'ac_value': int(feat_module.compute_ac_value(reds)),
                'consecutive': int(feat_module.compute_consecutive_count(reds)),
            },
        })

    # Fallback if candidates were insufficient
    if not candidates:
        logger.warning('Candidate pool empty after filtering, falling back to top model prediction.')
        raw_pred = model.predict(n_combinations=n_combinations, df=df)
        for p in raw_pred:
            u_res = scorer_obj.score(p['reds'], p['blue'], mode=mode)
            p['uniqueness_score'] = u_res['uniqueness_score']
            p['final_score'] = p['score']
            p['tags'] = u_res['tags']
        return raw_pred, {'pool_size': pool_size, 'rejected_count': 0, 'passed_count': len(raw_pred), 'mode': mode, 'beta': beta}

    # 2. Candidate pool Min-Max normalization for balanced hybrid ranking
    m_scores = [c['model_score'] for c in candidates]
    u_scores = [c['uniqueness_score'] for c in candidates]
    min_m, max_m = min(m_scores), max(m_scores)
    min_u, max_u = min(u_scores), max(u_scores)
    denom_m = (max_m - min_m) + 1e-9
    denom_u = (max_u - min_u) + 1e-9

    for c in candidates:
        norm_m = (c['model_score'] - min_m) / denom_m
        norm_u = (c['uniqueness_score'] - min_u) / denom_u
        c['norm_model'] = round(float(norm_m), 4)
        c['norm_uniqueness'] = round(float(norm_u), 4)
        c['final_score'] = round(float((1.0 - beta) * norm_m + beta * norm_u), 4)

    # Sort candidates by final_score descending
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    # 3. Select N tickets based on mode with max_ball_freq constraint
    selected: List[Dict[str, Any]] = []

    if mode == 'jackpot':
        # Lock single blue combining model probability and anti-popularity uniqueness
        blue_model_probs = [blue_probs.get(b, 1.0 / 16.0) for b in range(1, 17)]
        blue_uniqueness = []
        for b in range(1, 17):
            if b in POPULAR_BLUES:
                bu = 0.20  # Avoid 6, 7, 8, 9, 16
            elif b in AVOIDED_BLUES:
                bu = 0.85  # Boost 4, 11, 13, 14, 15
            else:
                bu = 0.50
            blue_uniqueness.append(bu)

        min_bm, max_bm = min(blue_model_probs), max(blue_model_probs)
        min_bu, max_bu = min(blue_uniqueness), max(blue_uniqueness)
        denom_bm = (max_bm - min_bm) + 1e-9
        denom_bu = (max_bu - min_bu) + 1e-9

        blue_composite = {}
        for idx, b in enumerate(range(1, 17)):
            norm_bm = (blue_model_probs[idx] - min_bm) / denom_bm
            norm_bu = (blue_uniqueness[idx] - min_bu) / denom_bu
            blue_composite[b] = (1.0 - beta) * norm_bm + beta * norm_bu

        locked_blue = max(blue_composite.keys(), key=lambda b: blue_composite[b])

        # Filter candidates for locked blue (or assign locked_blue if needed)
        jackpot_cands = [c for c in candidates if c['blue'] == locked_blue]
        if len(jackpot_cands) < n_combinations:
            for c in candidates:
                if c['blue'] != locked_blue and len(jackpot_cands) < pool_size:
                    c_copy = dict(c)
                    c_copy['blue'] = locked_blue
                    jackpot_cands.append(c_copy)

        # Spread red combinations with overlap <= 2 and single ball freq <= max_ball_freq
        for max_ov in [2, 3, 4, 6]:
            for max_freq in [max_ball_freq, max_ball_freq + 1, n_combinations]:
                selected = []
                ball_counts: Counter = Counter()
                for c in jackpot_cands:
                    if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in selected):
                        continue
                    if any(ball_counts[r] >= max_freq for r in c['reds']):
                        continue
                    selected.append(c)
                    for r in c['reds']:
                        ball_counts[r] += 1
                    if len(selected) == n_combinations:
                        break
                if len(selected) == n_combinations:
                    break
            if len(selected) == n_combinations:
                break
    else:
        # Balanced mode: 5 distinct blue balls, red overlap <= 3, single ball freq <= max_ball_freq
        for max_ov in [config.MAX_RED_OVERLAP, config.MAX_RED_OVERLAP + 1, 6]:
            for max_freq in [max_ball_freq, max_ball_freq + 1, n_combinations]:
                selected = []
                used_blues: Set[int] = set()
                ball_counts: Counter = Counter()
                for c in candidates:
                    if c['blue'] in used_blues:
                        continue
                    if any(len(set(c['reds']) & set(prev['reds'])) > max_ov for prev in selected):
                        continue
                    if any(ball_counts[r] >= max_freq for r in c['reds']):
                        continue
                    selected.append(c)
                    used_blues.add(c['blue'])
                    for r in c['reds']:
                        ball_counts[r] += 1
                    if len(selected) == n_combinations:
                        break
                if len(selected) == n_combinations:
                    break
            if len(selected) == n_combinations:
                break

    # If still fewer than requested, fill with top remaining candidates
    while len(selected) < n_combinations and len(selected) < len(candidates):
        for c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) == n_combinations:
                    break

    stats = {
        'pool_size': len(candidates) + rejected_count,
        'rejected_count': rejected_count,
        'passed_count': len(candidates),
        'mode': mode,
        'beta': beta,
        'max_ball_freq': max_ball_freq,
        'locked_blue': locked_blue if mode == 'jackpot' else None,
    }

    return selected, stats

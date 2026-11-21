"""Unit tests for Game Theory Anti-Popularity module."""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.game_theory import (
    AntiPopularityFilter,
    UniquenessScorer,
    CrowdBettingSimulator,
    select_game_theory_combinations,
)
from src.model import EnsembleModel


class TestAntiPopularityFilter:
    def test_past_draw_intercept(self):
        """Verify exact historical winning numbers are intercepted."""
        mock_df = pd.DataFrame([{
            'issue': '03001',
            'draw_date': '2003-02-23',
            'red_1': 1, 'red_2': 2, 'red_3': 3, 'red_4': 4, 'red_5': 5, 'red_6': 6,
            'blue': 7,
        }])
        f = AntiPopularityFilter(history_df=mock_df)
        is_pass, reasons = f.is_valid([1, 2, 3, 4, 5, 6], 7)
        assert not is_pass
        assert any('历史开奖原号' in r for r in reasons)

    def test_arithmetic_progression_6_intercept(self):
        """Verify 6-ball arithmetic progression is intercepted."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([5, 10, 15, 20, 25, 30], 8)
        assert not is_pass
        assert any('6红球完整等差数列' in r for r in reasons)

    def test_arithmetic_progression_5_intercept(self):
        """Verify any 5-ball subset forming an arithmetic progression is intercepted."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([2, 4, 6, 8, 10, 31], 5)
        assert not is_pass
        assert any('5项等差数列' in r for r in reasons)

    def test_consecutive_4_intercept(self):
        """Verify 4 or more consecutive natural numbers are intercepted."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([7, 8, 9, 10, 18, 25], 3)
        assert not is_pass
        assert any('连号' in r for r in reasons)

    def test_month_zone_intercept(self):
        """Verify combination with all 6 balls <= 12 is intercepted."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([1, 3, 5, 7, 9, 11], 2)
        assert not is_pass
        assert any('月份区' in r for r in reasons)

    def test_extreme_tails_intercept(self):
        """Verify combination with >= 4 same last digits is intercepted."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([3, 13, 23, 33, 7, 18], 9)
        assert not is_pass
        assert any('同尾号' in r for r in reasons)

    def test_valid_anti_popular_ticket_passes(self):
        """Verify high-quality anti-popular ticket passes clean."""
        f = AntiPopularityFilter(history_df=pd.DataFrame())
        is_pass, reasons = f.is_valid([4, 11, 17, 23, 32, 33], 4)
        assert is_pass
        assert len(reasons) == 0


class TestUniquenessScorer:
    def test_32_33_bonus_vs_birthday(self):
        """Verify combinations with 32 and 33 receive significant uniqueness bonus."""
        scorer = UniquenessScorer(history_df=pd.DataFrame())
        
        # Ticket with 32 and 33, avoided 4, cold blue 4
        res_cold = scorer.score([4, 11, 17, 23, 32, 33], 4)
        # Typical casual birthday ticket all <= 31, lucky blue 8
        res_crowd = scorer.score([2, 6, 8, 12, 18, 28], 8)

        assert res_cold['uniqueness_score'] > res_crowd['uniqueness_score']
        assert '含32/33冷号' in res_cold['tags'] or '含>31冷号' in res_cold['tags']
        assert '破除生日偏置' in res_cold['tags']
        assert '吉利尾数扎堆' in res_crowd['tags']

    def test_pattern_and_mirror_penalty(self):
        """Verify mirror pairs and 3-consecutive numbers are penalized."""
        scorer = UniquenessScorer(history_df=pd.DataFrame())
        res = scorer.score([1, 10, 14, 15, 16, 32], 2)
        assert any('镜像号' in t for t in res['tags']) or any('3连号' in t for t in res['tags'])

    def test_history_recent_overlap_penalty(self):
        """Verify overlap >= 4 with previous draw receives penalty."""
        last_reds = [5, 10, 15, 20, 25, 30]
        scorer = UniquenessScorer(history_df=pd.DataFrame(), latest_reds=last_reds)
        res = scorer.score([5, 10, 15, 20, 28, 32], 3)
        assert any('与上期重合' in t for t in res['tags'])
        assert res['breakdown']['history_overlap'] < 0.60


class TestCrowdBettingSimulator:
    def test_crowd_pool_generation(self):
        sim = CrowdBettingSimulator(last_draw_reds=[1, 2, 3, 4, 5, 6], seed=42)
        bets = sim.generate_crowd_pool(n_bets=200)
        assert len(bets) == 200
        for reds, blue in bets:
            assert len(reds) == 6
            assert 1 <= blue <= 16

    def test_relative_crowd_popularity(self):
        sim = CrowdBettingSimulator(last_draw_reds=[1, 2, 3, 4, 5, 6], seed=42)
        # Super popular combination
        pop_high = sim.compute_relative_crowd_popularity([6, 8, 16, 18, 26, 28], 8)
        # Super cold combination
        pop_low = sim.compute_relative_crowd_popularity([4, 11, 17, 23, 32, 33], 4)
        assert pop_high > 1.1
        assert pop_low < 0.8
        assert sim.quick_pick_ratio == 0.60


class TestSelectGameTheoryCombinations:
    @pytest.fixture
    def mock_fitted_ensemble(self):
        mock_df = pd.DataFrame([
            {
                'issue': f'24{i:03d}',
                'draw_date': '2024-01-01',
                'red_1': (i % 28) + 1,
                'red_2': (i % 28) + 2,
                'red_3': (i % 28) + 3,
                'red_4': (i % 28) + 4,
                'red_5': (i % 28) + 5,
                'red_6': (i % 28) + 6,
                'blue': (i % 16) + 1,
            }
            for i in range(100)
        ])
        m = EnsembleModel()
        m.fit(mock_df)
        return m, mock_df

    def test_balanced_mode_distinct_blues_and_max_freq(self, mock_fitted_ensemble):
        from collections import Counter
        m, df = mock_fitted_ensemble
        preds, stats = select_game_theory_combinations(
            m, n_combinations=5, mode='balanced', beta=0.35, max_ball_freq=2, df=df, random_seed=42
        )
        assert len(preds) == 5
        blues = [p['blue'] for p in preds]
        assert len(set(blues)) == 5
        assert stats['mode'] == 'balanced'

        # Check cross-ticket max_ball_freq <= 2
        all_reds = [r for p in preds for r in p['reds']]
        counts = Counter(all_reds)
        assert counts.most_common(1)[0][1] <= 2

        # Check normalization fields exist and are in [0, 1]
        for p in preds:
            assert 0.0 <= p['norm_model'] <= 1.0
            assert 0.0 <= p['norm_uniqueness'] <= 1.0

    def test_jackpot_mode_single_blue_lock_and_max_freq(self, mock_fitted_ensemble):
        from collections import Counter
        m, df = mock_fitted_ensemble
        preds, stats = select_game_theory_combinations(
            m, n_combinations=5, mode='jackpot', beta=0.35, max_ball_freq=2, df=df, random_seed=42
        )
        assert len(preds) == 5
        blues = [p['blue'] for p in preds]
        # All 5 tickets must lock the exact same single blue ball
        assert len(set(blues)) == 1
        assert stats['mode'] == 'jackpot'

        # Check cross-ticket max_ball_freq <= 2
        all_reds = [r for p in preds for r in p['reds']]
        counts = Counter(all_reds)
        assert counts.most_common(1)[0][1] <= 2

        # Red ball overlap constraint: max overlap between pairs <= 2 (or 3)
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                overlap = len(set(preds[i]['reds']) & set(preds[j]['reds']))
                assert overlap <= 3

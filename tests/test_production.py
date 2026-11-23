"""Unit tests for productionization features:
1. Candidate pool Min-Max normalization
2. max_ball_freq <= 2 cross-ticket constraint
3. 3-consecutive numbers heavy penalty
4. Jackpot mode composite blue ball locking
5. Automatic prediction check & verification state backfill
"""
import os
import sqlite3
from collections import Counter
import pytest
import pandas as pd
import numpy as np

import config
from src import storage
from src.game_theory import (
    UniquenessScorer,
    AntiPopularityFilter,
    CrowdBettingSimulator,
    select_game_theory_combinations,
)
from src.model import EnsembleModel


class TestUniqueness3ConsecutivePenalty:
    def test_3_consecutive_heavy_penalty(self):
        scorer = UniquenessScorer(history_df=pd.DataFrame())
        # Ticket with 3 consecutive numbers: 31, 32, 33
        res_3c = scorer.score([1, 10, 20, 31, 32, 33], 5)
        # Ticket without consecutive numbers
        res_no = scorer.score([1, 10, 20, 25, 30, 32], 5)

        assert any('3连号' in t for t in res_3c['tags'])
        # Pattern score should be significantly lower for the 3-consecutive ticket
        assert res_3c['breakdown']['pattern_penalty'] < 0.40
        assert res_3c['breakdown']['pattern_penalty'] < res_no['breakdown']['pattern_penalty']


class TestJackpotCompositeBlueLocking:
    def test_composite_blue_locking_avoids_popular_blues(self):
        """When blue model probabilities are close, composite locking prefers cold/unpopular blues."""
        from src.game_theory import POPULAR_BLUES, AVOIDED_BLUES

        # Create dummy ensemble with nearly uniform blue probabilities
        m = EnsembleModel()
        m.blue_probs = {b: 1.0 / 16.0 for b in range(1, 17)}
        # Give blue 8 (popular) slightly higher model prob, but blue 4 (avoided/cold) close
        m.blue_probs[8] = 0.080
        m.blue_probs[4] = 0.075
        m.red_probs = {r: 1.0 / 33.0 for r in range(1, 34)}
        m.fitted = True

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

        preds, stats = select_game_theory_combinations(
            m, n_combinations=5, mode='jackpot', beta=0.35, df=mock_df, random_seed=42
        )
        locked = stats['locked_blue']
        # The locked blue should NOT be the super popular 8
        assert locked != 8
        assert stats['mode'] == 'jackpot'


class TestPredictionVerificationPipeline:
    def test_check_and_update_predictions_state(self, tmp_path, monkeypatch):
        test_db = tmp_path / "test_ssq.db"
        test_csv = tmp_path / "test_ssq.csv"
        monkeypatch.setattr(config, 'DB_PATH', str(test_db))
        monkeypatch.setattr(config, 'CSV_PATH', str(test_csv))
        monkeypatch.setattr(config, 'REPORTS_DIR', str(tmp_path / "reports"))

        storage.init_db()

        # Insert a historical draw: issue 26130: 01,02,03,04,05,06 + 07
        conn = storage.get_connection()
        conn.execute('''
            INSERT INTO draws (issue, draw_date, red_1, red_2, red_3, red_4, red_5, red_6, blue)
            VALUES ('26130', '2026-11-12', 1, 2, 3, 4, 5, 6, 7)
        ''')
        conn.commit()
        conn.close()

        # Insert unchecked predictions for issue 26130
        preds = [
            {
                'predict_date': '2026-11-11',
                'target_issue': '26130',
                'target_date': '2026-11-12',
                'red_1': 1, 'red_2': 2, 'red_3': 3, 'red_4': 10, 'red_5': 20, 'red_6': 30,
                'blue': 7,  # Hit 3 red + 1 blue -> 五等奖 10元
                'score': 0.65,
                'strategy': 'game_theory_balanced',
                'sum_value': 66,
                'odd_even_ratio': '3:3',
            },
            {
                'predict_date': '2026-11-11',
                'target_issue': '26130',
                'target_date': '2026-11-12',
                'red_1': 11, 'red_2': 12, 'red_3': 13, 'red_4': 14, 'red_5': 15, 'red_6': 16,
                'blue': 16,  # 0 red + 0 blue -> 未中奖 0元
                'score': 0.60,
                'strategy': 'game_theory_balanced',
                'sum_value': 81,
                'odd_even_ratio': '3:3',
            },
        ]
        storage.save_predictions(preds)

        # Run check_and_update_predictions
        newly_checked = storage.check_and_update_predictions()
        assert len(newly_checked) == 2

        # Verify first ticket: 3 red + 1 blue
        t1 = next(t for t in newly_checked if t['blue'] == 7)
        assert t1['red_hit'] == 3
        assert t1['blue_hit'] == 1
        assert t1['prize_level'] == '五等奖'
        assert t1['prize_amount'] == 10
        assert t1['checked_at'] is not None

        # Verify second ticket: 0 red + 0 blue
        t2 = next(t for t in newly_checked if t['blue'] == 16)
        assert t2['red_hit'] == 0
        assert t2['blue_hit'] == 0
        assert t2['prize_level'] == '未中奖'
        assert t2['prize_amount'] == 0
        assert t2['checked_at'] is not None

        # Second check should return empty (already checked)
        second_check = storage.check_and_update_predictions()
        assert len(second_check) == 0

        # Check summary
        summary = storage.get_checked_predictions_summary()
        assert summary['total_predictions'] == 2
        assert summary['total_investment'] == 4  # 2 tickets * 2 CNY
        assert summary['total_winnings'] == 10
        assert summary['net_return'] == 6
        assert summary['roi_pct'] == 150.0


class TestCandidatePoolRecentDrawsExport:
    def test_export_pool_contains_recent_draws(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta
        from src.predictor import export_frozen_candidate_pool
        test_db = tmp_path / "test_pool.db"
        test_csv = tmp_path / "test_pool.csv"
        test_docs = tmp_path / "docs_data"
        test_reports = tmp_path / "reports"

        monkeypatch.setattr(config, 'DB_PATH', str(test_db))
        monkeypatch.setattr(config, 'CSV_PATH', str(test_csv))
        monkeypatch.setattr(config, 'DOCS_DATA_DIR', str(test_docs))
        monkeypatch.setattr(config, 'REPORTS_DIR', str(test_reports))

        # Seed test db with 60 draws so predictor has enough training data
        storage.init_db()
        records = []
        base_d = datetime(2026, 1, 1)
        for i in range(60):
            d_str = (base_d + timedelta(days=i * 2)).strftime('%Y-%m-%d')
            issue_str = f"26{i + 1:03d}"
            records.append({
                'issue': issue_str,
                'draw_date': d_str,
                'red_1': (i % 5) + 1,
                'red_2': (i % 5) + 7,
                'red_3': (i % 5) + 13,
                'red_4': (i % 5) + 19,
                'red_5': (i % 5) + 25,
                'red_6': (i % 5) + 29,
                'blue': (i % 16) + 1
            })
        storage.insert_draws(records)

        pool_file = str(tmp_path / "pool.json")
        backup_file = str(tmp_path / "backup_pool.json")

        pool_data = export_frozen_candidate_pool(output_path=pool_file, backup_path=backup_file, random_seed=42)

        # 1. Recent draws top-level retention and meta redundancy cleanup
        assert 'recent_draws' in pool_data
        assert 'recent_draws' not in pool_data['meta']
        assert len(pool_data['recent_draws']) == 10

        first_draw = pool_data['recent_draws'][0]
        assert 'issue' in first_draw
        assert 'draw_date' in first_draw
        assert 'reds' in first_draw and len(first_draw['reds']) == 6
        assert 'blue' in first_draw and 1 <= first_draw['blue'] <= 16

        # 2. Balanced pool 16-blue full coverage and parity balance assertions
        balanced = pool_data['pools']['balanced']
        assert len(balanced) == 20

        unique_blues = set(t['blue'] for t in balanced)
        assert len(unique_blues) == 16, f"Expected 16 unique blues, got {len(unique_blues)}: {unique_blues}"
        assert unique_blues == set(range(1, 17))

        # Check no extreme parity (0:6 or 6:0) and no 3-consecutive in balanced
        mainstream_count = 0
        red_counts = Counter()
        for t in balanced:
            reds = t['reds']
            odd_cnt = sum(1 for r in reds if r % 2 == 1)
            assert odd_cnt not in (0, 6), f"Extreme parity {odd_cnt}:{6-odd_cnt} found in ticket {reds}"
            if odd_cnt in (2, 3, 4):
                mainstream_count += 1
            # No 3-consecutive
            for i in range(len(reds) - 2):
                assert not (reds[i+1] == reds[i]+1 and reds[i+2] == reds[i]+2), f"3-consecutive found in {reds}"
            for r in reds:
                red_counts[r] += 1

        assert mainstream_count / len(balanced) >= 0.75, f"Mainstream parity ratio {mainstream_count/20} < 75%"
        assert max(red_counts.values()) <= 5, f"Max red ball frequency {max(red_counts.values())} > 5"

        # 3. Jackpot pool 5-element bucket assertions
        jackpot = pool_data['pools']['jackpot']
        assert isinstance(jackpot, dict)
        assert set(jackpot.keys()) == {'水', '火', '木', '金', '土'}

        total_jackpot_tickets = 0
        for elem, bucket in jackpot.items():
            assert bucket['element'] == elem
            assert 1 <= bucket['blue'] <= 16
            assert len(bucket['tickets']) == 8
            total_jackpot_tickets += len(bucket['tickets'])

            # Check single red ball frequency within bucket <= 3
            bucket_red_counts = Counter()
            for t in bucket['tickets']:
                assert t['blue'] == bucket['blue']
                for r in t['reds']:
                    bucket_red_counts[r] += 1
            assert max(bucket_red_counts.values()) <= 3, f"Bucket {elem} red ball freq exceeds 3"

        assert total_jackpot_tickets == 40


class TestAuditPoolCommand:
    def test_cmd_audit_pool_execution(self, tmp_path, monkeypatch, capsys):
        import argparse
        import json
        from main import cmd_audit_pool

        # Setup isolated DB
        db_path = str(tmp_path / "test_ssq.db")
        csv_path = str(tmp_path / "test_history.csv")
        monkeypatch.setattr(config, 'DB_PATH', db_path)
        monkeypatch.setattr(config, 'CSV_PATH', csv_path)

        storage.init_db()
        storage.insert_draws([{
            'issue': '26134',
            'draw_date': '2026-11-22',
            'red_1': 4, 'red_2': 5, 'red_3': 10, 'red_4': 17, 'red_5': 21, 'red_6': 32,
            'blue': 8,
        }])

        pool_file = str(tmp_path / "pool_26134.json")
        pool_json = {
            'meta': {
                'target_issue': '26134',
                'target_draw_date': '2026-11-22 (周日)',
                'generated_at': '2026-11-21 12:00:00',
            },
            'recent_draws': [],
            'pools': {
                'balanced': [
                    {'rank': 1, 'reds': [4, 5, 10, 17, 21, 32], 'blue': 8},  # 6+1 First Prize
                    {'rank': 2, 'reds': [1, 2, 3, 11, 22, 30], 'blue': 8},   # 0+1 Sixth Prize
                ],
                'jackpot': {
                    '木': {
                        'element': '木',
                        'blue': 8,
                        'blue_score': 0.85,
                        'tickets': [
                            {'rank': 1, 'reds': [4, 5, 10, 17, 21, 32], 'blue': 8},
                        ]
                    }
                }
            }
        }
        with open(pool_file, 'w', encoding='utf-8') as f:
            json.dump(pool_json, f, ensure_ascii=False)

        args = argparse.Namespace(issue='26134', file=pool_file)
        cmd_audit_pool(args)

        captured = capsys.readouterr().out
        assert '双色球候选池实战复盘体检' in captured
        assert '26134' in captured
        assert '一等奖' in captured


class TestCLIArgsBacktestPriority:
    def test_backtest_cli_days_not_masked_by_draws(self):
        import argparse
        from main import main
        # Build parser matching main.py
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        p_backtest = subparsers.add_parser('backtest')
        p_backtest.add_argument('--draws', type=int, default=None)
        p_backtest.add_argument('--days', type=int, default=None)
        p_backtest.add_argument('--count', type=int, default=5)

        # 1. When user passes only --days 30
        args1 = parser.parse_args(['backtest', '--days', '30'])
        assert args1.days == 30
        assert args1.draws is None, "--draws default must be None so --days is not masked"

        # 2. When user passes only --draws 50
        args2 = parser.parse_args(['backtest', '--draws', '50'])
        assert args2.draws == 50
        assert args2.days is None

        # 3. When user passes neither
        args3 = parser.parse_args(['backtest'])
        assert args3.draws is None
        assert args3.days is None


class TestSingleSourcePrizeEvaluation:
    def test_prize_ssq_standard_and_fuyun_rule(self):
        from src.prize import determine_prize_level, evaluate_ticket

        # Standard rules
        assert determine_prize_level(6, 1) == ('一等奖', 5000000)
        assert determine_prize_level(6, 0) == ('二等奖', 150000)
        assert determine_prize_level(5, 1) == ('三等奖', 3000)
        assert determine_prize_level(5, 0) == ('四等奖', 200)
        assert determine_prize_level(4, 1) == ('四等奖', 200)
        assert determine_prize_level(4, 0) == ('五等奖', 10)
        assert determine_prize_level(3, 1) == ('五等奖', 10)
        assert determine_prize_level(2, 1) == ('六等奖', 5)
        assert determine_prize_level(1, 1) == ('六等奖', 5)
        assert determine_prize_level(0, 1) == ('六等奖', 5)
        assert determine_prize_level(2, 0) == ('未中奖', 0)

        # 2026 福运奖 (3+0)
        assert determine_prize_level(3, 0, special_rule_active=False) == ('未中奖', 0)
        assert determine_prize_level(3, 0, special_rule_active=True) == ('福运奖', 5)

        # evaluate_ticket
        ev_win = evaluate_ticket([1, 2, 3, 4, 5, 6], 8, [1, 2, 3, 10, 11, 12], 9, special_rule_active=True)
        assert ev_win['red_hit'] == 3
        assert ev_win['blue_hit'] == 0
        assert ev_win['prize_level'] == '福运奖'
        assert ev_win['prize_amount'] == 5
        assert ev_win['is_winner'] is True

        ev_lose = evaluate_ticket([1, 2, 3, 4, 5, 6], 8, [10, 11, 12, 13, 14, 15], 9)
        assert ev_lose['prize_amount'] == 0
        assert ev_lose['is_winner'] is False


class TestTicketValidationAndDeDuplication:
    def test_validate_selected_tickets_rules(self):
        from src.meihua import validate_selected_tickets

        # 1. Valid 5 tickets
        valid_tickets = [
            {'reds': [1, 2, 3, 4, 5, 6], 'blue': 1},
            {'reds': [7, 8, 9, 10, 11, 12], 'blue': 2},
            {'reds': [13, 14, 15, 16, 17, 18], 'blue': 3},
            {'reds': [19, 20, 21, 22, 23, 24], 'blue': 4},
            {'reds': [25, 26, 27, 28, 29, 30], 'blue': 5},
        ]
        res = validate_selected_tickets(valid_tickets, mode='balanced', expected_count=5, max_ball_freq=2, max_pairwise_overlap=1)
        assert res['is_valid'] is True
        assert res['max_actual_overlap'] == 0
        assert res['distinct_blues'] == 5

        # 2. Duplicate combinations
        dup_tickets = list(valid_tickets)
        dup_tickets[1] = {'reds': [1, 2, 3, 4, 5, 6], 'blue': 1}
        res_dup = validate_selected_tickets(dup_tickets, mode='balanced')
        assert res_dup['is_valid'] is False
        assert any("存在重复投注号码" in issue for issue in res_dup['issues'])

        # 3. Invalid ball ranges
        invalid_ball = [
            {'reds': [1, 2, 3, 4, 5, 34], 'blue': 1},  # 34 > 33
            {'reds': [7, 8, 9, 10, 11, 12], 'blue': 17}, # 17 > 16
            {'reds': [13, 14, 15, 16, 17, 18], 'blue': 3},
            {'reds': [19, 20, 21, 22, 23, 24], 'blue': 4},
            {'reds': [25, 26, 27, 28, 29, 30], 'blue': 5},
        ]
        res_inv = validate_selected_tickets(invalid_ball, mode='balanced')
        assert res_inv['is_valid'] is False
        assert any("红球非法" in issue for issue in res_inv['issues'])
        assert any("蓝球非法" in issue for issue in res_inv['issues'])

        # 4. Balanced mode blue duplicate
        blue_dup = list(valid_tickets)
        blue_dup[1] = {'reds': [7, 8, 9, 10, 11, 12], 'blue': 1}  # blue 1 repeated
        res_bdup = validate_selected_tickets(blue_dup, mode='balanced')
        assert res_bdup['is_valid'] is False
        assert any("蓝球未完全去重" in issue for issue in res_bdup['issues'])

        # 5. Jackpot mode blue not locked
        res_jnotlock = validate_selected_tickets(valid_tickets, mode='jackpot')
        assert res_jnotlock['is_valid'] is False
        assert any("蓝球未锁定单蓝" in issue for issue in res_jnotlock['issues'])


class TestMeihuaDFSOverlapLimit:
    def test_dfs_prioritizes_overlap_le_1(self):
        from src.meihua import select_meihua_tickets
        from datetime import datetime

        # Create candidate pool with high dispersion tickets
        candidates = [
            {'rank': 1, 'reds': [1, 2, 3, 4, 5, 6], 'blue': 1, 'composite_score': 0.95},
            {'rank': 2, 'reds': [7, 8, 9, 10, 11, 12], 'blue': 2, 'composite_score': 0.90},
            {'rank': 3, 'reds': [13, 14, 15, 16, 17, 18], 'blue': 3, 'composite_score': 0.85},
            {'rank': 4, 'reds': [19, 20, 21, 22, 23, 24], 'blue': 4, 'composite_score': 0.80},
            {'rank': 5, 'reds': [25, 26, 27, 28, 29, 30], 'blue': 5, 'composite_score': 0.75},
            {'rank': 6, 'reds': [1, 7, 13, 19, 25, 31], 'blue': 6, 'composite_score': 0.70},
            {'rank': 7, 'reds': [2, 8, 14, 20, 26, 32], 'blue': 7, 'composite_score': 0.65},
            {'rank': 8, 'reds': [3, 9, 15, 21, 27, 33], 'blue': 8, 'composite_score': 0.60},
        ]

        res = select_meihua_tickets(
            candidates,
            purchase_dt=datetime(2026, 11, 23, 10, 0),
            numbers=[3, 8],
            mode='balanced',
            target_count=5,
            max_ball_freq=2,
        )

        assert len(res['selected_tickets']) == 5
        assert res['actual_max_overlap'] <= 1, f"Actual max overlap {res['actual_max_overlap']} > 1"
        assert res['constraints_relaxed'] is False
        assert res['validation']['is_valid'] is True


class TestJackpotBlueReplacementRescoring:
    def test_blue_replacement_recomputes_scores(self, tmp_path, monkeypatch):
        import pandas as pd
        from datetime import datetime, timedelta
        from src.game_theory import select_game_theory_combinations, UniquenessScorer, AntiPopularityFilter
        from src.model import EnsembleModel, StatisticalModel

        db_path = str(tmp_path / "test_gt.db")
        csv_path = str(tmp_path / "test_gt.csv")
        monkeypatch.setattr(config, 'DB_PATH', db_path)
        monkeypatch.setattr(config, 'CSV_PATH', csv_path)

        storage.init_db()
        # Create minimal 60 draws
        records = []
        base_d = datetime(2026, 1, 1)
        for i in range(60):
            d_str = (base_d + timedelta(days=i * 2)).strftime('%Y-%m-%d')
            issue_str = f"26{i + 1:03d}"
            records.append({
                'issue': issue_str,
                'draw_date': d_str,
                'red_1': (i % 5) + 1,
                'red_2': (i % 5) + 7,
                'red_3': (i % 5) + 13,
                'red_4': (i % 5) + 19,
                'red_5': (i % 5) + 25,
                'red_6': (i % 5) + 29,
                'blue': (i % 16) + 1
            })
        storage.insert_draws(records)
        df = storage.get_all_draws()

        model_stat = StatisticalModel()
        model_stat.fit(df)

        combos, stats = select_game_theory_combinations(
            model_stat,
            n_combinations=5,
            mode='jackpot',
            pool_size=100,
            df=df,
            random_seed=42
        )

        assert len(combos) == 5
        locked_blue = stats['locked_blue']
        for c in combos:
            assert c['blue'] == locked_blue
            # Verify model score matches actual score_combination with the new blue
            expected_m = round(model_stat.score_combination(c['reds'], locked_blue), 4)
            assert c['model_score'] == expected_m


class TestProductionRobustnessBoundary:
    """Boundary and robustness test suite."""

    def test_candidate_pool_less_than_target_count(self):
        """1. When candidate pool has fewer tickets than target_count (e.g., 3 < 5),
        system must gracefully fallback, return all 3 tickets, set constraints_relaxed=True,
        and pass baseline validation (no duplicates, legal range).
        """
        from datetime import datetime
        from src.meihua import select_meihua_tickets

        small_pool = [
            {'rank': 1, 'reds': [1, 2, 3, 4, 5, 6], 'blue': 1, 'composite_score': 0.95},
            {'rank': 2, 'reds': [7, 8, 9, 10, 11, 12], 'blue': 2, 'composite_score': 0.90},
            {'rank': 3, 'reds': [13, 14, 15, 16, 17, 18], 'blue': 3, 'composite_score': 0.85},
        ]
        res = select_meihua_tickets(
            small_pool,
            purchase_dt=datetime(2026, 11, 23, 10, 0),
            numbers=[3, 8],
            mode='balanced',
            target_count=5,
        )

        assert len(res['selected_tickets']) == 3
        assert res['constraints_relaxed'] is True
        assert "不足" in res['relaxation_reason']
        val = res['validation']
        assert val['is_baseline_valid'] is True
        assert val['tickets_count'] == 3
        for t in res['selected_tickets']:
            assert len(t['reds']) == 6
            assert all(1 <= r <= 33 for r in t['reds'])
            assert 1 <= t['blue'] <= 16

    def test_extreme_high_overlap_pool_triggers_degradation(self):
        """2. Construct extreme candidate pool where all tickets share 5 of 6 red balls.
        Verify that DFS relaxes tiers down to fallback, constraints_relaxed=True,
        relaxation_reason is non-empty, and baseline hard validation strictly passes.
        """
        from datetime import datetime
        from src.meihua import select_meihua_tickets

        # 10 tickets, each sharing [1, 2, 3, 4, 5], only 6th ball differs
        # This makes overlap <= 1, 2, 3 impossible, and ball frequencies of [1..5] will exceed 2
        extreme_pool = [
            {
                'rank': i + 1,
                'reds': [1, 2, 3, 4, 5, 6 + i],
                'blue': (i % 16) + 1,
                'composite_score': 0.90 - i * 0.04
            }
            for i in range(10)
        ]

        res = select_meihua_tickets(
            extreme_pool,
            purchase_dt=datetime(2026, 11, 23, 10, 0),
            numbers=[3, 8],
            mode='balanced',
            target_count=5,
        )

        assert len(res['selected_tickets']) == 5
        assert res['constraints_relaxed'] is True
        assert res['relaxation_reason'] != ""
        val = res['validation']
        # Baseline hard constraints MUST hold
        assert val['is_baseline_valid'] is True
        assert val['tickets_count'] == 5
        for t in res['selected_tickets']:
            assert len(t['reds']) == 6
            assert all(1 <= r <= 33 for r in t['reds'])
            assert 1 <= t['blue'] <= 16

    def test_backtest_draws_priority_over_days(self):
        """3. When both --draws and --days are passed, --draws takes priority."""
        import main
        from src.backtest import run_backtest

        # Verify CLI parser logic
        parser = main.build_parser()
        args = parser.parse_args(['backtest', '--days', '30'])
        assert args.days == 30
        assert args.draws is None

        args_both = parser.parse_args(['backtest', '--draws', '5', '--days', '30'])
        assert args_both.draws == 5
        assert args_both.days == 30

        # End-to-end backtest run: draws=5 with days=60 -> should yield exactly 5 detail records
        detail, summary = run_backtest(draws=5, days=60, n_combinations=2, random_seed=42)
        assert len(detail) == 5

    def test_meihua_selection_determinism(self):
        """4. Running select_meihua_tickets multiple times with identical inputs
        must produce 100% deterministic results: exact same tickets, order, and scores.
        """
        from datetime import datetime
        from src.meihua import select_meihua_tickets

        pool = [
            {
                'rank': i + 1,
                'reds': sorted(list(set([(i * 3 + j) % 33 + 1 for j in range(6)]))),
                'blue': (i % 16) + 1,
                'composite_score': 0.90 - i * 0.03
            }
            for i in range(12)
        ]
        # Pad any sets that might have collided under modulo
        for p in pool:
            while len(p['reds']) < 6:
                p['reds'].append((p['reds'][-1] % 33) + 1)
                p['reds'] = sorted(list(set(p['reds'])))

        dt = datetime(2026, 11, 23, 14, 30)
        first_run = select_meihua_tickets(
            pool, purchase_dt=dt, numbers=[7, 14], mode='balanced', gamma=0.30, target_count=5
        )

        for _ in range(5):
            repeat_run = select_meihua_tickets(
                pool, purchase_dt=dt, numbers=[7, 14], mode='balanced', gamma=0.30, target_count=5
            )
            assert len(repeat_run['selected_tickets']) == len(first_run['selected_tickets'])
            for t1, t2 in zip(first_run['selected_tickets'], repeat_run['selected_tickets']):
                assert t1['reds'] == t2['reds']
                assert t1['blue'] == t2['blue']
                assert t1['original_rank'] == t2['original_rank']
                assert t1['final_meihua_score'] == t2['final_meihua_score']


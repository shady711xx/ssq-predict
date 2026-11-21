"""Unit tests for productionization features:
1. Candidate pool Min-Max normalization
2. max_ball_freq <= 2 cross-ticket constraint
3. 3-consecutive numbers heavy penalty
4. Jackpot mode composite blue ball locking
5. Automatic prediction check & verification state backfill
"""
import os
import sqlite3
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

        pool_data = export_frozen_candidate_pool(output_path=pool_file, backup_path=backup_file)

        assert 'recent_draws' in pool_data
        assert 'recent_draws' in pool_data['meta']
        assert len(pool_data['recent_draws']) == 10

        first_draw = pool_data['recent_draws'][0]
        assert 'issue' in first_draw
        assert 'draw_date' in first_draw
        assert 'reds' in first_draw and len(first_draw['reds']) == 6
        assert 'blue' in first_draw and 1 <= first_draw['blue'] <= 16

        assert len(pool_data['pools']['balanced']) == 20
        assert len(pool_data['pools']['jackpot']) == 20

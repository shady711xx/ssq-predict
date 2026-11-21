"""Tests to verify zero data leakage in walk-forward evaluation."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src import storage


def make_sequential_df(n=50):
    base_date = datetime(2023, 1, 1)
    rows = []
    current = base_date
    issue_seq = 1
    while len(rows) < n:
        if current.weekday() in (1, 3, 6):
            issue = f"23{issue_seq:03d}"
            rows.append({
                'issue': issue,
                'draw_date': current.strftime('%Y-%m-%d'),
                'red_1': 1, 'red_2': 2, 'red_3': 3, 'red_4': 4, 'red_5': 5, 'red_6': 6,
                'blue': 1
            })
            issue_seq += 1
        current += timedelta(days=1)
    return pd.DataFrame(rows)


class TestNoDataLeakage:
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        test_db = tmp_path / "test_ssq.db"
        test_csv = tmp_path / "test_ssq.csv"
        import config
        monkeypatch.setattr(config, 'DB_PATH', str(test_db))
        monkeypatch.setattr(config, 'CSV_PATH', str(test_csv))
        storage.init_db()
        df = make_sequential_df(30)
        storage.insert_draws(df.to_dict('records'))

    def test_draws_before_excludes_target(self):
        target = '23015'
        history = storage.get_draws_before(target)
        assert target not in history['issue'].values
        assert all(history['issue'] < target)

    def test_target_not_in_training(self):
        target = '23020'
        history = storage.get_draws_before(target)
        assert len(history) == 19
        assert target not in history['issue'].values

    def test_future_data_not_in_training(self):
        target = '23010'
        history = storage.get_draws_before(target)
        future_issues = ['23011', '23012', '23025']
        for fi in future_issues:
            assert fi not in history['issue'].values

    def test_walk_forward_sequential(self):
        all_draws = storage.get_all_draws()
        for idx in range(10, len(all_draws)):
            target = all_draws.iloc[idx]['issue']
            history = storage.get_draws_before(target)
            assert len(history) == idx
            assert all(history['issue'] < target)

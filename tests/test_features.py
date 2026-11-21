"""Unit tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np

from src import features


def make_test_df():
    data = [
        {'issue': '24001', 'draw_date': '2024-01-02', 'red_1': 1, 'red_2': 5, 'red_3': 10, 'red_4': 15, 'red_5': 20, 'red_6': 25, 'blue': 8},
        {'issue': '24002', 'draw_date': '2024-01-04', 'red_1': 2, 'red_2': 5, 'red_3': 12, 'red_4': 18, 'red_5': 22, 'red_6': 30, 'blue': 16},
        {'issue': '24003', 'draw_date': '2024-01-07', 'red_1': 1, 'red_2': 6, 'red_3': 11, 'red_4': 16, 'red_5': 21, 'red_6': 31, 'blue': 8},
        {'issue': '24004', 'draw_date': '2024-01-09', 'red_1': 3, 'red_2': 7, 'red_3': 13, 'red_4': 19, 'red_5': 23, 'red_6': 32, 'blue': 1},
        {'issue': '24005', 'draw_date': '2024-01-11', 'red_1': 4, 'red_2': 8, 'red_3': 14, 'red_4': 20, 'red_5': 24, 'red_6': 33, 'blue': 9},
    ]
    return pd.DataFrame(data)


class TestFrequency:
    def test_red_frequency(self):
        df = make_test_df()
        freq = features.compute_frequency(df, 'red')
        assert freq[1] == 2
        assert freq[5] == 2
        assert freq[2] == 1
        assert freq[9] == 0

    def test_blue_frequency(self):
        df = make_test_df()
        freq = features.compute_frequency(df, 'blue')
        assert freq[8] == 2
        assert freq[16] == 1
        assert freq[2] == 0

    def test_frequency_with_window(self):
        df = make_test_df()
        freq = features.compute_frequency(df, 'red', window=2)
        assert freq[1] == 0  # 1 was in draws 1 and 3, window=2 has only draws 4 and 5
        assert freq[4] == 1


class TestMissingValues:
    def test_red_missing(self):
        df = make_test_df()
        missing = features.compute_missing_values(df, 'red')
        assert missing[4] == 0   # appeared in draw 5
        assert missing[1] == 2   # appeared in draw 3, draws 4 and 5 missed -> 2
        assert missing[9] == 5   # never appeared in 5 draws -> 5

    def test_blue_missing(self):
        df = make_test_df()
        missing = features.compute_missing_values(df, 'blue')
        assert missing[9] == 0   # in draw 5
        assert missing[8] == 2   # in draw 3
        assert missing[2] == 5   # never


class TestMaxMissing:
    def test_max_missing_basic(self):
        df = make_test_df()
        max_m = features.compute_max_missing(df, 'red')
        assert isinstance(max_m[1], int)
        assert max_m[9] == 5


class TestOddEvenRatio:
    def test_basic(self):
        odd, even = features.compute_odd_even_ratio([1, 3, 5, 7, 9, 11])
        assert odd == 6 and even == 0

    def test_mixed(self):
        odd, even = features.compute_odd_even_ratio([1, 2, 3, 4, 5, 6])
        assert odd == 3 and even == 3


class TestSumValue:
    def test_basic(self):
        assert features.compute_sum_value([1, 2, 3, 4, 5, 6]) == 21


class TestConsecutiveCount:
    def test_no_consecutive(self):
        assert features.compute_consecutive_count([1, 3, 5, 7, 9, 11]) == 0

    def test_consecutive(self):
        assert features.compute_consecutive_count([1, 2, 3, 10, 11, 20]) == 3  # (1,2), (2,3), (10,11)


class TestACValue:
    def test_basic(self):
        # [1, 2, 3, 4, 5, 6]: diffs: 1, 2, 3, 4, 5 -> 5 distinct diffs. AC = 5 - 5 = 0
        ac = features.compute_ac_value([1, 2, 3, 4, 5, 6])
        assert ac == 0

    def test_high_ac(self):
        # [1, 5, 12, 18, 27, 33]
        ac = features.compute_ac_value([1, 5, 12, 18, 27, 33])
        assert ac > 5


class TestDrawFeatures:
    def test_basic(self):
        df = make_test_df()
        draw_feats = features.compute_draw_features(df)
        assert len(draw_feats) == len(df)
        assert 'sum_value' in draw_feats.columns
        assert 'ac_value' in draw_feats.columns

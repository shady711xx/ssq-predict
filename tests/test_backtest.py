"""Unit tests for backtesting and prize level determination."""
import pytest
import numpy as np
from src import backtest


class TestPrizeLevel:
    def test_first_prize(self):
        desc, prize = backtest.determine_prize_level(6, 1)
        assert desc == '一等奖' and prize == 5000000

    def test_second_prize(self):
        desc, prize = backtest.determine_prize_level(6, 0)
        assert desc == '二等奖' and prize == 150000

    def test_third_prize(self):
        desc, prize = backtest.determine_prize_level(5, 1)
        assert desc == '三等奖' and prize == 3000

    def test_fourth_prize_5_0(self):
        desc, prize = backtest.determine_prize_level(5, 0)
        assert desc == '四等奖' and prize == 200

    def test_fourth_prize_4_1(self):
        desc, prize = backtest.determine_prize_level(4, 1)
        assert desc == '四等奖' and prize == 200

    def test_fifth_prize_4_0(self):
        desc, prize = backtest.determine_prize_level(4, 0)
        assert desc == '五等奖' and prize == 10

    def test_fifth_prize_3_1(self):
        desc, prize = backtest.determine_prize_level(3, 1)
        assert desc == '五等奖' and prize == 10

    def test_sixth_prize_2_1(self):
        desc, prize = backtest.determine_prize_level(2, 1)
        assert desc == '六等奖' and prize == 5

    def test_sixth_prize_1_1(self):
        desc, prize = backtest.determine_prize_level(1, 1)
        assert desc == '六等奖' and prize == 5

    def test_sixth_prize_0_1(self):
        desc, prize = backtest.determine_prize_level(0, 1)
        assert desc == '六等奖' and prize == 5

    def test_no_prize(self):
        desc, prize = backtest.determine_prize_level(3, 0)
        assert desc == '未中奖' and prize == 0

    def test_no_prize_0_0(self):
        desc, prize = backtest.determine_prize_level(0, 0)
        assert desc == '未中奖' and prize == 0


class TestEvaluateSingleDraw:
    def test_full_match(self):
        res = backtest.evaluate_single_draw([1, 2, 3, 4, 5, 6], 8, [1, 2, 3, 4, 5, 6], 8)
        assert res['prize_level'] == '一等奖'
        assert res['prize_amount'] == 5000000
        assert res['red_hit'] == 6
        assert res['blue_hit'] == 1

    def test_partial_match(self):
        res = backtest.evaluate_single_draw([1, 2, 3, 4, 5, 6], 8, [1, 2, 3, 7, 8, 9], 8)
        assert res['prize_level'] == '五等奖'
        assert res['prize_amount'] == 10

    def test_no_match(self):
        res = backtest.evaluate_single_draw([1, 2, 3, 4, 5, 6], 8, [7, 8, 9, 10, 11, 12], 16)
        assert res['prize_level'] == '未中奖'
        assert res['prize_amount'] == 0

    def test_special_fuyun_rule_active(self):
        desc, prize = backtest.determine_prize_level(3, 0, special_rule=True)
        assert desc == '福运奖' and prize == 5

    def test_special_fuyun_rule_inactive(self):
        desc, prize = backtest.determine_prize_level(3, 0, special_rule=False)
        assert desc == '未中奖' and prize == 0


class TestCalibrationMetrics:
    def test_calibration_computation(self):
        p_red = np.full(33, 6.0 / 33.0)
        y_red = np.zeros(33)
        y_red[:6] = 1.0
        
        # Marginal loss
        loss = backtest.compute_multi_label_log_loss([y_red], [p_red])
        assert 0.45 < loss < 0.50

    def test_red_marginal_probability_scale(self):
        # Scale test: sum of marginal probabilities is approximately 6
        p_red_norm = np.full(33, 1.0 / 33.0)
        p_marginal = np.clip(p_red_norm * 6.0, 1e-6, 1.0 - 1e-6)
        assert np.isclose(p_marginal.sum(), 6.0)


class TestMonteCarloBaseline:
    def test_mc_simulation_structure_and_scale(self):
        # 10 draws, 5 tickets
        actuals = [([1, 2, 3, 4, 5, 6], 8) for _ in range(10)]
        res = backtest.run_monte_carlo_baseline(actuals, n_simulations=10, n_combinations=5, random_seed=42)
        assert 'distinct_blue' in res
        assert 'pure_random' in res
        assert 'median' in res['distinct_blue']
        assert 'p25' in res['distinct_blue']
        assert 'p75' in res['distinct_blue']

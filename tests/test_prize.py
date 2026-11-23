"""Unit tests for single authoritative prize evaluation engine (src/prize.py)."""

from datetime import date, datetime
import pytest
from src.prize import determine_prize_level, evaluate_ticket, is_special_rule_date


class TestStandardPrizeLevels:
    """Test all 1st through 6th prize levels and non-winning conditions."""

    def test_first_prize_6_1(self):
        desc, prize = determine_prize_level(6, 1)
        assert desc == '一等奖'
        assert prize == 5000000

    def test_second_prize_6_0(self):
        desc, prize = determine_prize_level(6, 0)
        assert desc == '二等奖'
        assert prize == 150000

    def test_third_prize_5_1(self):
        desc, prize = determine_prize_level(5, 1)
        assert desc == '三等奖'
        assert prize == 3000

    def test_fourth_prize_5_0_and_4_1(self):
        assert determine_prize_level(5, 0) == ('四等奖', 200)
        assert determine_prize_level(4, 1) == ('四等奖', 200)

    def test_fifth_prize_4_0_and_3_1(self):
        assert determine_prize_level(4, 0) == ('五等奖', 10)
        assert determine_prize_level(3, 1) == ('五等奖', 10)

    def test_sixth_prize_2_1_and_1_1_and_0_1(self):
        assert determine_prize_level(2, 1) == ('六等奖', 5)
        assert determine_prize_level(1, 1) == ('六等奖', 5)
        assert determine_prize_level(0, 1) == ('六等奖', 5)

    def test_non_winning_cases(self):
        assert determine_prize_level(0, 0) == ('未中奖', 0)
        assert determine_prize_level(1, 0) == ('未中奖', 0)
        assert determine_prize_level(2, 0) == ('未中奖', 0)


class TestAutomaticDateBasedSpecialRule:
    """Test automatic switching of 2026 福运奖 (3+0 = 5元) by draw_date."""

    def test_date_detection_utility(self):
        assert is_special_rule_date('2026-01-01') is True
        assert is_special_rule_date('2026-11-22 21:15:00') is True
        assert is_special_rule_date(date(2026, 5, 20)) is True
        assert is_special_rule_date(datetime(2026, 11, 23, 12, 0)) is True

        assert is_special_rule_date('2025-12-31') is False
        assert is_special_rule_date('2024-01-01') is False
        assert is_special_rule_date(date(2025, 12, 31)) is False
        assert is_special_rule_date(None) is False

    def test_3_0_in_2026_automatically_wins_fuyun(self):
        # In 2026, 3+0 should automatically award 福运奖 (5 CNY)
        desc, prize = determine_prize_level(3, 0, draw_date='2026-01-01')
        assert desc == '福运奖'
        assert prize == 5

        desc2, prize2 = determine_prize_level(3, 0, draw_date='2026-11-22')
        assert desc2 == '福运奖'
        assert prize2 == 5

    def test_3_0_before_2026_automatically_unwon(self):
        # Before 2026, 3+0 was not a prize
        desc, prize = determine_prize_level(3, 0, draw_date='2025-12-31')
        assert desc == '未中奖'
        assert prize == 0

        desc2, prize2 = determine_prize_level(3, 0, draw_date='2024-10-15')
        assert desc2 == '未中奖'
        assert prize2 == 0

    def test_explicit_override_takes_precedence(self):
        # Even before 2026, explicit special_rule_active=True activates it
        assert determine_prize_level(3, 0, draw_date='2024-01-01', special_rule_active=True) == ('福运奖', 5)
        # Even in 2026, explicit special_rule_active=False disables it
        assert determine_prize_level(3, 0, draw_date='2026-05-01', special_rule_active=False) == ('未中奖', 0)
        # Check special_rule alias
        assert determine_prize_level(3, 0, draw_date='2024-01-01', special_rule=True) == ('福运奖', 5)


class TestEvaluateTicket:
    """Test evaluate_ticket output format and correctness."""

    def test_evaluate_ticket_winner(self):
        res = evaluate_ticket(
            pred_reds=[1, 2, 3, 4, 5, 6],
            pred_blue=8,
            actual_reds=[1, 2, 3, 10, 11, 12],
            actual_blue=8,
            draw_date='2026-02-15'
        )
        assert res['red_hit'] == 3
        assert res['blue_hit'] == 1
        assert res['prize_level'] == '五等奖'
        assert res['prize_amount'] == 10
        assert res['is_winner'] is True

    def test_evaluate_ticket_fuyun_by_date(self):
        res_2026 = evaluate_ticket(
            pred_reds=[1, 2, 3, 4, 5, 6],
            pred_blue=8,
            actual_reds=[1, 2, 3, 10, 11, 12],
            actual_blue=9,
            draw_date='2026-02-15'
        )
        assert res_2026['red_hit'] == 3
        assert res_2026['blue_hit'] == 0
        assert res_2026['prize_level'] == '福运奖'
        assert res_2026['prize_amount'] == 5
        assert res_2026['is_winner'] is True

        res_2025 = evaluate_ticket(
            pred_reds=[1, 2, 3, 4, 5, 6],
            pred_blue=8,
            actual_reds=[1, 2, 3, 10, 11, 12],
            actual_blue=9,
            draw_date='2025-11-15'
        )
        assert res_2025['red_hit'] == 3
        assert res_2025['blue_hit'] == 0
        assert res_2025['prize_level'] == '未中奖'
        assert res_2025['prize_amount'] == 0
        assert res_2025['is_winner'] is False

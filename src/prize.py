"""Single authoritative prize evaluation engine for China Union Lotto (双色球).

Unifies prize determination across:
- Backtesting (src/backtest.py)
- Production verification & storage (src/storage.py)
- Candidate pool audit (main.py cmd_audit_pool)
"""

from typing import Any, Dict, Iterable, Optional, Tuple
import config


def determine_prize_level(
    red_hit: int,
    blue_hit: int,
    special_rule_active: bool = False,
    special_rule: Optional[bool] = None,
) -> Tuple[str, int]:
    """Determine prize level and amount based on official SSQ rules and optional 2026 promotion.
    
    Args:
        red_hit: number of red balls matched (0-6)
        blue_hit: number of blue balls matched (0 or 1)
        special_rule_active: whether 2026 福运奖 (3+0 = 5元) is active
        special_rule: alias for special_rule_active
    
    Returns:
        tuple: (prize_level_desc, prize_amount_cny)
    """
    if special_rule is not None:
        special_rule_active = special_rule

    # 2026 new promotion rule: 福运奖 (3 red + 0 blue = 5 CNY)
    if special_rule_active and (red_hit, blue_hit) == (3, 0):
        return (config.SPECIAL_FUYUN_PRIZE['desc'], config.SPECIAL_FUYUN_PRIZE['prize'])

    rule = config.PRIZE_RULES.get((red_hit, blue_hit))
    if rule is not None:
        return (rule['desc'], rule['prize'])

    return ('未中奖', 0)


def evaluate_ticket(
    pred_reds: Iterable[int],
    pred_blue: int,
    actual_reds: Iterable[int],
    actual_blue: int,
    special_rule_active: bool = False,
    special_rule: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate a single ticket combination against actual winning numbers.
    
    Args:
        pred_reds: predicted 6 red balls
        pred_blue: predicted blue ball
        actual_reds: actual winning 6 red balls
        actual_blue: actual winning blue ball
        special_rule_active: whether 2026 福运奖 (3+0 = 5元) is active
        special_rule: alias for special_rule_active
        
    Returns:
        dict: {
            'red_hit': int,
            'blue_hit': int,
            'prize_level': str,
            'prize_amount': int,
            'is_winner': bool,
        }
    """
    if special_rule is not None:
        special_rule_active = special_rule

    red_hit = len(set(pred_reds) & set(actual_reds))
    blue_hit = 1 if int(pred_blue) == int(actual_blue) else 0
    prize_level, prize_amount = determine_prize_level(
        red_hit, blue_hit, special_rule_active=special_rule_active
    )

    return {
        'red_hit': red_hit,
        'blue_hit': blue_hit,
        'prize_level': prize_level,
        'prize_amount': prize_amount,
        'is_winner': prize_amount > 0,
    }

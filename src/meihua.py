"""Meihua Yishu (梅花易数) Engine for SSQ Lottery.

Pure mathematical and Yi-Jing divination engine based on:
1. Xiantian Bagua & 3-bit line encoding
2. Purchasing time (hour -> 12 earthly branches) & user numbers
3. Hexagram derivation: Main, Mutual, and Transformed hexagrams
4. Ti-Yong (体用) dynamic identification & Wuxing interaction
5. Hetu (河图) ending digits mapping to Wuxing
6. Non-destructive final audit ranking for Top-20 candidate pool
"""
import math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# 1. Xiantian Bagua definitions (1-indexed)
# 3-bit binary: (first_line, second_line, third_line) from bottom to top (Yang=1, Yin=0)
BAGUA_CONFIG: Dict[int, Dict[str, Any]] = {
    1: {'name': '乾', 'symbol': '☰', 'element': '金', 'bits': (1, 1, 1)},
    2: {'name': '兑', 'symbol': '☱', 'element': '金', 'bits': (1, 1, 0)},
    3: {'name': '离', 'symbol': '☲', 'element': '火', 'bits': (1, 0, 1)},
    4: {'name': '震', 'symbol': '☳', 'element': '木', 'bits': (1, 0, 0)},
    5: {'name': '巽', 'symbol': '☴', 'element': '木', 'bits': (0, 1, 1)},
    6: {'name': '坎', 'symbol': '☵', 'element': '水', 'bits': (0, 1, 0)},
    7: {'name': '艮', 'symbol': '☶', 'element': '土', 'bits': (0, 0, 1)},
    8: {'name': '坤', 'symbol': '☷', 'element': '土', 'bits': (0, 0, 0)},
}

# Reverse lookup: bits tuple -> bagua number
BITS_TO_BAGUA = {v['bits']: k for k, v in BAGUA_CONFIG.items()}

# Earthly branches for hours
# 子(23-1)=1, 丑(1-3)=2, 寅(3-5)=3, 卯(5-7)=4, 辰(7-9)=5, 巳(9-11)=6,
# 午(11-13)=7, 未(13-15)=8, 申(15-17)=9, 酉(17-19)=10, 戌(19-21)=11, 亥(21-23)=12
SHICHEN_NAMES = [
    '子时', '丑时', '寅时', '卯时', '辰时', '巳时',
    '午时', '未时', '申时', '酉时', '戌时', '亥时'
]

# Wuxing generation and counter relationships
WUXING_GENERATES = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
WUXING_COUNTERS = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}

# Hetu last-digit to Wuxing mapping
# 水: 1, 6; 火: 2, 7; 木: 3, 8; 金: 4, 9; 土: 5, 0
DIGIT_TO_WUXING = {
    1: '水', 6: '水',
    2: '火', 7: '火',
    3: '木', 8: '木',
    4: '金', 9: '金',
    5: '土', 0: '土',
}


def get_shichen_num(dt: datetime) -> int:
    """Map datetime hour to 12 earthly branch number (1-12)."""
    hour = dt.hour
    if hour >= 23 or hour < 1:
        return 1  # 子
    return (hour + 1) // 2 + 1


def get_shichen_name(dt: datetime) -> str:
    """Return Chinese earthly branch name for datetime."""
    idx = get_shichen_num(dt) - 1
    return SHICHEN_NAMES[idx]


def get_ball_wuxing(ball_number: int) -> str:
    """Map SSQ ball number (1-33) to Hetu Wuxing element by its last digit."""
    last_digit = int(ball_number) % 10
    return DIGIT_TO_WUXING[last_digit]


def parse_user_numbers(numbers: List[int], shichen_num: int) -> Tuple[int, int, int]:
    """Parse user numbers into (upper_gua_num, lower_gua_num, moving_line).
    
    Rules:
    - 2 numbers [n1, n2]:
        upper = n1 % 8 or 8
        lower = n2 % 8 or 8
        moving = (n1 + n2 + shichen) % 6 or 6
    - 1 number [n1]:
        if n1 >= 10: split string in middle into n1, n2, then apply 2-number rule
        else: upper = n1 % 8 or 8, lower = shichen % 8 or 8, moving = (n1 + shichen) % 6 or 6
    - 3+ numbers:
        upper = n1 % 8 or 8
        lower = n2 % 8 or 8
        moving = (sum(numbers) + shichen) % 6 or 6
    """
    if not numbers:
        numbers = [shichen_num, shichen_num + 3]

    if len(numbers) == 1:
        n = numbers[0]
        s_num = str(n)
        if len(s_num) >= 2:
            mid = len(s_num) // 2
            n1 = int(s_num[:mid])
            n2 = int(s_num[mid:])
            upper = (n1 % 8) or 8
            lower = (n2 % 8) or 8
            moving = ((n1 + n2 + shichen_num) % 6) or 6
        else:
            upper = (n % 8) or 8
            lower = (shichen_num % 8) or 8
            moving = ((n + shichen_num) % 6) or 6
    elif len(numbers) == 2:
        n1, n2 = numbers[0], numbers[1]
        upper = (n1 % 8) or 8
        lower = (n2 % 8) or 8
        moving = ((n1 + n2 + shichen_num) % 6) or 6
    else:
        n1, n2 = numbers[0], numbers[1]
        upper = (n1 % 8) or 8
        lower = (n2 % 8) or 8
        moving = ((sum(numbers) + shichen_num) % 6) or 6

    return upper, lower, moving


def calculate_hexagrams(upper_num: int, lower_num: int, moving_line: int) -> Dict[str, Any]:
    """Calculate Main (主卦), Mutual (互卦), Transformed (变卦) and Ti-Yong division."""
    upper_bits = BAGUA_CONFIG[upper_num]['bits']
    lower_bits = BAGUA_CONFIG[lower_num]['bits']

    # 6 lines from bottom (idx 0) to top (idx 5)
    main_lines = lower_bits + upper_bits

    # Mutual hexagram (互卦):
    # Lower mutual = lines 2, 3, 4 (idx 1, 2, 3)
    # Upper mutual = lines 3, 4, 5 (idx 2, 3, 4)
    lower_mutual_bits = main_lines[1:4]
    upper_mutual_bits = main_lines[2:5]
    lower_mutual_num = BITS_TO_BAGUA[lower_mutual_bits]
    upper_mutual_num = BITS_TO_BAGUA[upper_mutual_bits]

    # Transformed hexagram (变卦): invert moving line (1-indexed)
    trans_lines = list(main_lines)
    trans_lines[moving_line - 1] = 1 - trans_lines[moving_line - 1]
    trans_lower_bits = tuple(trans_lines[0:3])
    trans_upper_bits = tuple(trans_lines[3:6])
    trans_lower_num = BITS_TO_BAGUA[trans_lower_bits]
    trans_upper_num = BITS_TO_BAGUA[trans_upper_bits]

    # Ti-Yong identification:
    # moving in 1, 2, 3 -> upper is Ti, lower is Yong
    # moving in 4, 5, 6 -> lower is Ti, upper is Yong
    if moving_line in (1, 2, 3):
        ti_position = '上卦'
        ti_num = upper_num
        yong_position = '下卦'
        yong_num = lower_num
        trans_yong_num = trans_lower_num
    else:
        ti_position = '下卦'
        ti_num = lower_num
        yong_position = '上卦'
        yong_num = upper_num
        trans_yong_num = trans_upper_num

    ti_element = BAGUA_CONFIG[ti_num]['element']
    yong_element = BAGUA_CONFIG[yong_num]['element']
    trans_yong_element = BAGUA_CONFIG[trans_yong_num]['element']

    return {
        'moving_line': moving_line,
        'main': {
            'name': f"{BAGUA_CONFIG[upper_num]['name']}{BAGUA_CONFIG[lower_num]['name']}",
            'upper': BAGUA_CONFIG[upper_num],
            'lower': BAGUA_CONFIG[lower_num],
            'upper_num': upper_num,
            'lower_num': lower_num,
            'lines': main_lines,
        },
        'mutual': {
            'name': f"{BAGUA_CONFIG[upper_mutual_num]['name']}{BAGUA_CONFIG[lower_mutual_num]['name']}",
            'upper': BAGUA_CONFIG[upper_mutual_num],
            'lower': BAGUA_CONFIG[lower_mutual_num],
            'upper_num': upper_mutual_num,
            'lower_num': lower_mutual_num,
        },
        'transformed': {
            'name': f"{BAGUA_CONFIG[trans_upper_num]['name']}{BAGUA_CONFIG[trans_lower_num]['name']}",
            'upper': BAGUA_CONFIG[trans_upper_num],
            'lower': BAGUA_CONFIG[trans_lower_num],
            'upper_num': trans_upper_num,
            'lower_num': trans_lower_num,
            'lines': tuple(trans_lines),
        },
        'ti': {
            'position': ti_position,
            'num': ti_num,
            'name': BAGUA_CONFIG[ti_num]['name'],
            'element': ti_element,
            'symbol': BAGUA_CONFIG[ti_num]['symbol'],
        },
        'yong': {
            'position': yong_position,
            'num': yong_num,
            'name': BAGUA_CONFIG[yong_num]['name'],
            'element': yong_element,
            'symbol': BAGUA_CONFIG[yong_num]['symbol'],
        },
        'trans_yong': {
            'num': trans_yong_num,
            'name': BAGUA_CONFIG[trans_yong_num]['name'],
            'element': trans_yong_element,
        }
    }


def analyze_wuxing_relation(source: str, target: str) -> str:
    """Analyze relation between source and target element.
    Returns: 'same' (比和), 'generate' (生体), 'counter' (克体), 'leak' (体生用/泄气), 'control' (体克用/财星)
    """
    if source == target:
        return 'same'
    if WUXING_GENERATES[source] == target:
        return 'generate'
    if WUXING_COUNTERS[source] == target:
        return 'counter'
    if WUXING_GENERATES[target] == source:
        return 'leak'
    if WUXING_COUNTERS[target] == source:
        return 'control'
    return 'neutral'


def evaluate_divination_fortune(hex_info: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overall divination rating, guidance, and Wuxing weights."""
    ti_elem = hex_info['ti']['element']
    yong_elem = hex_info['yong']['element']
    trans_yong_elem = hex_info['trans_yong']['element']
    mut_upper_elem = hex_info['mutual']['upper']['element']
    mut_lower_elem = hex_info['mutual']['lower']['element']

    # 1. Relations
    main_rel = analyze_wuxing_relation(yong_elem, ti_elem)
    trans_rel = analyze_wuxing_relation(trans_yong_elem, ti_elem)
    mut_upper_rel = analyze_wuxing_relation(mut_upper_elem, ti_elem)
    mut_lower_rel = analyze_wuxing_relation(mut_lower_elem, ti_elem)

    # 2. Comprehensive Fortune assessment (Evaluating Main, Mutuals, and Transformed)
    # 变卦为终局（权重最高）：若变用克体，大忌；若变用生体，吉兆
    if trans_rel == 'counter':
        rating = '偏克 · 慎出'
        color = '#ef4444'
        if main_rel == 'counter':
            desc = '主用相克且变卦回头交克，互卦亦无生扶，当刻气机偏逆，大受制约，建议减注观望或换个时辰定夺。'
        elif main_rel == 'leak':
            desc = '初卦体生用耗散元气，变卦回头见克，互卦克泄交加，当刻气机偏逆，建议减注观望或换个时辰定夺。'
        elif main_rel in ('generate', 'same'):
            desc = '初卦虽见生扶比和，但变卦回头克体，先吉后凶，后劲受阻，建议谨慎观望。'
        else:
            desc = '终局见变用回头克体，破耗受制，气机受阻，建议保守观望。'
    elif trans_rel == 'leak':
        if main_rel in ('counter', 'leak'):
            rating = '泄气 · 宜守'
            desc = '体卦生用泄耗元气，全盘气机虚耗，难聚财星，不宜重注，宜守不宜急。'
            color = '#8b5cf6'
        elif main_rel in ('generate', 'same'):
            rating = '平卦'
            desc = '初有生扶而终见耗泄，吉凶平缓，互见消长，按需小额娱乐。'
            color = '#6b7280'
        else:
            rating = '平卦'
            desc = '体虽克用求财但终局耗损，得失相抵，平缓定夺。'
            color = '#6b7280'
    elif trans_rel in ('generate', 'same'):
        if main_rel in ('generate', 'same'):
            rating = '大吉'
            desc = '主用相生比和，变卦回头生旺体卦，生扶得力，后劲充沛，可按计划果断出手。'
            color = '#10b981'
        elif main_rel in ('counter', 'leak'):
            rating = '中吉 · 转吉'
            desc = '初卦虽有克泄波折，但变卦生扶体卦，转危为安，后劲充足，可顺势小额跟进。'
            color = '#3b82f6'
        else:
            rating = '大吉'
            desc = '体克用求财得地，变卦更回头生旺体卦，财印两全，利于果断出击。'
            color = '#10b981'
    elif trans_rel == 'control':
        if main_rel in ('generate', 'same', 'control'):
            rating = '中吉 · 财星得地'
            desc = '体克用为财星，求财可得，利于主动出击，顺势跟进。'
            color = '#f59e0b'
        elif main_rel == 'counter':
            rating = '平卦'
            desc = '初受克制而终局克用求财，劳心费力，吉凶互见，小额参与。'
            color = '#6b7280'
        else:
            rating = '平卦'
            desc = '初虽泄气但终能克用得财，耗力求进，平和对待。'
            color = '#6b7280'
    else:
        rating = '平卦'
        desc = '吉凶互见，体用平缓无冲克，平和定夺，按需小额娱乐。'
        color = '#6b7280'

    # 互卦全克泄降档保护
    if rating == '平卦':
        all_harm = all(r in ('counter', 'leak') for r in (main_rel, mut_upper_rel, mut_lower_rel, trans_rel))
        if all_harm:
            rating = '泄气 · 宜守'
            desc = '主卦、互卦与变卦克泄交加，全无生扶，气机消退，宜守不宜急。'
            color = '#8b5cf6'

    # 3. Wuxing weight allocation
    # Baseline for 5 elements based on Ti element
    weights: Dict[str, float] = {}
    for elem in ['金', '木', '水', '火', '土']:
        rel = analyze_wuxing_relation(elem, ti_elem)
        if rel == 'generate':
            w = 1.0  # 印/生扶
        elif rel == 'same':
            w = 0.8  # 比和
        elif rel == 'control':
            w = 0.4  # 财星
        elif rel == 'leak':
            w = -0.5 # 泄气
        elif rel == 'counter':
            w = -1.0 # 忌神
        else:
            w = 0.0

        # Fine-tune with transformed and mutual tendencies
        if elem == trans_yong_elem and trans_rel in ('generate', 'same'):
            w += 0.1
        if elem in (mut_upper_elem, mut_lower_elem) and analyze_wuxing_relation(elem, ti_elem) in ('generate', 'same'):
            w += 0.05
        weights[elem] = round(w, 2)

    favored_elements = [e for e, w in sorted(weights.items(), key=lambda x: x[1], reverse=True) if w > 0]
    avoided_elements = [e for e, w in sorted(weights.items(), key=lambda x: x[1]) if w < 0]

    return {
        'rating': rating,
        'description': desc,
        'color': color,
        'main_relation': main_rel,
        'trans_relation': trans_rel,
        'wuxing_weights': weights,
        'favored_elements': favored_elements,
        'avoided_elements': avoided_elements,
    }


def score_ticket_meihua(reds: List[int], blue: int, wuxing_weights: Dict[str, float]) -> float:
    """Calculate raw Meihua Wuxing affinity score for a ticket.
    
    Formula:
    mean(red_wuxing_weights) + 1.5 * blue_wuxing_weight
    """
    red_weights = [wuxing_weights.get(get_ball_wuxing(r), 0.0) for r in reds]
    red_mean = sum(red_weights) / len(red_weights) if red_weights else 0.0
    blue_w = wuxing_weights.get(get_ball_wuxing(blue), 0.0)
    raw_score = red_mean + 1.5 * blue_w
    return float(raw_score)


def validate_selected_tickets(
    tickets: List[Dict[str, Any]],
    mode: str = 'balanced',
    expected_count: int = 5,
    max_ball_freq: int = 2,
    max_pairwise_overlap: int = 1,
) -> Dict[str, Any]:
    """Strictly validate selected tickets against hard constraints.
    
    Checks:
    1. Ticket count == expected_count
    2. Combination uniqueness (no duplicate tickets)
    3. Legal ball ranges: reds in [1, 33] distinct 6 balls, blue in [1, 16]
    4. Ball frequency across tickets: each red ball freq <= max_ball_freq
    5. Blue ball constraints:
       - 'balanced': all blue balls must be distinct
       - 'jackpot': all blue balls must be identical (locked single blue)
    6. Pairwise red ball overlap: for all pairs (i, j), len(reds_i & reds_j) <= max_pairwise_overlap
    
    Returns:
        dict: {
            'is_valid': bool,
            'issues': List[str],
            'max_actual_freq': int,
            'max_actual_overlap': int,
            'distinct_blues': int,
            'tickets_count': int,
        }
    """
    issues: List[str] = []
    
    # 1. Count
    if len(tickets) != expected_count:
        issues.append(f"注数不符: 期望 {expected_count} 注，实际 {len(tickets)} 注")

    # 2. Duplicate combinations
    seen_combos = set()
    for i, t in enumerate(tickets):
        red_tuple = tuple(sorted(t['reds']))
        combo_key = (red_tuple, int(t['blue']))
        if combo_key in seen_combos:
            issues.append(f"存在重复投注号码: 第 {i+1} 注 ({combo_key})")
        seen_combos.add(combo_key)

    # 3. Legal ranges
    for i, t in enumerate(tickets):
        reds = t['reds']
        if len(reds) != 6 or len(set(reds)) != 6 or any(not (1 <= int(r) <= 33) for r in reds):
            issues.append(f"第 {i+1} 注红球非法: {reds}")
        if not (1 <= int(t['blue']) <= 16):
            issues.append(f"第 {i+1} 注蓝球非法: {t['blue']}")

    # 4. Red ball frequency
    red_counts = Counter()
    for t in tickets:
        for r in t['reds']:
            red_counts[int(r)] += 1
    max_actual_freq = max(red_counts.values()) if red_counts else 0
    if max_actual_freq > max_ball_freq:
        over_balls = [f"{b}({c}次)" for b, c in red_counts.items() if c > max_ball_freq]
        issues.append(f"单红球频次超标 (上限 {max_ball_freq} 次): {', '.join(over_balls)}")

    # 5. Blue ball mode constraints
    blues = [int(t['blue']) for t in tickets]
    distinct_blues = len(set(blues))
    if mode == 'balanced':
        if distinct_blues != len(tickets):
            issues.append(f"稳健保底模式下蓝球未完全去重: 出现 {distinct_blues} 个独立蓝球 (期望 {len(tickets)} 个)")
    elif mode == 'jackpot':
        if distinct_blues > 1:
            issues.append(f"冲大奖模式下蓝球未锁定单蓝: 出现 {distinct_blues} 个不同蓝球")

    # 6. Pairwise red overlap
    max_actual_overlap = 0
    for i in range(len(tickets)):
        for j in range(i + 1, len(tickets)):
            ov = len(set(tickets[i]['reds']) & set(tickets[j]['reds']))
            if ov > max_actual_overlap:
                max_actual_overlap = ov
    if max_actual_overlap > max_pairwise_overlap:
        issues.append(f"两两红球重叠超标 (上限 {max_pairwise_overlap} 个): 实际最大重叠为 {max_actual_overlap} 个")

    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'max_actual_freq': max_actual_freq,
        'max_actual_overlap': max_actual_overlap,
        'distinct_blues': distinct_blues,
        'tickets_count': len(tickets),
    }


def select_meihua_tickets(
    candidate_pool: Any,
    purchase_dt: Optional[datetime] = None,
    numbers: Optional[List[int]] = None,
    mode: str = 'balanced',
    gamma: float = 0.30,
    target_count: int = 5,
    max_ball_freq: int = 2,
) -> Dict[str, Any]:
    """Perform non-destructive Meihua final audit on candidate pool.
    
    CRITICAL INVARIANT:
    Tickets in candidate_pool are NEVER modified (no balls changed, swapped or replaced).
    Only re-ranked and selected via constrained greedy pass.
    
    Modes:
    - balanced: candidate_pool is a list of 20 tickets. Enforces distinct blues and max_ball_freq <= 2.
    - jackpot: candidate_pool is a dict of 5 Wuxing buckets (or list). Dynamically locks 1 best bucket
               based on 'blue_base_score + meihua_wuxing_score', then selects 5 tickets from its 8 tickets.
    
    Returns:
        dict containing hexagram_info, fortune, selected_tickets, locked_element, locked_blue, etc.
    """
    if purchase_dt is None:
        purchase_dt = datetime.now()
    if not numbers:
        numbers = [3, 8]

    shichen_num = get_shichen_num(purchase_dt)
    shichen_name = get_shichen_name(purchase_dt)
    upper_num, lower_num, moving_line = parse_user_numbers(numbers, shichen_num)

    hex_info = calculate_hexagrams(upper_num, lower_num, moving_line)
    hex_info['shichen_name'] = shichen_name
    hex_info['purchase_time'] = purchase_dt.strftime('%Y-%m-%d %H:%M')

    fortune = evaluate_divination_fortune(hex_info)
    wuxing_weights = fortune['wuxing_weights']

    locked_element: Optional[str] = None
    locked_blue: Optional[int] = None
    target_pool: List[Dict[str, Any]] = []

    # Handle jackpot 5-element bucket dynamic locking
    if mode == 'jackpot':
        if isinstance(candidate_pool, dict):
            # candidate_pool has 5 element buckets: {"水": {...}, "火": {...}, ...}
            buckets = list(candidate_pool.values())
            b_scores = [float(b.get('blue_score', 0.5)) for b in buckets]
            m_scores = [float(wuxing_weights.get(b.get('element', ''), 0.0)) for b in buckets]

            min_b, max_b = min(b_scores), max(b_scores)
            min_m, max_m = min(m_scores), max(m_scores)
            denom_b = (max_b - min_b) + 1e-9
            denom_m = (max_m - min_m) + 1e-9

            scored_buckets = []
            for b in buckets:
                elem = b.get('element', '')
                norm_b = (float(b.get('blue_score', 0.5)) - min_b) / denom_b
                norm_m = (float(wuxing_weights.get(elem, 0.0)) - min_m) / denom_m
                bucket_composite = (1.0 - gamma) * norm_b + gamma * norm_m
                scored_buckets.append((bucket_composite, b))

            # Sort buckets by composite score descending
            scored_buckets.sort(key=lambda x: x[0], reverse=True)
            best_bucket = scored_buckets[0][1]
            locked_element = best_bucket.get('element')
            locked_blue = int(best_bucket.get('blue'))
            target_pool = best_bucket.get('tickets', [])
        elif isinstance(candidate_pool, list) and candidate_pool:
            # Fallback for flat list: check if multiple blues exist
            blues_in_pool = list(set(t['blue'] for t in candidate_pool))
            if len(blues_in_pool) == 1:
                locked_blue = blues_in_pool[0]
                locked_element = get_ball_wuxing(locked_blue)
                target_pool = candidate_pool
            else:
                # Group by wuxing element
                elem_groups: Dict[str, List[Dict[str, Any]]] = {}
                for t in candidate_pool:
                    elem = get_ball_wuxing(t['blue'])
                    elem_groups.setdefault(elem, []).append(t)

                best_elem = max(elem_groups.keys(), key=lambda e: wuxing_weights.get(e, 0.0))
                locked_element = best_elem
                target_pool = elem_groups[best_elem]
                locked_blue = target_pool[0]['blue']
        else:
            target_pool = []
    else:
        # Balanced mode: flat list of 20 tickets
        if isinstance(candidate_pool, dict):
            target_pool = candidate_pool.get('balanced', [])
        else:
            target_pool = candidate_pool

    # Score each ticket in target pool
    scored_pool: List[Dict[str, Any]] = []
    raw_meihua_scores = []
    base_scores = []

    for item in target_pool:
        raw_m = score_ticket_meihua(item['reds'], item['blue'], wuxing_weights)
        raw_meihua_scores.append(raw_m)
        base_s = float(item.get('composite_score', item.get('final_score', item.get('model_score', 0.5))))
        base_scores.append(base_s)

    if raw_meihua_scores:
        min_m, max_m = min(raw_meihua_scores), max(raw_meihua_scores)
        min_b, max_b = min(base_scores), max(base_scores)
        denom_m = (max_m - min_m) + 1e-9
        denom_b = (max_b - min_b) + 1e-9

        for idx, item in enumerate(target_pool):
            norm_meihua = (raw_meihua_scores[idx] - min_m) / denom_m
            norm_base = (base_scores[idx] - min_b) / denom_b
            final_score = (1.0 - gamma) * norm_base + gamma * norm_meihua

            favored_set = set(fortune['favored_elements'])
            matched_reds = [r for r in item['reds'] if get_ball_wuxing(r) in favored_set]
            matched_blue = get_ball_wuxing(item['blue']) in favored_set

            ticket_copy = dict(item)
            ticket_copy['original_rank'] = item.get('rank', idx + 1)
            ticket_copy['meihua_score'] = round(float(norm_meihua), 4)
            ticket_copy['norm_composite'] = round(float(norm_base), 4)
            ticket_copy['final_meihua_score'] = round(float(final_score), 4)
            ticket_copy['matched_favored_reds'] = matched_reds
            ticket_copy['matched_favored_blue'] = matched_blue
            ticket_copy['red_wuxing'] = [get_ball_wuxing(r) for r in item['reds']]
            ticket_copy['blue_wuxing'] = get_ball_wuxing(item['blue'])
            scored_pool.append(ticket_copy)

    # Re-rank strictly by final_meihua_score descending
    scored_pool.sort(key=lambda x: x['final_meihua_score'], reverse=True)

    # Constrained optimal selection using pruned DFS across tickets
    selected: List[Dict[str, Any]] = []
    constraints_relaxed = False
    relaxation_reason = ""

    # Tiered constraint ladder:
    # Prioritizes pairwise red overlap <= 1 and freq <= max_ball_freq (ideal mutually exclusive)
    # Then steps down overlap (2 -> 3 -> 6), preserving freq <= max_ball_freq as long as possible
    tiers = [
        (1, max_ball_freq),
        (2, max_ball_freq),
        (3, max_ball_freq),
        (6, max_ball_freq),
        (2, max_ball_freq + 1),
        (3, max_ball_freq + 1),
        (6, max_ball_freq + 1),
        (6, target_count),
    ]

    for allowed_ov, allowed_freq in tiers:
        best_combo: Optional[List[Dict[str, Any]]] = None
        best_sum: float = -1e9

        def dfs(idx: int, current: List[Dict[str, Any]], used_blues: set, ball_counts: Dict[int, int], curr_sum: float):
            nonlocal best_combo, best_sum
            if len(current) == target_count:
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_combo = list(current)
                return
            if idx >= len(scored_pool):
                return
            # Pruning branch
            rem = target_count - len(current)
            if curr_sum + rem * scored_pool[idx]['final_meihua_score'] <= best_sum:
                return

            t = scored_pool[idx]
            can_add = True
            if mode == 'balanced' and t['blue'] in used_blues:
                can_add = False
            if can_add and any(ball_counts.get(r, 0) >= allowed_freq for r in t['reds']):
                can_add = False
            if can_add:
                t_reds_set = set(t['reds'])
                if any(len(t_reds_set & set(prev['reds'])) > allowed_ov for prev in current):
                    can_add = False

            if can_add:
                if mode == 'balanced':
                    used_blues.add(t['blue'])
                for r in t['reds']:
                    ball_counts[r] = ball_counts.get(r, 0) + 1
                current.append(t)

                dfs(idx + 1, current, used_blues, ball_counts, curr_sum + t['final_meihua_score'])

                current.pop()
                for r in t['reds']:
                    ball_counts[r] -= 1
                if mode == 'balanced':
                    used_blues.remove(t['blue'])

            # Branch without current ticket
            dfs(idx + 1, current, used_blues, ball_counts, curr_sum)

        dfs(0, [], set(), {}, 0.0)

        if best_combo and len(best_combo) == target_count:
            selected = best_combo
            if allowed_ov > 1 or allowed_freq > max_ball_freq:
                constraints_relaxed = True
                relaxation_reason = f"降级为: 两两重叠<={allowed_ov}, 单球频次<={allowed_freq}"
            break

    # Final fallback if pool is extremely small or heavily constrained
    if len(selected) < target_count:
        constraints_relaxed = True
        relaxation_reason = f"候选池可用注数不足，兜底填充至 {len(scored_pool)} 注"
        for t in scored_pool:
            if not any((t['reds'] == prev['reds'] and t['blue'] == prev['blue']) for prev in selected):
                selected.append(t)
                if len(selected) == target_count:
                    break

    # Actual max overlap
    actual_max_overlap = 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            ov = len(set(selected[i]['reds']) & set(selected[j]['reds']))
            if ov > actual_max_overlap:
                actual_max_overlap = ov

    validation = validate_selected_tickets(
        selected,
        mode=mode,
        expected_count=target_count,
        max_ball_freq=max_ball_freq,
        max_pairwise_overlap=1,
    )

    return {
        'purchase_time': purchase_dt.strftime('%Y-%m-%d %H:%M'),
        'shichen': f"{shichen_name} ({shichen_num})",
        'numbers': numbers,
        'hexagram_info': hex_info,
        'fortune': fortune,
        'mode': mode,
        'gamma': gamma,
        'selected_tickets': selected,
        'ranked_pool': scored_pool,
        'constraints_relaxed': constraints_relaxed,
        'relaxation_reason': relaxation_reason,
        'actual_max_overlap': actual_max_overlap,
        'validation': validation,
        'fallback_relaxed': constraints_relaxed,
        'locked_element': locked_element,
        'locked_blue': locked_blue,
    }

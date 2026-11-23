import pytest
from datetime import datetime
from collections import Counter
from src.meihua import (
    BAGUA_CONFIG,
    BITS_TO_BAGUA,
    DIGIT_TO_WUXING,
    get_shichen_num,
    get_shichen_name,
    get_ball_wuxing,
    parse_user_numbers,
    calculate_hexagrams,
    analyze_wuxing_relation,
    evaluate_divination_fortune,
    score_ticket_meihua,
    select_meihua_tickets,
)

def test_trigram_definitions():
    """测试先天八卦与 3-bit 爻象定义"""
    assert BAGUA_CONFIG[1]["name"] == "乾"
    assert BAGUA_CONFIG[1]["element"] == "金"
    assert BAGUA_CONFIG[1]["bits"] == (1, 1, 1)

    assert BAGUA_CONFIG[2]["name"] == "兑"
    assert BAGUA_CONFIG[2]["element"] == "金"
    assert BAGUA_CONFIG[2]["bits"] == (1, 1, 0)

    assert BAGUA_CONFIG[3]["name"] == "离"
    assert BAGUA_CONFIG[3]["element"] == "火"
    assert BAGUA_CONFIG[3]["bits"] == (1, 0, 1)

    assert BAGUA_CONFIG[4]["name"] == "震"
    assert BAGUA_CONFIG[4]["element"] == "木"
    assert BAGUA_CONFIG[4]["bits"] == (1, 0, 0)

    assert BAGUA_CONFIG[5]["name"] == "巽"
    assert BAGUA_CONFIG[5]["element"] == "木"
    assert BAGUA_CONFIG[5]["bits"] == (0, 1, 1)

    assert BAGUA_CONFIG[6]["name"] == "坎"
    assert BAGUA_CONFIG[6]["element"] == "水"
    assert BAGUA_CONFIG[6]["bits"] == (0, 1, 0)

    assert BAGUA_CONFIG[7]["name"] == "艮"
    assert BAGUA_CONFIG[7]["element"] == "土"
    assert BAGUA_CONFIG[7]["bits"] == (0, 0, 1)

    assert BAGUA_CONFIG[8]["name"] == "坤"
    assert BAGUA_CONFIG[8]["element"] == "土"
    assert BAGUA_CONFIG[8]["bits"] == (0, 0, 0)

    # 验证反向映射
    for num, conf in BAGUA_CONFIG.items():
        assert BITS_TO_BAGUA[conf["bits"]] == num

def test_shichen_mapping():
    """测试时辰边界换算与地支名称"""
    # 子时: 23:00 - 00:59
    assert get_shichen_num(datetime(2026, 3, 1, 23, 0)) == 1
    assert get_shichen_num(datetime(2026, 3, 1, 23, 45)) == 1
    assert get_shichen_num(datetime(2026, 3, 1, 0, 0)) == 1
    assert get_shichen_num(datetime(2026, 3, 1, 0, 59)) == 1
    assert get_shichen_name(datetime(2026, 3, 1, 23, 30)) == "子时"

    # 丑时: 01:00 - 02:59
    assert get_shichen_num(datetime(2026, 3, 1, 1, 0)) == 2
    assert get_shichen_num(datetime(2026, 3, 1, 2, 30)) == 2
    assert get_shichen_name(datetime(2026, 3, 1, 1, 30)) == "丑时"

    # 午时: 11:00 - 12:59
    assert get_shichen_num(datetime(2026, 3, 1, 11, 0)) == 7
    assert get_shichen_num(datetime(2026, 3, 1, 12, 15)) == 7
    assert get_shichen_name(datetime(2026, 3, 1, 12, 0)) == "午时"

    # 酉时: 17:00 - 18:59
    assert get_shichen_num(datetime(2026, 3, 1, 17, 30)) == 10
    assert get_shichen_num(datetime(2026, 3, 1, 18, 59)) == 10
    assert get_shichen_name(datetime(2026, 3, 1, 18, 0)) == "酉时"

    # 亥时: 21:00 - 22:59
    assert get_shichen_num(datetime(2026, 3, 1, 21, 0)) == 12
    assert get_shichen_num(datetime(2026, 3, 1, 22, 59)) == 12
    assert get_shichen_name(datetime(2026, 3, 1, 22, 0)) == "亥时"

def test_divination_numbers_parsing():
    """测试心动数字起卦逻辑"""
    # 1. 两个数字 [7, 18]，时辰 辰时 (shichen = 5)
    # upper = 7%8 = 7, lower = 18%8 = 2, moving = (7+18+5)%6 = 30%6 = 6
    up, low, mov = parse_user_numbers([7, 18], shichen_num=5)
    assert up == 7
    assert low == 2
    assert mov == 6

    # 2. 单个个位数 [9]，时辰 午时 (shichen = 7)
    # n=9, len<2 -> upper = 9%8 = 1, lower = 7%8 = 7, moving = (9+7)%6 = 4
    up, low, mov = parse_user_numbers([9], shichen_num=7)
    assert up == 1
    assert low == 7
    assert mov == 4

    # 3. 单个两位数 [38]，时辰 巳时 (shichen = 6)
    # len>=2 -> n1=3, n2=8 -> upper=3%8=3, lower=8%8=8, moving=(3+8+6)%6=17%6=5
    up, low, mov = parse_user_numbers([38], shichen_num=6)
    assert up == 3
    assert low == 8
    assert mov == 5

    # 4. 三个及以上数字 [3, 8, 15]，时辰 亥时 (shichen = 12)
    # upper = 3%8 = 3, lower = 8%8 = 8, moving = (3+8+15+12)%6 = 38%6 = 2
    up, low, mov = parse_user_numbers([3, 8, 15], shichen_num=12)
    assert up == 3
    assert low == 8
    assert mov == 2

def test_hexagram_calculation_ti_yong():
    """测试主卦、互卦、变卦与体用划分"""
    # 设 upper=1 (乾 111), lower=8 (坤 000), moving=4 (动在上卦初爻，即第4爻)
    # main: 000 111 (从下往上: 0,0,0, 1,1,1)
    # moving=4 -> 变第4爻: 1变为0 -> trans: 000 011 -> lower=000(坤8), upper=011(巽5)
    # 体用: moving=4 (在4,5,6) -> 下卦为体(坤), 上卦为用(乾)
    hex_info = calculate_hexagrams(upper_num=1, lower_num=8, moving_line=4)
    assert hex_info["ti"]["position"] == "下卦"
    assert hex_info["ti"]["name"] == "坤"
    assert hex_info["ti"]["element"] == "土"
    assert hex_info["yong"]["position"] == "上卦"
    assert hex_info["yong"]["name"] == "乾"
    assert hex_info["yong"]["element"] == "金"
    assert hex_info["transformed"]["upper_num"] == 5
    assert hex_info["transformed"]["lower_num"] == 8

    # 测试动在下卦: moving=2 (在1,2,3) -> 上卦为体, 下卦为用
    hex_info2 = calculate_hexagrams(upper_num=3, lower_num=7, moving_line=2)
    assert hex_info2["ti"]["position"] == "上卦"
    assert hex_info2["ti"]["name"] == "离"
    assert hex_info2["ti"]["element"] == "火"
    assert hex_info2["yong"]["position"] == "下卦"
    assert hex_info2["yong"]["name"] == "艮"
    assert hex_info2["yong"]["element"] == "土"

def test_hetu_wuxing_and_ball_mapping():
    """测试河图尾数五行映射与双色球球号五行"""
    # 水: 1, 6
    assert DIGIT_TO_WUXING[1] == "水"
    assert DIGIT_TO_WUXING[6] == "水"
    assert get_ball_wuxing(1) == "水"
    assert get_ball_wuxing(16) == "水"
    assert get_ball_wuxing(21) == "水"

    # 火: 2, 7
    assert DIGIT_TO_WUXING[2] == "火"
    assert DIGIT_TO_WUXING[7] == "火"
    assert get_ball_wuxing(7) == "火"
    assert get_ball_wuxing(12) == "火"
    assert get_ball_wuxing(27) == "火"

    # 木: 3, 8
    assert DIGIT_TO_WUXING[3] == "木"
    assert DIGIT_TO_WUXING[8] == "木"
    assert get_ball_wuxing(3) == "木"
    assert get_ball_wuxing(18) == "木"
    assert get_ball_wuxing(33) == "木"

    # 金: 4, 9
    assert DIGIT_TO_WUXING[4] == "金"
    assert DIGIT_TO_WUXING[9] == "金"
    assert get_ball_wuxing(4) == "金"
    assert get_ball_wuxing(19) == "金"
    assert get_ball_wuxing(29) == "金"

    # 土: 5, 0
    assert DIGIT_TO_WUXING[5] == "土"
    assert DIGIT_TO_WUXING[0] == "土"
    assert get_ball_wuxing(5) == "土"
    assert get_ball_wuxing(10) == "土"
    assert get_ball_wuxing(30) == "土"

def test_audit_strictly_preserves_numbers():
    """
    铁律验证：起卦终审严禁篡改原注号码
    """
    sample_pool = [
        {"id": 1, "reds": [1, 8, 12, 19, 25, 32], "blue": 6, "composite_score": 0.8},
        {"id": 2, "reds": [3, 11, 16, 22, 28, 33], "blue": 9, "composite_score": 0.75},
        {"id": 3, "reds": [5, 10, 15, 20, 26, 31], "blue": 12, "composite_score": 0.7},
    ]

    dt = datetime(2026, 3, 1, 10, 0)
    res = select_meihua_tickets(sample_pool, purchase_dt=dt, numbers=[6, 16], target_count=3)
    ranked = res["ranked_pool"]

    assert len(ranked) == 3
    orig_by_id = {item["id"]: item for item in sample_pool}
    for item in ranked:
        orig = orig_by_id[item["id"]]
        # 严格验证原注红球数组和蓝球完全未被修改
        assert item["reds"] == orig["reds"]
        assert item["blue"] == orig["blue"]
        assert "meihua_score" in item
        assert "final_meihua_score" in item
        assert 0.0 <= item["meihua_score"] <= 1.0

def test_select_meihua_tickets_constraints():
    """
    测试选出的 5 注严格遵守蓝球互不相同和单红球频次 <= 2
    """
    candidates = []
    # 构造 20 注候选池
    for i in range(20):
        b_num = (i % 16) + 1
        # 让红球 1 在前几注出现
        if i < 4:
            reds = [1, 2, 11, 15, 22, 30]
        elif i < 8:
            reds = [1, 5, 12, 18, 25, 32]
        elif i < 14:
            reds = [3, 7, 14, 19, 27, 33]
        else:
            reds = [4, 8, 16, 20, 24, 31]

        candidates.append({
            "id": i + 1,
            "reds": reds,
            "blue": b_num,
            "composite_score": 0.9 - i * 0.02
        })

    dt = datetime(2026, 3, 1, 12, 0)
    res = select_meihua_tickets(
        candidate_pool=candidates,
        purchase_dt=dt,
        numbers=[8],
        mode="balanced",
        target_count=5,
        max_ball_freq=2
    )

    selected = res["selected_tickets"]
    assert len(selected) == 5

    # 1. 蓝球完全互不相同
    blues = [t["blue"] for t in selected]
    assert len(set(blues)) == 5

    # 2. 任意单个红球在 5 注中出现次数 <= 2
    all_reds = []
    for t in selected:
        all_reds.extend(t["reds"])
    red_counts = Counter(all_reds)
    for ball, freq in red_counts.items():
        assert freq <= 2, f"红球 {ball} 频次 {freq} 超过了上限 2"

def test_select_meihua_tickets_jackpot_mode():
    """
    测试 jackpot 模式下单蓝球锁定与单红球频次 <= 2
    """
    candidates = []
    # 构造 20 注同蓝球的候选池
    for i in range(20):
        if i < 4:
            reds = [1, 2, 11, 15, 22, 30]
        elif i < 8:
            reds = [1, 5, 12, 18, 25, 32]
        elif i < 14:
            reds = [3, 7, 14, 19, 27, 33]
        else:
            reds = [4, 8, 16, 20, 24, 31]

        candidates.append({
            "id": i + 1,
            "reds": reds,
            "blue": 15,  # 锁蓝 15
            "composite_score": 0.9 - i * 0.02
        })

    dt = datetime(2026, 3, 1, 12, 0)
    res = select_meihua_tickets(
        candidate_pool=candidates,
        purchase_dt=dt,
        numbers=[12, 27],
        mode="jackpot",
        target_count=5,
        max_ball_freq=2
    )

    selected = res["selected_tickets"]
    assert len(selected) == 5

    # 1. 全部注均为锁定的蓝球 15
    for t in selected:
        assert t["blue"] == 15

    # 2. 任意单个红球在 5 注中出现次数 <= 2
    all_reds = []
    for t in selected:
        all_reds.extend(t["reds"])
    red_counts = Counter(all_reds)
    for ball, freq in red_counts.items():
        assert freq <= 2, f"红球 {ball} 频次 {freq} 超过了上限 2"

def test_python_js_vector_consistency():
    """
    测试向量：固定输入下梅花排盘参数，用于保障 JS 引擎与 Python 引擎结果 100% 绝对一致
    """
    # 向量 1: 2026-03-01 18:00 (酉时: 10), 心动数字 [3, 9]
    # parse: upper = 3%8 = 3, lower = 9%8 = 1, moving = (3+9+10)%6 = 22%6 = 4
    # upper_num: 3 (离), lower_num: 1 (乾)
    # moving: 4 (在4,5,6) -> 下卦为体 (乾/金), 上卦为用 (离/火)
    # 主卦: 离乾 -> 火天大有
    up, low, mov = parse_user_numbers([3, 9], shichen_num=10)
    assert up == 3
    assert low == 1
    assert mov == 4

    hex_info = calculate_hexagrams(up, low, mov)
    assert hex_info["ti"]["position"] == "下卦"
    assert hex_info["ti"]["name"] == "乾"
    assert hex_info["ti"]["element"] == "金"
    assert hex_info["yong"]["position"] == "上卦"
    assert hex_info["yong"]["name"] == "离"
    assert hex_info["yong"]["element"] == "火"

    # 用生体 / 克体 / 比和分析
    # yong(火) vs ti(金): 火克金 -> counter
    rel = analyze_wuxing_relation("火", "金")
    assert rel == "counter"

    fortune = evaluate_divination_fortune(hex_info)
    assert fortune["main_relation"] == "counter"
    # 金为体，生体为土(1.0)，比和为金(0.8)，克体为火(-1.0)
    weights = fortune["wuxing_weights"]
    assert weights["土"] >= 1.0
    assert weights["金"] >= 0.8
    assert weights["火"] <= -0.9

def test_select_meihua_tickets_jackpot_5_buckets_dynamic_locking():
    """测试 5 五行蓝球桶在 jackpot 模式下的动态锁定与约束"""
    # 构造 5 个河图五行桶，每个桶 8 注
    buckets = {}
    elements = [
        ('水', 11, 0.60),
        ('火', 7, 0.70),
        ('木', 8, 0.50),
        ('金', 4, 0.65),
        ('土', 10, 0.55),
    ]
    for elem, b_val, b_score in elements:
        tickets = []
        for i in range(8):
            reds = [
                (i * 2 + 1) % 33 + 1,
                (i * 2 + 5) % 33 + 1,
                (i * 2 + 11) % 33 + 1,
                (i * 2 + 17) % 33 + 1,
                (i * 2 + 23) % 33 + 1,
                (i * 2 + 27) % 33 + 1,
            ]
            reds = sorted(list(set(reds)))
            while len(reds) < 6:
                reds.append((reds[-1] % 33) + 1)
                reds = sorted(list(set(reds)))
            tickets.append({
                'rank': i + 1,
                'reds': reds,
                'blue': b_val,
                'composite_score': 0.85 - i * 0.03,
            })
        buckets[elem] = {
            'element': elem,
            'blue': b_val,
            'blue_score': b_score,
            'tickets': tickets,
        }

    # 测试巳时 (10:00) 报数 [3, 8] -> 体土用火 (火生土为印星, 权重最高)
    dt = datetime(2026, 11, 23, 10, 0)
    res = select_meihua_tickets(
        candidate_pool=buckets,
        purchase_dt=dt,
        numbers=[3, 8],
        mode="jackpot",
        gamma=0.30,
        target_count=5,
        max_ball_freq=2,
    )

    assert res['locked_element'] in ('火', '土'), f"喜用为火/土，锁定了 {res['locked_element']}"
    assert res['locked_blue'] == buckets[res['locked_element']]['blue']
    assert len(res['selected_tickets']) == 5
    for t in res['selected_tickets']:
        assert t['blue'] == res['locked_blue']

    # 频次检查
    all_reds = [r for t in res['selected_tickets'] for r in t['reds']]
    counts = Counter(all_reds)
    assert max(counts.values()) <= 2


def test_regression_counter_leak_fortune():
    """
    回归测试：修复主用泄气且变用回头克误判为平卦的 Bug
    输入：numbers=[3, 8], time='2026-11-21 15:30' (申时: 9)
    排盘：主卦离坤 (上体离火/下用坤土 -> 泄气), 互卦坎艮, 变卦离坎 (变用坎水克离火 -> 变用克体)
    断言：总断绝不能为 '平卦'，必须判定为 '偏克 · 慎出'
    """
    dt = datetime(2026, 11, 21, 15, 30)
    shichen_num = get_shichen_num(dt)
    assert shichen_num == 9  # 申时

    up, low, mov = parse_user_numbers([3, 8], shichen_num)
    assert up == 3  # 离
    assert low == 8  # 坤
    assert mov == 2  # 第2爻动 -> 1..3 爻动上体下用

    hex_info = calculate_hexagrams(up, low, mov)
    assert hex_info["ti"]["element"] == "火"  # 离
    assert hex_info["yong"]["element"] == "土"  # 坤
    assert hex_info["trans_yong"]["element"] == "水"  # 坎

    fortune = evaluate_divination_fortune(hex_info)
    assert fortune["rating"] != "平卦", "严重Bug：全卦克泄交加且变用回头克，绝不可断为平卦！"
    assert fortune["rating"] == "偏克 · 慎出"
    assert fortune["color"] == "#ef4444"
    assert "气机偏逆" in fortune["description"]

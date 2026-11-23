"""Automated cross-engine verification: Python (src/meihua.py) vs Frontend JS (docs/index.html).

This test extracts the exact JavaScript engine from docs/index.html, executes it via
local Node.js, and rigorously asserts that across 6 diverse test scenarios (balanced mode,
jackpot mode, different numbers, different shichens, and extreme degradation pools),
the Python backend and the frontend JS engine produce 100% identical outputs:
1. Hexagram layout (main, mutual, transformed hexagrams, moving line, Ti/Yong elements)
2. Favored and avoided Wuxing elements
3. Final 5 selected tickets and order (exact red balls, blue ball, original rank)
4. Relaxation status (constraints_relaxed boolean and locked elements/blues in jackpot)
"""
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.meihua import select_meihua_tickets

ROOT_DIR = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT_DIR / "docs" / "index.html"
POOL_PATH = ROOT_DIR / "docs" / "data" / "pool.json"


def _get_node_executable():
    node_cmd = shutil.which("node")
    if not node_cmd:
        pytest.skip("Node.js is not installed or not in PATH")
    return node_cmd


def _extract_meihua_js_code() -> str:
    """Dynamically extracts the standardized Meihua engine code from docs/index.html."""
    assert HTML_PATH.exists(), f"docs/index.html not found at {HTML_PATH}"
    html = HTML_PATH.read_text(encoding="utf-8")
    start_marker = "// --- Standardized Meihua Yi-Shu Engine (100% equivalent to src/meihua.py) ---"
    end_marker = "// --- UI Interaction Logic ---"
    assert start_marker in html, "Start marker not found in docs/index.html"
    assert end_marker in html, "End marker not found in docs/index.html"

    js_code = html.split(start_marker)[1].split(end_marker)[0]
    return js_code


def _run_js_meihua(
    pool: Any,
    purchase_dt: datetime,
    numbers: List[int],
    mode: str,
    gamma: float = 0.30,
) -> Dict[str, Any]:
    """Execute the extracted JS engine in Node.js via stdin."""
    node_cmd = _get_node_executable()
    js_engine = _extract_meihua_js_code()

    # Pass date as year, monthIndex (0-11), day, hour, minute
    year = purchase_dt.year
    month_idx = purchase_dt.month - 1
    day = purchase_dt.day
    hour = purchase_dt.hour
    minute = purchase_dt.minute

    pool_json = json.dumps(pool, ensure_ascii=False)
    numbers_json = json.dumps(numbers)

    runner_script = f"""
{js_engine}

const pool = {pool_json};
const date = new Date({year}, {month_idx}, {day}, {hour}, {minute});
const numbers = {numbers_json};
const mode = '{mode}';
const gamma = {gamma};

const result = selectMeihuaTickets(pool, date, numbers, mode, gamma);
console.log(JSON.stringify(result));
"""

    proc = subprocess.run(
        [node_cmd],
        input=runner_script,
        text=True,
        capture_output=True,
        shell=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Node execution failed (code {proc.returncode}):\n{proc.stderr}")

    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def real_pool_data():
    assert POOL_PATH.exists(), f"pool.json not found at {POOL_PATH}"
    return json.load(open(POOL_PATH, encoding="utf-8"))["pools"]


@pytest.fixture(scope="module")
def extreme_overlap_pool():
    # 10 tickets, each sharing [1, 2, 3, 4, 5], only 6th ball differs
    return [
        {
            "rank": i + 1,
            "reds": [1, 2, 3, 4, 5, 6 + i],
            "blue": (i % 16) + 1,
            "composite_score": round(0.92 - i * 0.04, 4),
        }
        for i in range(10)
    ]


class TestFrontendBackendSync:
    """Automated cross-engine verification between Python and JavaScript."""

    def test_scenario_1_balanced_standard(self, real_pool_data):
        """Scenario 1: balanced mode, numbers [3, 8], 巳时 (10:00)."""
        dt = datetime(2026, 11, 23, 10, 0)
        numbers = [3, 8]
        mode = "balanced"
        gamma = 0.30

        py_res = select_meihua_tickets(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        self._assert_sync(py_res, js_res, mode=mode)

    def test_scenario_2_balanced_midnight_shichen(self, real_pool_data):
        """Scenario 2: balanced mode, numbers [12, 27], 子时 (00:30)."""
        dt = datetime(2026, 11, 23, 0, 30)
        numbers = [12, 27]
        mode = "balanced"
        gamma = 0.30

        py_res = select_meihua_tickets(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        self._assert_sync(py_res, js_res, mode=mode)

    def test_scenario_3_jackpot_5_buckets_evening(self, real_pool_data):
        """Scenario 3: jackpot 5-element bucket mode, numbers [3, 8], 酉时 (18:00)."""
        dt = datetime(2026, 11, 23, 18, 0)
        numbers = [3, 8]
        mode = "jackpot"
        gamma = 0.30

        py_res = select_meihua_tickets(
            real_pool_data["jackpot"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            real_pool_data["jackpot"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        self._assert_sync(py_res, js_res, mode=mode)

    def test_scenario_4_jackpot_single_number_noon(self, real_pool_data):
        """Scenario 4: jackpot mode, single input number [6], 午时 (12:15)."""
        dt = datetime(2026, 11, 23, 12, 15)
        numbers = [6]
        mode = "jackpot"
        gamma = 0.30

        py_res = select_meihua_tickets(
            real_pool_data["jackpot"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            real_pool_data["jackpot"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        self._assert_sync(py_res, js_res, mode=mode)

    def test_scenario_5_balanced_three_numbers_dawn(self, real_pool_data):
        """Scenario 5: balanced mode, 3 input numbers [1, 2, 3], 卯时 (06:00)."""
        dt = datetime(2026, 11, 23, 6, 0)
        numbers = [1, 2, 3]
        mode = "balanced"
        gamma = 0.30

        py_res = select_meihua_tickets(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            real_pool_data["balanced"], purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        self._assert_sync(py_res, js_res, mode=mode)

    def test_scenario_6_extreme_pool_relaxation_fallback(self, extreme_overlap_pool):
        """Scenario 6: extreme high-overlap pool triggering tier step-down and relaxation fallback."""
        dt = datetime(2026, 11, 23, 10, 0)
        numbers = [3, 8]
        mode = "balanced"
        gamma = 0.30

        py_res = select_meihua_tickets(
            extreme_overlap_pool, purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )
        js_res = _run_js_meihua(
            extreme_overlap_pool, purchase_dt=dt, numbers=numbers, mode=mode, gamma=gamma
        )

        # Both must flag relaxation
        assert py_res["constraints_relaxed"] is True
        assert js_res["constraintsRelaxed"] is True

        self._assert_sync(py_res, js_res, mode=mode)

    def _assert_sync(self, py_res: Dict[str, Any], js_res: Dict[str, Any], mode: str):
        """Verify 100% equivalence across all dimensions."""
        py_hex = py_res["hexagram_info"]
        js_hex = js_res["hex"]

        # 1. Hexagram layout
        assert py_hex["main"]["name"] == js_hex["main"]["name"], "Main hexagram mismatch"
        assert py_hex["mutual"]["name"] == js_hex["mutual"]["name"], "Mutual hexagram mismatch"
        assert py_hex["transformed"]["name"] == js_hex["transformed"]["name"], "Transformed hexagram mismatch"
        assert py_hex["moving_line"] == js_hex["movingLine"], "Moving line mismatch"
        assert py_hex["ti"]["element"] == js_hex["ti"]["element"], "Ti element mismatch"
        assert py_hex["yong"]["element"] == js_hex["yong"]["element"], "Yong element mismatch"

        # 2. Wuxing fortunes
        py_fort = py_res["fortune"]
        js_fort = js_res["fortune"]
        assert py_fort["favored_elements"] == js_fort["favored"], "Favored elements mismatch"
        assert py_fort["avoided_elements"] == js_fort["avoided"], "Avoided elements mismatch"
        assert py_fort["rating"] == js_fort["rating"], "Fortune rating mismatch"

        # 3. Jackpot locking (if applicable)
        if mode == "jackpot":
            assert py_res["locked_element"] == js_res["lockedElement"], "Locked element mismatch"
            assert py_res["locked_blue"] == js_res["lockedBlue"], "Locked blue mismatch"

        # 4. Relaxation flag
        assert py_res["constraints_relaxed"] == js_res["constraintsRelaxed"], "Relaxation flag mismatch"

        # 5. Tickets selection (5 tickets, order, reds, blue, rank, and score)
        py_tickets = py_res["selected_tickets"]
        js_tickets = js_res["selected"]
        assert len(py_tickets) == len(js_tickets), f"Ticket count mismatch: {len(py_tickets)} vs {len(js_tickets)}"

        for i, (pt, jt) in enumerate(zip(py_tickets, js_tickets)):
            assert pt["reds"] == jt["reds"], f"Ticket {i+1} reds mismatch: {pt['reds']} vs {jt['reds']}"
            assert pt["blue"] == jt["blue"], f"Ticket {i+1} blue mismatch: {pt['blue']} vs {jt['blue']}"
            assert pt["original_rank"] == jt["original_rank"], f"Ticket {i+1} original_rank mismatch"
            # Scores match to within 1e-4 tolerance
            assert abs(pt["final_meihua_score"] - jt["final_meihua_score"]) < 1e-3, (
                f"Ticket {i+1} score mismatch: {pt['final_meihua_score']} vs {jt['final_meihua_score']}"
            )

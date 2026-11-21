"""Mock data generator for SSQ lottery testing."""
import random
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import storage

logger = logging.getLogger('ssq.mock')


def generate_mock_data(years: int = 3, end_date: Any = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate mock historical SSQ data for unit testing."""
    random.seed(seed)
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=years * 365)
    records = []
    current = start_date
    issue_num = 1
    
    while current <= end_date:
        # SSQ draws on Tue(1), Thu(3), Sun(6)
        if current.weekday() in (1, 3, 6):
            issue = f"{current.year % 100:02d}{issue_num:03d}"
            reds = sorted(random.sample(range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1), config.RED_BALL_COUNT))
            blue = random.randint(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX)
            records.append({
                'issue': issue,
                'draw_date': current.strftime('%Y-%m-%d'),
                'red_1': reds[0], 'red_2': reds[1], 'red_3': reds[2],
                'red_4': reds[3], 'red_5': reds[4], 'red_6': reds[5],
                'blue': blue
            })
            issue_num += 1
            if current.month == 12 and current.day >= 28:
                issue_num = 1
        current += timedelta(days=1)
    return records


def load_mock_data(years: int = 3, seed: int = 42) -> int:
    """Generate and load mock data into SQLite database."""
    storage.init_db()
    records = generate_mock_data(years=years, seed=seed)
    return storage.insert_draws(records)

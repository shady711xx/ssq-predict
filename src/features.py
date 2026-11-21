"""Feature engineering for SSQ lottery prediction.

Computes statistical features from historical draw data including:
- Hot/cold number frequency
- Missing value (遗漏值) for each number
- Odd/even ratio
- Large/small ratio
- Sum value distribution
- Consecutive number rate (连号率)
- AC value
- Sliding window features
"""
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger('ssq.features')


def get_red_balls_from_row(row: pd.Series) -> List[int]:
    """Extract red ball list from a DataFrame row."""
    return [int(row[f'red_{i}']) for i in range(1, 7)]


def compute_frequency(
    df: pd.DataFrame, ball_type: str = 'red', window: Optional[int] = None
) -> Dict[int, int]:
    """Compute frequency of each number appearing in recent draws."""
    if window:
        df = df.tail(window)

    if df.empty:
        if ball_type == 'red':
            return {n: 0 for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1)}
        else:
            return {n: 0 for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1)}

    if ball_type == 'red':
        red_cols = [f'red_{i}' for i in range(1, 7)]
        red_vals = df[red_cols].to_numpy()
        freq = {}
        for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1):
            freq[n] = int((red_vals == n).any(axis=1).sum())
        return freq
    else:  # blue
        blue_vals = df['blue'].to_numpy()
        freq = {}
        for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1):
            freq[n] = int((blue_vals == n).sum())
        return freq


def compute_missing_values(df: pd.DataFrame, ball_type: str = 'red') -> Dict[int, int]:
    """Compute current missing value (当前遗漏值) for each number."""
    if df.empty:
        if ball_type == 'red':
            return {n: 0 for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1)}
        else:
            return {n: 0 for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1)}

    n_rows = len(df)
    if ball_type == 'red':
        red_cols = [f'red_{i}' for i in range(1, 7)]
        red_vals = df[red_cols].to_numpy()
        missing = {}
        for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1):
            matches = np.where((red_vals == n).any(axis=1))[0]
            if len(matches) > 0:
                missing[n] = int(n_rows - 1 - matches[-1])
            else:
                missing[n] = int(n_rows)
        return missing
    else:
        blue_vals = df['blue'].to_numpy()
        missing = {}
        for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1):
            matches = np.where(blue_vals == n)[0]
            if len(matches) > 0:
                missing[n] = int(n_rows - 1 - matches[-1])
            else:
                missing[n] = int(n_rows)
        return missing


def compute_max_missing(df: pd.DataFrame, ball_type: str = 'red') -> Dict[int, int]:
    """Compute historical maximum missing value for each ball."""
    if df.empty:
        if ball_type == 'red':
            return {n: 0 for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1)}
        else:
            return {n: 0 for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1)}

    n_rows = len(df)
    max_missing = {}
    if ball_type == 'red':
        red_cols = [f'red_{i}' for i in range(1, 7)]
        red_vals = df[red_cols].to_numpy()
        for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1):
            matches = np.where((red_vals == n).any(axis=1))[0]
            if len(matches) == 0:
                max_missing[n] = n_rows
            else:
                diffs = [matches[0]]
                for i in range(1, len(matches)):
                    diffs.append(matches[i] - matches[i-1] - 1)
                diffs.append(n_rows - 1 - matches[-1])
                max_missing[n] = int(max(diffs))
    else:
        blue_vals = df['blue'].to_numpy()
        for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1):
            matches = np.where(blue_vals == n)[0]
            if len(matches) == 0:
                max_missing[n] = n_rows
            else:
                diffs = [matches[0]]
                for i in range(1, len(matches)):
                    diffs.append(matches[i] - matches[i-1] - 1)
                diffs.append(n_rows - 1 - matches[-1])
                max_missing[n] = int(max(diffs))
    return max_missing


def compute_odd_even_ratio(numbers: List[int]) -> Tuple[int, int]:
    """Compute odd count and even count."""
    odd = sum(1 for x in numbers if x % 2 == 1)
    even = len(numbers) - odd
    return odd, even


def compute_big_small_ratio(numbers: List[int], threshold: int = 17) -> Tuple[int, int]:
    """Compute big (>threshold) and small (<=threshold) count."""
    big = sum(1 for x in numbers if x >= threshold)
    small = len(numbers) - big
    return big, small


def compute_sum_value(numbers: List[int]) -> int:
    """Compute sum value of balls."""
    return sum(numbers)


def compute_consecutive_count(numbers: List[int]) -> int:
    """Compute count of consecutive adjacent numbers."""
    s = sorted(numbers)
    count = 0
    for i in range(len(s) - 1):
        if s[i+1] - s[i] == 1:
            count += 1
    return count


def compute_ac_value(numbers: List[int]) -> int:
    """Compute AC value (complexity value) of red balls."""
    s = sorted(numbers)
    diffs = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            diffs.add(s[j] - s[i])
    return len(diffs) - (len(numbers) - 1)


def compute_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-draw aggregate features for all rows."""
    red_cols = [f'red_{i}' for i in range(1, 7)]
    red_vals = df[red_cols].to_numpy()
    
    sums = red_vals.sum(axis=1)
    odds = (red_vals % 2 == 1).sum(axis=1)
    bigs = (red_vals >= 17).sum(axis=1)
    spans = red_vals[:, -1] - red_vals[:, 0]
    
    consecutives = []
    ac_values = []
    for row in red_vals:
        consecutives.append(compute_consecutive_count(list(row)))
        ac_values.append(compute_ac_value(list(row)))
        
    return pd.DataFrame({
        'sum_value': sums,
        'odd_count': odds,
        'big_count': bigs,
        'span': spans,
        'consecutive_count': consecutives,
        'ac_value': ac_values
    })


def build_feature_matrix(
    df: pd.DataFrame, window_sizes: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build rolling feature matrix X and target labels y_red, y_blue."""
    if window_sizes is None:
        window_sizes = config.MODEL_WINDOW_SIZES

    max_window = max(window_sizes)
    n_rows = len(df)
    if n_rows <= max_window:
        logger.warning(f'Not enough data ({n_rows} rows) for window size {max_window}')
        return pd.DataFrame(), np.array([]), np.array([])

    red_cols = [f'red_{i}' for i in range(1, 7)]
    red_vals = df[red_cols].to_numpy()
    blue_vals = df['blue'].to_numpy()

    R = np.zeros((n_rows, 33), dtype=np.float32)
    B = np.zeros((n_rows, 16), dtype=np.float32)
    for i in range(n_rows):
        R[i, red_vals[i] - 1] = 1.0
        B[i, int(blue_vals[i]) - 1] = 1.0

    draw_feats = compute_draw_features(df)
    stat_cols = ['sum_value', 'odd_count', 'big_count', 'consecutive_count', 'ac_value', 'span']
    draw_stat_vals = {col: draw_feats[col].to_numpy() for col in stat_cols}

    X_rows = []
    y_red_rows = []
    y_blue_rows = []

    curr_red_missing = np.zeros(33, dtype=np.int32)
    curr_blue_missing = np.zeros(16, dtype=np.int32)

    for t in range(n_rows):
        if t >= max_window:
            features = {}
            for w in window_sizes:
                r_freq = R[t-w:t].sum(axis=0) / w
                b_freq = B[t-w:t].sum(axis=0) / w
                for n in range(33):
                    features[f'red_freq_{n+1}_w{w}'] = float(r_freq[n])
                for n in range(16):
                    features[f'blue_freq_{n+1}_w{w}'] = float(b_freq[n])

                for col in stat_cols:
                    w_vals = draw_stat_vals[col][t-w:t]
                    features[f'{col}_mean_w{w}'] = float(np.mean(w_vals))
                    features[f'{col}_std_w{w}'] = float(np.std(w_vals, ddof=1)) if w > 1 else 0.0

            for n in range(33):
                features[f'red_missing_{n+1}'] = int(curr_red_missing[n])
            for n in range(16):
                features[f'blue_missing_{n+1}'] = int(curr_blue_missing[n])

            X_rows.append(features)
            y_red_rows.append(R[t])
            y_blue_rows.append(B[t])

        curr_red_missing = np.where(R[t] > 0, 0, curr_red_missing + 1)
        curr_blue_missing = np.where(B[t] > 0, 0, curr_blue_missing + 1)

    X = pd.DataFrame(X_rows).fillna(0)
    y_red = np.array(y_red_rows)
    y_blue = np.array(y_blue_rows)
    return X, y_red, y_blue


def get_current_features(
    df: pd.DataFrame, window_sizes: Optional[List[int]] = None
) -> Dict[str, float]:
    """Get features for current state up to the latest draw (for prediction)."""
    if window_sizes is None:
        window_sizes = config.MODEL_WINDOW_SIZES

    red_cols = [f'red_{i}' for i in range(1, 7)]
    red_vals = df[red_cols].to_numpy()
    blue_vals = df['blue'].to_numpy()
    n_rows = len(df)

    R = np.zeros((n_rows, 33), dtype=np.float32)
    B = np.zeros((n_rows, 16), dtype=np.float32)
    for i in range(n_rows):
        R[i, red_vals[i] - 1] = 1.0
        B[i, int(blue_vals[i]) - 1] = 1.0

    draw_feats = compute_draw_features(df)
    stat_cols = ['sum_value', 'odd_count', 'big_count', 'consecutive_count', 'ac_value', 'span']

    features = {}
    for w in window_sizes:
        r_freq = R[-w:].sum(axis=0) / w
        b_freq = B[-w:].sum(axis=0) / w
        for n in range(33):
            features[f'red_freq_{n+1}_w{w}'] = float(r_freq[n])
        for n in range(16):
            features[f'blue_freq_{n+1}_w{w}'] = float(b_freq[n])

        for col in stat_cols:
            w_vals = draw_feats[col].iloc[-w:].to_numpy()
            features[f'{col}_mean_w{w}'] = float(np.mean(w_vals))
            features[f'{col}_std_w{w}'] = float(np.std(w_vals, ddof=1)) if w > 1 else 0.0

    red_missing = compute_missing_values(df, 'red')
    blue_missing = compute_missing_values(df, 'blue')
    for n in range(1, 34):
        features[f'red_missing_{n}'] = red_missing[n]
    for n in range(1, 17):
        features[f'blue_missing_{n}'] = blue_missing[n]

    return features

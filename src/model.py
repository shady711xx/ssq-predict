"""Prediction models for SSQ lottery.

Provides three individual strategies, probability smoothing, LLR scoring,
and an ensemble model:
- StatisticalModel (Strategy A): Frequency, missing values regression, and distribution constraints
- RandomForestModel (Strategy B): Regularized multi-output / multi-class Random Forest
- LogisticRegressionModel (Strategy C): L2-regularized Logistic Regression baseline
- EnsembleModel: Weighted blend of Strategies A, B, and C
"""
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import features as feat_module

logger = logging.getLogger('ssq.model')


def smooth_probabilities(
    probs: Dict[int, float],
    n_items: int,
    alpha: float = config.SMOOTHING_ALPHA,
    eps: float = config.SMOOTHING_EPS,
) -> Dict[int, float]:
    """Apply uniform smoothing and epsilon floor to probability distribution.
    
    P_smooth(x) = (1 - alpha) * P(x) + alpha * (1 / n_items)
    Guarantees every item has probability strictly > 0 and >= eps.
    """
    uniform_p = 1.0 / float(n_items)
    smoothed = {}
    for i in range(1, n_items + 1):
        raw_p = probs.get(i, uniform_p)
        p = (1.0 - alpha) * raw_p + alpha * uniform_p
        smoothed[i] = max(p, eps)
    total = sum(smoothed.values())
    return {k: float(v / total) for k, v in smoothed.items()}


def calculate_llr_score(
    red_probs: Dict[int, float],
    blue_probs: Dict[int, float],
    reds: List[int],
    blue: int,
) -> float:
    """Calculate combination score based on Log-Likelihood Ratio (LLR) against uniform prior.
    
    LLR = sum_{r in reds} ln(P(r) / (1/33)) + ln(P(blue) / (1/16))
    score = 1 / (1 + exp(-0.5 * LLR))
    
    Score is smooth, strictly in (0, 1), centered at 0.5000 for uniform random combinations.
    Never collapses to 0.0000.
    """
    uniform_red = 1.0 / 33.0
    uniform_blue = 1.0 / 16.0

    llr = 0.0
    for r in reds:
        p_r = max(red_probs.get(r, uniform_red), 1e-6)
        llr += np.log(p_r / uniform_red)
    p_b = max(blue_probs.get(blue, uniform_blue), 1e-6)
    llr += np.log(p_b / uniform_blue)

    # Sigmoid mapping: LLR=0 -> 0.5000; LLR=+2 -> 0.7311; LLR=-2 -> 0.2689
    score = 1.0 / (1.0 + np.exp(-0.5 * llr))
    return float(np.clip(score, 0.001, 0.999))


def generate_diversified_combinations(
    red_probs: Dict[int, float],
    blue_probs: Dict[int, float],
    n_combinations: int = 5,
    strategy_name: str = 'model',
    max_overlap: int = config.MAX_RED_OVERLAP,
    sum_mean: float = 102.0,
    sum_std: float = 16.0,
) -> List[Dict[str, Any]]:
    """Generate N combinations with diversified blue balls and controlled red overlap.
    
    1. Blue ball deduplication: Top-N distinct blue balls by predicted probability.
    2. Red ball overlap control: |R_i ∩ R_j| <= max_overlap (relaxed if necessary).
    """
    # 1. Select distinct top blue balls
    sorted_blues = sorted(blue_probs.keys(), key=lambda b: blue_probs[b], reverse=True)
    assigned_blues = [sorted_blues[i % len(sorted_blues)] for i in range(n_combinations)]

    red_nums = list(range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1))
    red_weights = np.array([red_probs.get(n, 1.0 / 33.0) for n in red_nums], dtype=float)
    red_weights = np.maximum(red_weights, 1e-6)
    red_weights /= red_weights.sum()

    results: List[Dict[str, Any]] = []
    current_max_overlap = max_overlap

    for i in range(n_combinations):
        blue = assigned_blues[i]
        best_candidate: Optional[List[int]] = None
        best_candidate_score = -1.0
        found = False

        # Attempt sampling with strict overlap constraint, relax if necessary
        for attempt_limit, allowed_overlap in [(150, current_max_overlap), (100, current_max_overlap + 1), (50, 6)]:
            for _ in range(attempt_limit):
                # Sample 6 distinct red balls
                avail_nums = red_nums.copy()
                avail_w = red_weights.copy()
                reds = []
                for _ in range(6):
                    w = avail_w / avail_w.sum()
                    idx = int(np.random.choice(len(avail_nums), p=w))
                    reds.append(avail_nums[idx])
                    avail_nums.pop(idx)
                    avail_w = np.delete(avail_w, idx)
                reds.sort()

                # Basic sum and odd/even filter
                sum_val = sum(reds)
                if abs(sum_val - sum_mean) > 3.0 * max(sum_std, 1.0):
                    continue
                odd_count = sum(1 for r in reds if r % 2 == 1)
                if odd_count == 0 or odd_count == 6:
                    continue

                # Overlap check with already selected combinations
                overlap_ok = True
                for prev in results:
                    overlap = len(set(reds) & set(prev['reds']))
                    if overlap > allowed_overlap:
                        overlap_ok = False
                        break

                if overlap_ok:
                    score = calculate_llr_score(red_probs, blue_probs, reds, blue)
                    best_candidate = reds
                    best_candidate_score = score
                    found = True
                    break

            if found:
                break

        if best_candidate is None:
            # Fallback: top 6 red balls
            top_reds = sorted(red_probs.keys(), key=lambda r: red_probs[r], reverse=True)[:6]
            best_candidate = sorted(top_reds)
            best_candidate_score = calculate_llr_score(red_probs, blue_probs, best_candidate, blue)

        odd, even = feat_module.compute_odd_even_ratio(best_candidate)
        results.append({
            'reds': best_candidate,
            'blue': blue,
            'score': round(best_candidate_score, 4),
            'strategy': strategy_name,
            'features': {
                'sum_value': int(sum(best_candidate)),
                'odd_even_ratio': f'{odd}:{even}',
                'ac_value': int(feat_module.compute_ac_value(best_candidate)),
                'consecutive': int(feat_module.compute_consecutive_count(best_candidate)),
            },
        })

    # Sort results by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


class StatisticalModel:
    """Strategy A: Statistical probability-weighted prediction model."""

    def __init__(self) -> None:
        self.red_probs: Dict[int, float] = {}
        self.blue_probs: Dict[int, float] = {}
        self.fitted: bool = False
        self.sum_mean: float = 102.0
        self.sum_std: float = 16.0

    def fit(self, df: pd.DataFrame) -> None:
        """Fit statistical frequency and missing value regression weights."""
        if df.empty or len(df) < 10:
            logger.warning('Insufficient data for statistical model fitting')
            return

        all_red_freq = feat_module.compute_frequency(df, 'red')
        recent_red_freq = feat_module.compute_frequency(df, 'red', window=30)
        all_blue_freq = feat_module.compute_frequency(df, 'blue')
        recent_blue_freq = feat_module.compute_frequency(df, 'blue', window=30)

        red_missing = feat_module.compute_missing_values(df, 'red')
        blue_missing = feat_module.compute_missing_values(df, 'blue')
        red_max_missing = feat_module.compute_max_missing(df, 'red')
        blue_max_missing = feat_module.compute_max_missing(df, 'blue')

        total_draws = len(df)
        raw_red: Dict[int, float] = {}
        for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1):
            base_freq = all_red_freq.get(n, 0) / total_draws if total_draws > 0 else 1 / 33
            w30 = min(30, total_draws)
            recent_freq = recent_red_freq.get(n, 0) / w30 if w30 > 0 else 1 / 33
            curr_miss = red_missing.get(n, 0)
            max_miss = red_max_missing.get(n, 1)
            miss_ratio = curr_miss / max(max_miss, 1)
            miss_weight = 1.0 + 1.5 * (miss_ratio**2)
            raw_red[n] = 0.40 * base_freq + 0.30 * recent_freq + 0.30 * (miss_weight * base_freq)

        self.red_probs = smooth_probabilities(raw_red, 33)

        raw_blue: Dict[int, float] = {}
        for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1):
            base_freq = all_blue_freq.get(n, 0) / total_draws if total_draws > 0 else 1 / 16
            w30 = min(30, total_draws)
            recent_freq = recent_blue_freq.get(n, 0) / w30 if w30 > 0 else 1 / 16
            curr_miss = blue_missing.get(n, 0)
            max_miss = blue_max_missing.get(n, 1)
            miss_ratio = curr_miss / max(max_miss, 1)
            miss_weight = 1.0 + 1.5 * (miss_ratio**2)
            raw_blue[n] = 0.40 * base_freq + 0.30 * recent_freq + 0.30 * (miss_weight * base_freq)

        self.blue_probs = smooth_probabilities(raw_blue, 16)

        draw_feats = feat_module.compute_draw_features(df)
        self.sum_mean = float(draw_feats['sum_value'].mean())
        self.sum_std = float(draw_feats['sum_value'].std())
        self.fitted = True

    def predict(
        self,
        n_combinations: int = 5,
        df: Optional[pd.DataFrame] = None,
        max_overlap: int = config.MAX_RED_OVERLAP,
    ) -> List[Dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError('Model not fitted. Call fit() first.')
        return generate_diversified_combinations(
            self.red_probs,
            self.blue_probs,
            n_combinations=n_combinations,
            strategy_name='statistical',
            max_overlap=max_overlap,
            sum_mean=self.sum_mean,
            sum_std=self.sum_std,
        )

    def score_combination(self, reds: List[int], blue: int) -> float:
        if not self.fitted:
            return 0.5000
        return calculate_llr_score(self.red_probs, self.blue_probs, reds, blue)

    def get_probabilities(self) -> Dict[str, Dict[int, float]]:
        return {'red': dict(self.red_probs), 'blue': dict(self.blue_probs)}


class RandomForestModel:
    """Strategy B: Regularized multi-output / multi-class Random Forest classifier."""

    def __init__(self, n_estimators: int = 50) -> None:
        self.n_estimators = n_estimators
        self.red_model: Optional[RandomForestClassifier] = None
        self.blue_model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.fitted: bool = False
        self.red_probs: Dict[int, float] = {}
        self.blue_probs: Dict[int, float] = {}
        self.feature_names: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> None:
        if len(df) < 50:
            logger.warning(f'Insufficient data for RandomForest ({len(df)} rows, need >= 50)')
            return

        X, y_red, y_blue = feat_module.build_feature_matrix(df)
        if X.empty:
            return

        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        # Multi-output binary Random Forest with regularization against overfitting
        self.red_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        )
        self.red_model.fit(X_scaled, y_red)

        # Multi-class Random Forest for blue ball (1..16)
        blue_target = np.argmax(y_blue, axis=1) + 1
        self.blue_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        )
        self.blue_model.fit(X_scaled, blue_target)

        self._update_probabilities(X_scaled[-1:])
        self.fitted = True

    def fit_with_matrix(
        self,
        X_train: pd.DataFrame,
        y_red_train: np.ndarray,
        y_blue_train: np.ndarray,
        X_latest: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fast fit using pre-computed feature matrix."""
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)

        self.red_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        )
        self.red_model.fit(X_scaled, y_red_train)

        blue_target = np.argmax(y_blue_train, axis=1) + 1
        self.blue_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        )
        self.blue_model.fit(X_scaled, blue_target)

        if X_latest is not None:
            X_latest_scaled = self.scaler.transform(X_latest)
            self._update_probabilities(X_latest_scaled)
        else:
            self._update_probabilities(X_scaled[-1:])
        self.fitted = True

    def _update_probabilities(self, X_latest: np.ndarray) -> None:
        raw_red: Dict[int, float] = {}
        if self.red_model is not None:
            try:
                red_proba = self.red_model.predict_proba(X_latest)
                for i in range(33):
                    if isinstance(red_proba, list) and i < len(red_proba):
                        arr = red_proba[i]
                        raw_red[i + 1] = float(arr[0, 1]) if arr.shape[1] > 1 else 0.5
                    elif isinstance(red_proba, np.ndarray):
                        raw_red[i + 1] = float(red_proba[0, i])
            except Exception as e:
                logger.debug(f'Error getting RF red probabilities: {e}')
        self.red_probs = smooth_probabilities(raw_red, 33)

        raw_blue: Dict[int, float] = {}
        if self.blue_model is not None:
            try:
                blue_proba = self.blue_model.predict_proba(X_latest)
                if isinstance(blue_proba, np.ndarray) and blue_proba.ndim == 2:
                    probs_row = blue_proba[0]
                    classes = self.blue_model.classes_
                    for ball in range(1, 17):
                        if ball in classes:
                            idx = list(classes).index(ball)
                            raw_blue[ball] = float(probs_row[idx])
                        else:
                            raw_blue[ball] = 1.0 / 16.0
            except Exception as e:
                logger.debug(f'Error getting RF blue probabilities: {e}')
        self.blue_probs = smooth_probabilities(raw_blue, 16)

    def predict_with_features(self, df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
        current_features = feat_module.get_current_features(df)
        if not current_features or not self.feature_names:
            return dict(self.red_probs), dict(self.blue_probs)

        X = pd.DataFrame([current_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        self._update_probabilities(X_scaled)
        return dict(self.red_probs), dict(self.blue_probs)

    def predict(
        self,
        n_combinations: int = 5,
        df: Optional[pd.DataFrame] = None,
        max_overlap: int = config.MAX_RED_OVERLAP,
    ) -> List[Dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError('Model not fitted. Call fit() first.')
        if df is not None and not df.empty:
            self.predict_with_features(df)
        return generate_diversified_combinations(
            self.red_probs,
            self.blue_probs,
            n_combinations=n_combinations,
            strategy_name='random_forest',
            max_overlap=max_overlap,
        )

    def score_combination(self, reds: List[int], blue: int) -> float:
        if not self.fitted:
            return 0.5000
        return calculate_llr_score(self.red_probs, self.blue_probs, reds, blue)

    def get_probabilities(self) -> Dict[str, Dict[int, float]]:
        return {'red': dict(self.red_probs), 'blue': dict(self.blue_probs)}


# Backwards compatibility alias
MLModel = RandomForestModel


class LogisticRegressionModel:
    """Strategy C: L2-regularized Logistic Regression multi-label predictor."""

    def __init__(self, C: float = 0.001, C_blue: float = 0.0005) -> None:
        self.C = C
        self.C_blue = C_blue
        self.scaler = StandardScaler()
        self.red_clfs: Dict[int, Any] = {}
        self.blue_clf: Optional[Any] = None
        self.fitted: bool = False
        self.red_probs: Dict[int, float] = {}
        self.blue_probs: Dict[int, float] = {}
        self.feature_names: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> None:
        if len(df) < 50:
            logger.warning(f'Insufficient data for LogisticRegression ({len(df)} rows, need >= 50)')
            return

        X, y_red, y_blue = feat_module.build_feature_matrix(df)
        if X.empty:
            return

        self.feature_names = X.columns.tolist()

        # Train 33 L2 logistic regressions for red balls using Pipeline
        self.red_clfs = {}
        for i in range(33):
            ball = i + 1
            if len(np.unique(y_red[:, i])) < 2:
                continue
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=self.C, max_iter=1000, solver='lbfgs', random_state=42)),
            ])
            pipe.fit(X, y_red[:, i])
            self.red_clfs[ball] = pipe

        # Train multinomial logistic regression for blue ball using Pipeline
        blue_target = np.argmax(y_blue, axis=1) + 1
        if len(np.unique(blue_target)) >= 2:
            self.blue_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=self.C_blue, max_iter=1000, solver='lbfgs', random_state=42)),
            ])
            self.blue_clf.fit(X, blue_target)
        else:
            self.blue_clf = None

        self._update_probabilities(X.iloc[-1:])
        self.fitted = True

    def fit_with_matrix(
        self,
        X_train: pd.DataFrame,
        y_red_train: np.ndarray,
        y_blue_train: np.ndarray,
        X_latest: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fast fit using pre-computed feature matrix and Pipeline."""
        self.feature_names = X_train.columns.tolist()

        self.red_clfs = {}
        for i in range(33):
            ball = i + 1
            if len(np.unique(y_red_train[:, i])) < 2:
                continue
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=self.C, max_iter=1000, solver='lbfgs', random_state=42)),
            ])
            pipe.fit(X_train, y_red_train[:, i])
            self.red_clfs[ball] = pipe

        blue_target = np.argmax(y_blue_train, axis=1) + 1
        if len(np.unique(blue_target)) >= 2:
            self.blue_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=self.C_blue, max_iter=1000, solver='lbfgs', random_state=42)),
            ])
            self.blue_clf.fit(X_train, blue_target)
        else:
            self.blue_clf = None

        if X_latest is not None:
            self._update_probabilities(X_latest)
        else:
            self._update_probabilities(X_train.iloc[-1:])
        self.fitted = True

    def _update_probabilities(self, X_latest: Any) -> None:
        raw_red: Dict[int, float] = {}
        for ball in range(1, 34):
            if ball in self.red_clfs:
                try:
                    proba = self.red_clfs[ball].predict_proba(X_latest)[0]
                    raw_red[ball] = float(proba[1]) if len(proba) > 1 else 0.5
                except Exception:
                    raw_red[ball] = 1.0 / 33.0
            else:
                raw_red[ball] = 1.0 / 33.0
        self.red_probs = smooth_probabilities(raw_red, 33)

        raw_blue: Dict[int, float] = {}
        if self.blue_clf is not None:
            try:
                proba = self.blue_clf.predict_proba(X_latest)[0]
                classes = self.blue_clf.named_steps['clf'].classes_
                for ball in range(1, 17):
                    if ball in classes:
                        idx = list(classes).index(ball)
                        raw_blue[ball] = float(proba[idx])
                    else:
                        raw_blue[ball] = 1.0 / 16.0
            except Exception:
                for ball in range(1, 17):
                    raw_blue[ball] = 1.0 / 16.0
        else:
            for ball in range(1, 17):
                raw_blue[ball] = 1.0 / 16.0
        self.blue_probs = smooth_probabilities(raw_blue, 16, alpha=0.25)

    def predict_with_features(self, df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
        current_features = feat_module.get_current_features(df)
        if not current_features or not self.feature_names:
            return dict(self.red_probs), dict(self.blue_probs)

        X = pd.DataFrame([current_features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]

        self._update_probabilities(X)
        return dict(self.red_probs), dict(self.blue_probs)

    def predict(
        self,
        n_combinations: int = 5,
        df: Optional[pd.DataFrame] = None,
        max_overlap: int = config.MAX_RED_OVERLAP,
    ) -> List[Dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError('Model not fitted. Call fit() first.')
        if df is not None and not df.empty:
            self.predict_with_features(df)
        return generate_diversified_combinations(
            self.red_probs,
            self.blue_probs,
            n_combinations=n_combinations,
            strategy_name='logistic_regression',
            max_overlap=max_overlap,
        )

    def score_combination(self, reds: List[int], blue: int) -> float:
        if not self.fitted:
            return 0.5000
        return calculate_llr_score(self.red_probs, self.blue_probs, reds, blue)

    def get_probabilities(self) -> Dict[str, Dict[int, float]]:
        return {'red': dict(self.red_probs), 'blue': dict(self.blue_probs)}


class EnsembleModel:
    """Ensemble model blending Strategy A (Stat), Strategy B (RF), and Strategy C (LR)."""

    def __init__(
        self,
        stat_weight: float = config.STAT_MODEL_WEIGHT,
        rf_weight: float = config.RF_MODEL_WEIGHT,
        lr_weight: float = config.LR_MODEL_WEIGHT,
    ) -> None:
        self.stat_weight = stat_weight
        self.rf_weight = rf_weight
        self.lr_weight = lr_weight
        self.stat_model = StatisticalModel()
        self.rf_model = RandomForestModel()
        self.lr_model = LogisticRegressionModel()
        self.red_probs: Dict[int, float] = {}
        self.blue_probs: Dict[int, float] = {}
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit all component models and blend probabilities."""
        self.stat_model.fit(df)

        if len(df) >= 50:
            self.rf_model.fit(df)
            self.lr_model.fit(df)
            self._merge_probabilities(df)
        else:
            self.red_probs = dict(self.stat_model.red_probs)
            self.blue_probs = dict(self.stat_model.blue_probs)

        self.fitted = True
        logger.info('Ensemble model fitted successfully')

    def fit_with_components(
        self,
        stat_model: StatisticalModel,
        rf_model: RandomForestModel,
        lr_model: LogisticRegressionModel,
    ) -> None:
        """Fast ensemble using pre-fitted component models."""
        self.stat_model = stat_model
        self.rf_model = rf_model
        self.lr_model = lr_model
        self._merge_probabilities()
        self.fitted = True

    def _merge_probabilities(self, df: Optional[pd.DataFrame] = None) -> None:
        p_stat = self.stat_model.get_probabilities()
        p_rf = self.rf_model.get_probabilities() if self.rf_model.fitted else p_stat
        p_lr = self.lr_model.get_probabilities() if self.lr_model.fitted else p_stat

        w_stat = self.stat_weight
        w_rf = self.rf_weight
        w_lr = self.lr_weight
        total_w = w_stat + w_rf + w_lr

        raw_red = {}
        for n in range(config.RED_BALL_MIN, config.RED_BALL_MAX + 1):
            val = (
                w_stat * p_stat['red'].get(n, 1 / 33)
                + w_rf * p_rf['red'].get(n, 1 / 33)
                + w_lr * p_lr['red'].get(n, 1 / 33)
            ) / total_w
            raw_red[n] = val
        self.red_probs = smooth_probabilities(raw_red, 33)

        raw_blue = {}
        for n in range(config.BLUE_BALL_MIN, config.BLUE_BALL_MAX + 1):
            val = (
                w_stat * p_stat['blue'].get(n, 1 / 16)
                + w_rf * p_rf['blue'].get(n, 1 / 16)
                + w_lr * p_lr['blue'].get(n, 1 / 16)
            ) / total_w
            raw_blue[n] = val
        self.blue_probs = smooth_probabilities(raw_blue, 16)

    def predict(
        self,
        n_combinations: int = 5,
        df: Optional[pd.DataFrame] = None,
        max_overlap: int = config.MAX_RED_OVERLAP,
    ) -> List[Dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError('Model not fitted. Call fit() first.')

        if df is not None and not df.empty:
            if self.rf_model.fitted:
                self.rf_model.predict_with_features(df)
            if self.lr_model.fitted:
                self.lr_model.predict_with_features(df)
            self._merge_probabilities(df)

        return generate_diversified_combinations(
            self.red_probs,
            self.blue_probs,
            n_combinations=n_combinations,
            strategy_name='ensemble',
            max_overlap=max_overlap,
        )

    def score_combination(self, reds: List[int], blue: int) -> float:
        if not self.fitted:
            return 0.5000
        return calculate_llr_score(self.red_probs, self.blue_probs, reds, blue)

    def get_probabilities(self) -> Dict[str, Dict[int, float]]:
        return {'red': dict(self.red_probs), 'blue': dict(self.blue_probs)}

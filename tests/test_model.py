"""Unit tests for SSQ prediction models, probability smoothing, and diversified sampling."""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import (
    StatisticalModel,
    RandomForestModel,
    LogisticRegressionModel,
    EnsembleModel,
    smooth_probabilities,
    calculate_llr_score,
    generate_diversified_combinations,
)


def make_mock_df(n=60):
    """Generate minimal valid DataFrame for model training."""
    np.random.seed(42)
    rows = []
    for i in range(1, n + 1):
        reds = sorted(np.random.choice(range(1, 34), 6, replace=False).tolist())
        blue = int(np.random.randint(1, 17))
        rows.append({
            'issue': f'{24000 + i:05d}',
            'draw_date': f'2024-01-{(i % 28) + 1:02d}',
            'red_1': reds[0], 'red_2': reds[1], 'red_3': reds[2],
            'red_4': reds[3], 'red_5': reds[4], 'red_6': reds[5],
            'blue': blue,
        })
    return pd.DataFrame(rows)


class TestProbabilitySmoothing:
    def test_smooth_probabilities_strictly_positive(self):
        # Raw probabilities with many zeros
        raw_p = {1: 0.9, 2: 0.1}  # others missing -> 0
        smoothed = smooth_probabilities(raw_p, 33, alpha=0.10, eps=1e-4)

        assert len(smoothed) == 33
        assert abs(sum(smoothed.values()) - 1.0) < 1e-5
        for ball, p in smoothed.items():
            assert p > 0.0, f"Ball {ball} has non-positive probability"
            assert p >= 1e-4, f"Ball {ball} probability below eps floor"

    def test_smooth_blue_probabilities(self):
        raw_blue = {1: 1.0}
        smoothed = smooth_probabilities(raw_blue, 16, alpha=0.10, eps=1e-4)
        assert len(smoothed) == 16
        assert abs(sum(smoothed.values()) - 1.0) < 1e-5
        for ball, p in smoothed.items():
            assert p > 0.0


class TestLLRScoreNonZero:
    def test_score_never_collapses_to_zero(self):
        """Even with deliberately disfavored numbers, score must remain > 0."""
        # Unfavorable probabilities
        red_probs = {i: 0.001 for i in range(1, 34)}
        blue_probs = {j: 0.001 for j in range(1, 17)}

        score = calculate_llr_score(red_probs, blue_probs, [1, 2, 3, 4, 5, 6], 7)
        assert score > 0.0, "Score collapsed to zero!"
        assert score >= 0.001

    def test_score_uniform_baseline_is_neutral(self):
        """Uniform distribution should produce score near 0.5000."""
        red_probs = {i: 1.0 / 33.0 for i in range(1, 34)}
        blue_probs = {j: 1.0 / 16.0 for j in range(1, 17)}

        score = calculate_llr_score(red_probs, blue_probs, [1, 2, 3, 4, 5, 6], 7)
        assert abs(score - 0.5000) < 0.001

    def test_score_favored_is_above_half(self):
        """Favored numbers should score > 0.5000."""
        red_probs = {i: 0.01 for i in range(1, 34)}
        for r in [1, 2, 3, 4, 5, 6]:
            red_probs[r] = 0.10
        blue_probs = {j: 0.01 for j in range(1, 17)}
        blue_probs[7] = 0.50

        score = calculate_llr_score(red_probs, blue_probs, [1, 2, 3, 4, 5, 6], 7)
        assert score > 0.5000


class TestDiversifiedSampling:
    def test_blue_ball_distinctness(self):
        """Top 5 recommendations must have 5 distinct blue balls."""
        red_probs = {i: 1.0 / 33.0 for i in range(1, 34)}
        blue_probs = {j: float(j) / 136.0 for j in range(1, 17)}

        combos = generate_diversified_combinations(
            red_probs, blue_probs, n_combinations=5, max_overlap=3
        )
        assert len(combos) == 5
        blue_balls = [c['blue'] for c in combos]
        assert len(set(blue_balls)) == 5, f"Blue balls not distinct: {blue_balls}"

    def test_red_ball_overlap_constraint(self):
        """Any two combinations should have limited red ball overlap."""
        red_probs = {i: 1.0 / 33.0 for i in range(1, 34)}
        blue_probs = {j: 1.0 / 16.0 for j in range(1, 17)}

        combos = generate_diversified_combinations(
            red_probs, blue_probs, n_combinations=5, max_overlap=3
        )
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                overlap = len(set(combos[i]['reds']) & set(combos[j]['reds']))
                assert overlap <= 4, f"Overlap between {i} and {j} too high: {overlap}"


class TestModelTrainingAndScoring:
    @pytest.fixture
    def mock_df(self):
        return make_mock_df(60)

    def test_statistical_model(self, mock_df):
        m = StatisticalModel()
        m.fit(mock_df)
        assert m.fitted
        probs = m.get_probabilities()
        assert len(probs['red']) == 33
        assert len(probs['blue']) == 16
        for p in probs['red'].values():
            assert p > 0.0

        score = m.score_combination([1, 2, 3, 4, 5, 6], 7)
        assert score > 0.0

        preds = m.predict(n_combinations=5)
        assert len(preds) == 5
        for p in preds:
            assert p['score'] > 0.0

    def test_random_forest_model(self, mock_df):
        m = RandomForestModel(n_estimators=10)
        m.fit(mock_df)
        assert m.fitted

        probs = m.get_probabilities()
        for p in probs['red'].values():
            assert p > 0.0
        for p in probs['blue'].values():
            assert p > 0.0

        preds = m.predict(n_combinations=5)
        assert len(preds) == 5
        for p in preds:
            assert p['score'] > 0.0

    def test_logistic_regression_model(self, mock_df):
        m = LogisticRegressionModel(C=0.1)
        m.fit(mock_df)
        assert m.fitted

        probs = m.get_probabilities()
        for p in probs['red'].values():
            assert p > 0.0
        for p in probs['blue'].values():
            assert p > 0.0

        preds = m.predict(n_combinations=5)
        assert len(preds) == 5
        for p in preds:
            assert p['score'] > 0.0

    def test_ensemble_model(self, mock_df):
        m = EnsembleModel()
        m.fit(mock_df)
        assert m.fitted

        probs = m.get_probabilities()
        for p in probs['red'].values():
            assert p > 0.0

        preds = m.predict(n_combinations=5)
        assert len(preds) == 5
        for p in preds:
            assert p['score'] > 0.0

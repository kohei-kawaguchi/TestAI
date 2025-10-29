"""
Unit tests for MDP estimator subroutines.

These tests are based solely on the pseudo code specifications:
- Input types and shapes as specified
- Output types and shapes as specified
- Mathematical properties as defined in pseudo code

Tests do NOT examine internal implementation details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pytest
from mdp_estimator import EstimateGamma


class TestEstimateGamma:
    """Test EstimateGamma procedure based on pseudo code."""

    def test_returns_float(self):
        """Should return float."""
        states = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        actions = np.array([[0, 1, 0], [1, 0, 1]])

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert isinstance(gamma_hat, (float, np.floating))

    def test_accepts_array_inputs(self):
        """Should accept Array[M×T] inputs."""
        M, T = 5, 10
        states = np.random.rand(M, T)
        actions = np.random.randint(0, 2, size=(M, T))

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert isinstance(gamma_hat, (float, np.floating))

    def test_ols_formula_for_known_data(self):
        """Should recover gamma from perfectly linear data."""
        # Create data with known gamma
        true_gamma = 0.2
        M, T = 10, 20

        states = np.zeros((M, T))
        actions = np.zeros((M, T), dtype=int)

        # Generate data following s_{t+1} = (1 - gamma) * s_t + a_t
        for m in range(M):
            states[m, 0] = np.random.rand()  # Initial state
            for t in range(T - 1):
                actions[m, t] = np.random.randint(0, 2)
                states[m, t + 1] = (1 - true_gamma) * states[m, t] + actions[m, t]

        gamma_hat = EstimateGamma(states=states, actions=actions)

        # Should recover true gamma (within numerical precision)
        assert abs(gamma_hat - true_gamma) < 1e-6

    def test_with_zero_depreciation(self):
        """Should return gamma=0 when states fully persist."""
        # Create data with gamma = 0 (full persistence)
        M, T = 5, 10

        states = np.zeros((M, T))
        actions = np.zeros((M, T), dtype=int)

        for m in range(M):
            states[m, 0] = m + 1.0  # Initial state
            for t in range(T - 1):
                actions[m, t] = np.random.randint(0, 2)
                states[m, t + 1] = states[m, t] + actions[m, t]  # gamma = 0

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert abs(gamma_hat) < 1e-6

    def test_with_full_depreciation(self):
        """Should return gamma=1 when states fully depreciate."""
        # Create data with gamma = 1 (full depreciation)
        M, T = 5, 10

        states = np.zeros((M, T))
        actions = np.zeros((M, T), dtype=int)

        for m in range(M):
            states[m, 0] = 5.0  # Initial state (doesn't matter)
            for t in range(T - 1):
                actions[m, t] = np.random.randint(0, 2)
                states[m, t + 1] = actions[m, t]  # s' = a (gamma = 1)

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert abs(gamma_hat - 1.0) < 1e-6

    def test_uses_all_transitions(self):
        """Should use T-1 transitions per path (exclude last period)."""
        # This is tested implicitly by correct OLS formula
        M, T = 3, 5
        states = np.random.rand(M, T)
        actions = np.random.randint(0, 2, size=(M, T))

        gamma_hat = EstimateGamma(states=states, actions=actions)

        # Should produce finite result
        assert np.isfinite(gamma_hat)

    def test_handles_different_initial_states(self):
        """Should work with varying initial states."""
        true_gamma = 0.3
        M, T = 8, 15

        states = np.zeros((M, T))
        actions = np.zeros((M, T), dtype=int)

        # Different initial states for each path
        for m in range(M):
            states[m, 0] = np.random.rand() * 10  # Varying initial states
            for t in range(T - 1):
                actions[m, t] = np.random.randint(0, 2)
                states[m, t + 1] = (1 - true_gamma) * states[m, t] + actions[m, t]

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert abs(gamma_hat - true_gamma) < 1e-6

    def test_positive_output_for_stable_system(self):
        """Gamma should be positive for stationary systems."""
        # Create data with gamma in (0, 1)
        true_gamma = 0.15
        M, T = 10, 20

        states = np.zeros((M, T))
        actions = np.zeros((M, T), dtype=int)

        for m in range(M):
            states[m, 0] = 2.0
            for t in range(T - 1):
                actions[m, t] = np.random.randint(0, 2)
                states[m, t + 1] = (1 - true_gamma) * states[m, t] + actions[m, t]

        gamma_hat = EstimateGamma(states=states, actions=actions)

        # Should be positive
        assert gamma_hat > 0

    def test_formula_1_minus_numer_over_denom(self):
        """Should implement gamma_hat = 1 - numer/denom formula."""
        # Create simple data to manually verify
        states = np.array([
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ])
        actions = np.array([
            [1, 1, 0],
            [1, 1, 0]
        ])

        # Manually compute OLS
        numer = 0.0
        denom = 0.0
        M, T = states.shape

        for m in range(M):
            for t in range(T - 1):
                s_t = states[m, t]
                a_t = actions[m, t]
                s_t_plus_1 = states[m, t + 1]
                numer += s_t * (s_t_plus_1 - a_t)
                denom += s_t ** 2

        expected_gamma = 1.0 - numer / denom

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert abs(gamma_hat - expected_gamma) < 1e-10

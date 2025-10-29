"""
Unit tests for MDP simulator subroutines.

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
import torch
import pytest
from mdp_simulator import DrawAction
from mdp_solver import MonotonicNetwork, InitializeNetworks


class TestDrawAction:
    """Test DrawAction procedure based on pseudo code."""

    def test_returns_int(self):
        """Should return int (0 or 1)."""
        P_0 = 0.7
        P_1 = 0.3

        action = DrawAction(P_0=P_0, P_1=P_1)

        assert isinstance(action, (int, np.integer))

    def test_returns_valid_action(self):
        """Should return 0 or 1."""
        P_0 = 0.6
        P_1 = 0.4

        action = DrawAction(P_0=P_0, P_1=P_1)

        assert action in [0, 1]

    def test_deterministic_when_prob_is_one(self):
        """Should always return 0 when P_0=1.0."""
        P_0 = 1.0
        P_1 = 0.0

        actions = [DrawAction(P_0=P_0, P_1=P_1) for _ in range(10)]

        assert all(a == 0 for a in actions)

    def test_deterministic_when_prob_is_zero(self):
        """Should always return 1 when P_0=0.0."""
        P_0 = 0.0
        P_1 = 1.0

        actions = [DrawAction(P_0=P_0, P_1=P_1) for _ in range(10)]

        assert all(a == 1 for a in actions)

    def test_respects_probability_distribution(self):
        """Should draw actions according to probabilities (statistical test)."""
        np.random.seed(42)
        P_0 = 0.7
        P_1 = 0.3
        n_samples = 10000

        actions = [DrawAction(P_0=P_0, P_1=P_1) for _ in range(n_samples)]
        freq_0 = sum(1 for a in actions if a == 0) / n_samples

        # Statistical test: frequency should be close to probability
        # With 10000 samples, we expect ~0.7 ± 0.01 (2 standard errors)
        assert abs(freq_0 - P_0) < 0.02

    def test_inverse_cdf_method(self):
        """Should implement inverse CDF method: if u < P_0 then return 0."""
        # This is implicit in the behavior, tested via other tests
        P_0 = 0.5
        P_1 = 0.5

        # With equal probabilities, should get mix of 0 and 1
        np.random.seed(123)
        actions = [DrawAction(P_0=P_0, P_1=P_1) for _ in range(100)]

        # Should have both actions
        assert 0 in actions
        assert 1 in actions

    def test_handles_edge_cases(self):
        """Should handle edge case probabilities."""
        # Test with very small but non-zero probability
        P_0 = 0.01
        P_1 = 0.99

        np.random.seed(42)
        actions = [DrawAction(P_0=P_0, P_1=P_1) for _ in range(1000)]

        # Should mostly return 1, but occasionally 0
        freq_1 = sum(1 for a in actions if a == 1) / len(actions)
        assert freq_1 > 0.95

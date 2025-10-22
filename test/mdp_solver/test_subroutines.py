"""
Unit tests for MDP solver subroutines.

These tests are based solely on the pseudo code specifications:
- Input types and shapes as specified
- Output types and shapes as specified
- Mathematical properties as defined in pseudo code

Tests do NOT examine internal implementation details.
"""

import torch
import pytest
from mdp_solver import (
    InitializeNetworks,
    GenerateStateGrid,
    ComputeNextState,
    ComputeMeanReward,
    LogSumExp,
    ComputeExpectedValue,
    ComputeBellmanTargets,
    ComputeLoss,
    CheckConvergence,
    ComputeChoiceProbability,
    MonotonicNetwork,
)


class TestInitializeNetworks:
    """Test InitializeNetworks procedure based on pseudo code."""

    def test_returns_two_networks(self):
        """Should return tuple of (Network, Network)."""
        hyperparameters = {'hidden_sizes': [32, 32]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        assert isinstance(v_theta_0, MonotonicNetwork)
        assert isinstance(v_theta_1, MonotonicNetwork)

    def test_networks_are_distinct(self):
        """Two networks should be separate instances."""
        hyperparameters = {'hidden_sizes': [32, 32]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        assert v_theta_0 is not v_theta_1

    def test_accepts_hyperparameters(self):
        """Should accept hyperparameters dict."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        assert v_theta_0 is not None
        assert v_theta_1 is not None


class TestGenerateStateGrid:
    """Test GenerateStateGrid procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×1]."""
        N = 50
        state_range = (0.0, 10.0)
        S = GenerateStateGrid(N=N, state_range=state_range)

        assert S.shape == (N, 1)

    def test_output_type(self):
        """Should return torch.Tensor."""
        N = 50
        state_range = (0.0, 10.0)
        S = GenerateStateGrid(N=N, state_range=state_range)

        assert isinstance(S, torch.Tensor)

    def test_covers_state_range(self):
        """Generated states should cover the specified range."""
        N = 100
        state_range = (2.0, 8.0)
        S = GenerateStateGrid(N=N, state_range=state_range)

        assert S.min().item() >= state_range[0]
        assert S.max().item() <= state_range[1]

    def test_uniform_grid(self):
        """Should create uniform grid as specified in pseudo code."""
        N = 10
        state_range = (0.0, 9.0)
        S = GenerateStateGrid(N=N, state_range=state_range)

        # Check if spacing is uniform
        diffs = S[1:] - S[:-1]
        assert torch.allclose(diffs, diffs[0], atol=1e-6)


class TestComputeNextState:
    """Test ComputeNextState procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×1] given Tensor[N×1]."""
        s = torch.tensor([[1.0], [2.0], [3.0]])
        a = 0
        gamma = 0.1

        s_next = ComputeNextState(s=s, a=a, gamma=gamma)

        assert s_next.shape == s.shape

    def test_formula_action_0(self):
        """Should compute s' = (1-gamma)*s + a for action 0."""
        s = torch.tensor([[5.0]])
        a = 0
        gamma = 0.2

        s_next = ComputeNextState(s=s, a=a, gamma=gamma)
        expected = (1 - gamma) * s + a

        assert torch.allclose(s_next, expected)

    def test_formula_action_1(self):
        """Should compute s' = (1-gamma)*s + a for action 1."""
        s = torch.tensor([[5.0]])
        a = 1
        gamma = 0.2

        s_next = ComputeNextState(s=s, a=a, gamma=gamma)
        expected = (1 - gamma) * s + a

        assert torch.allclose(s_next, expected)

    def test_batch_computation(self):
        """Should work for batches of states."""
        s = torch.tensor([[1.0], [2.0], [3.0]])
        a = 1
        gamma = 0.1

        s_next = ComputeNextState(s=s, a=a, gamma=gamma)

        assert s_next.shape == (3, 1)


class TestComputeMeanReward:
    """Test ComputeMeanReward procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×1] given Tensor[N×1]."""
        s = torch.tensor([[1.0], [2.0], [3.0]])
        a = 0
        beta = 0.5

        r = ComputeMeanReward(s=s, a=a, beta=beta)

        assert r.shape == s.shape

    def test_formula(self):
        """Should compute r = beta*s - a."""
        s = torch.tensor([[4.0]])
        a = 1
        beta = 0.5

        r = ComputeMeanReward(s=s, a=a, beta=beta)
        expected = beta * s - a

        assert torch.allclose(r, expected)

    def test_action_0(self):
        """Reward for action 0 should be beta*s."""
        s = torch.tensor([[3.0]])
        a = 0
        beta = 0.6

        r = ComputeMeanReward(s=s, a=a, beta=beta)

        assert torch.allclose(r, torch.tensor([[1.8]]))

    def test_action_1(self):
        """Reward for action 1 should be beta*s - 1."""
        s = torch.tensor([[3.0]])
        a = 1
        beta = 0.6

        r = ComputeMeanReward(s=s, a=a, beta=beta)

        assert torch.allclose(r, torch.tensor([[0.8]]))


class TestLogSumExp:
    """Test LogSumExp procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×1] given two Tensor[N×1]."""
        v_0 = torch.tensor([[1.0], [2.0]])
        v_1 = torch.tensor([[1.5], [2.5]])

        result = LogSumExp(v_0=v_0, v_1=v_1)

        assert result.shape == v_0.shape

    def test_mathematical_property(self):
        """Should satisfy log(exp(v0) + exp(v1))."""
        v_0 = torch.tensor([[1.0]])
        v_1 = torch.tensor([[2.0]])

        result = LogSumExp(v_0=v_0, v_1=v_1)
        expected = torch.log(torch.exp(v_0) + torch.exp(v_1))

        assert torch.allclose(result, expected, atol=1e-5)

    def test_numerical_stability(self):
        """Should be numerically stable for large values."""
        v_0 = torch.tensor([[100.0]])
        v_1 = torch.tensor([[101.0]])

        result = LogSumExp(v_0=v_0, v_1=v_1)

        # Should not overflow or return inf/nan
        assert torch.isfinite(result).all()

    def test_symmetry(self):
        """LogSumExp(a, b) should equal LogSumExp(b, a)."""
        v_0 = torch.tensor([[3.0], [4.0]])
        v_1 = torch.tensor([[5.0], [2.0]])

        result1 = LogSumExp(v_0=v_0, v_1=v_1)
        result2 = LogSumExp(v_0=v_1, v_1=v_0)

        assert torch.allclose(result1, result2)


class TestComputeExpectedValue:
    """Test ComputeExpectedValue procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×1]."""
        s_prime = torch.tensor([[1.0], [2.0]])
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        gamma_E = 0.5772156649015329

        EV = ComputeExpectedValue(s_prime=s_prime, v_theta_0=v_theta_0, v_theta_1=v_theta_1, gamma_E=gamma_E)

        assert EV.shape == s_prime.shape

    def test_includes_euler_mascheroni(self):
        """Expected value should include gamma_E constant."""
        s_prime = torch.tensor([[1.0]])
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        gamma_E = 0.5772156649015329

        with torch.no_grad():
            v_0 = v_theta_0(s_prime)
            v_1 = v_theta_1(s_prime)
            lse = LogSumExp(v_0=v_0, v_1=v_1)

        EV = ComputeExpectedValue(s_prime=s_prime, v_theta_0=v_theta_0, v_theta_1=v_theta_1, gamma_E=gamma_E)

        # EV should be approximately lse + gamma_E
        assert torch.allclose(EV, lse + gamma_E, atol=1e-5)


class TestComputeBellmanTargets:
    """Test ComputeBellmanTargets procedure based on pseudo code."""

    def test_output_shape(self):
        """Should return Tensor[N×2]."""
        N = 10
        S = torch.linspace(0, 10, N).reshape(-1, 1)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329

        targets = ComputeBellmanTargets(S=S, v_theta_0=v_theta_0, v_theta_1=v_theta_1, beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E)

        assert targets.shape == (N, 2)

    def test_has_targets_for_both_actions(self):
        """Should have targets for action 0 and action 1."""
        N = 5
        S = torch.linspace(0, 5, N).reshape(-1, 1)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329

        targets = ComputeBellmanTargets(S=S, v_theta_0=v_theta_0, v_theta_1=v_theta_1, beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E)

        # Both columns should have valid (not nan) values
        assert torch.isfinite(targets[:, 0]).all()
        assert torch.isfinite(targets[:, 1]).all()


class TestComputeLoss:
    """Test ComputeLoss procedure based on pseudo code."""

    def test_output_type(self):
        """Should return float."""
        N = 10
        S = torch.linspace(0, 10, N).reshape(-1, 1)
        targets = torch.randn(N, 2)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        L = ComputeLoss(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert isinstance(L, torch.Tensor)
        assert L.ndim == 0  # Scalar

    def test_loss_is_non_negative(self):
        """MSE loss should be non-negative."""
        N = 10
        S = torch.linspace(0, 10, N).reshape(-1, 1)
        targets = torch.randn(N, 2)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        L = ComputeLoss(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert L.item() >= 0

    def test_zero_loss_for_perfect_predictions(self):
        """Loss should be zero when predictions match targets."""
        N = 5
        S = torch.linspace(0, 5, N).reshape(-1, 1)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        # Use network predictions as targets
        with torch.no_grad():
            pred_0 = v_theta_0(S)
            pred_1 = v_theta_1(S)
            targets = torch.cat([pred_0, pred_1], dim=1)

        L = ComputeLoss(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert torch.allclose(L, torch.tensor(0.0), atol=1e-5)


class TestCheckConvergence:
    """Test CheckConvergence procedure based on pseudo code."""

    def test_output_type(self):
        """Should return float."""
        N = 10
        S = torch.linspace(0, 10, N).reshape(-1, 1)
        targets = torch.randn(N, 2)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        max_error = CheckConvergence(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert isinstance(max_error, float)

    def test_error_is_non_negative(self):
        """Maximum absolute error should be non-negative."""
        N = 10
        S = torch.linspace(0, 10, N).reshape(-1, 1)
        targets = torch.randn(N, 2)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        max_error = CheckConvergence(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert max_error >= 0

    def test_zero_error_for_perfect_match(self):
        """Error should be zero when predictions match targets."""
        N = 5
        S = torch.linspace(0, 5, N).reshape(-1, 1)
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        # Use network predictions as targets
        with torch.no_grad():
            pred_0 = v_theta_0(S)
            pred_1 = v_theta_1(S)
            targets = torch.cat([pred_0, pred_1], dim=1)

        max_error = CheckConvergence(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert max_error < 1e-5


class TestComputeChoiceProbability:
    """Test ComputeChoiceProbability procedure based on pseudo code."""

    def test_returns_two_probabilities(self):
        """Should return tuple of (float, float)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        s = 2.5

        prob_a0, prob_a1 = ComputeChoiceProbability(s=s, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert isinstance(prob_a0, float)
        assert isinstance(prob_a1, float)

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 (logit formula property)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        s = 3.0

        prob_a0, prob_a1 = ComputeChoiceProbability(s=s, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert abs(prob_a0 + prob_a1 - 1.0) < 1e-6

    def test_probabilities_in_valid_range(self):
        """Probabilities should be in [0, 1]."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        s = 5.0

        prob_a0, prob_a1 = ComputeChoiceProbability(s=s, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        assert 0.0 <= prob_a0 <= 1.0
        assert 0.0 <= prob_a1 <= 1.0

    def test_implements_logit_formula(self):
        """Should implement P(a|s) = exp(v(s,a)) / sum(exp(v(s,a')))."""
        import numpy as np

        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        s = 4.0

        # Get probabilities from function
        prob_a0, prob_a1 = ComputeChoiceProbability(s=s, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        # Compute expected probabilities using logit formula
        s_tensor = torch.tensor([[s]], dtype=torch.float32)
        with torch.no_grad():
            v_0 = v_theta_0(s_tensor).item()
            v_1 = v_theta_1(s_tensor).item()

        denom = np.exp(v_0) + np.exp(v_1)
        expected_prob_a0 = np.exp(v_0) / denom
        expected_prob_a1 = np.exp(v_1) / denom

        assert abs(prob_a0 - expected_prob_a0) < 1e-6
        assert abs(prob_a1 - expected_prob_a1) < 1e-6

    def test_higher_value_gives_higher_probability(self):
        """Action with higher value should have higher probability."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        # Test multiple states
        for s in [1.0, 3.0, 5.0, 7.0]:
            prob_a0, prob_a1 = ComputeChoiceProbability(s=s, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

            # Get values
            s_tensor = torch.tensor([[s]], dtype=torch.float32)
            with torch.no_grad():
                v_0 = v_theta_0(s_tensor).item()
                v_1 = v_theta_1(s_tensor).item()

            # Check that higher value corresponds to higher probability
            if v_0 > v_1:
                assert prob_a0 > prob_a1
            elif v_1 > v_0:
                assert prob_a1 > prob_a0
            else:  # v_0 == v_1
                assert abs(prob_a0 - 0.5) < 1e-6
                assert abs(prob_a1 - 0.5) < 1e-6

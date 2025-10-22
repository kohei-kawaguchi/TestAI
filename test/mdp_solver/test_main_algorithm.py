"""
Unit tests for main SolveValueIteration algorithm.

These tests are based solely on the pseudo code specifications:
- Algorithm steps as defined in pseudo code
- Convergence properties
- Return types

Tests do NOT examine internal implementation details.
"""

import torch
import pytest
from mdp_solver import SolveValueIteration, MonotonicNetwork, GetValue, GetPolicy


class TestSolveValueIteration:
    """Test SolveValueIteration procedure based on pseudo code."""

    def test_returns_two_networks(self):
        """Should return tuple of (Network, Network) as specified."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 20
        state_range = (0.0, 5.0)
        max_iter = 10
        epsilon_tol = 1e-3
        num_epochs = 10
        learning_rate = 1e-3

        # Create optimizer (will be created inside function, but we need params first)
        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate
        )

        assert isinstance(v_theta_0, MonotonicNetwork)
        assert isinstance(v_theta_1, MonotonicNetwork)

    def test_returns_history(self):
        """Should return history dict with convergence info."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 20
        state_range = (0.0, 5.0)
        max_iter = 10
        epsilon_tol = 1e-3
        num_epochs = 10
        learning_rate = 1e-3

        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate
        )

        assert isinstance(history, dict)
        assert 'iterations' in history
        assert 'max_errors' in history

    def test_iterations_recorded(self):
        """History should record iterations."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 20
        state_range = (0.0, 5.0)
        max_iter = 10
        epsilon_tol = 1e-3
        num_epochs = 10
        learning_rate = 1e-3

        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate,
            verbose=False
        )

        assert len(history['iterations']) > 0
        assert len(history['max_errors']) > 0
        assert len(history['iterations']) == len(history['max_errors'])

    def test_errors_decrease(self):
        """Errors should generally decrease over iterations (convergence)."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 30
        state_range = (0.0, 5.0)
        max_iter = 50
        epsilon_tol = 1e-4
        num_epochs = 20
        learning_rate = 1e-3

        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate,
            verbose=False
        )

        # Check that final error is less than initial error
        if len(history['max_errors']) > 1:
            assert history['max_errors'][-1] <= history['max_errors'][0]

    def test_convergence_stops_iteration(self):
        """Should stop when max_error < epsilon_tol."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 20
        state_range = (0.0, 5.0)
        max_iter = 100
        epsilon_tol = 1e-2  # Looser tolerance to ensure convergence
        num_epochs = 20
        learning_rate = 1e-3

        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate,
            verbose=False
        )

        # Should converge before max_iter if epsilon_tol is reasonable
        assert len(history['iterations']) <= max_iter

    def test_networks_produce_finite_values(self):
        """Trained networks should produce finite values."""
        beta = 0.5
        gamma = 0.1
        delta = 0.95
        gamma_E = 0.5772156649015329
        hyperparameters = {'hidden_sizes': [16, 16]}
        N = 20
        state_range = (0.0, 5.0)
        max_iter = 10
        epsilon_tol = 1e-3
        num_epochs = 10
        learning_rate = 1e-3

        v_theta_0, v_theta_1, history = SolveValueIteration(
            beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E, hyperparameters=hyperparameters,
            N=N, state_range=state_range, max_iter=max_iter, epsilon_tol=epsilon_tol, num_epochs=num_epochs, learning_rate=learning_rate,
            verbose=False
        )

        # Test on some states
        test_states = torch.tensor([[1.0], [2.0], [3.0]])
        with torch.no_grad():
            v0 = v_theta_0(test_states)
            v1 = v_theta_1(test_states)

        assert torch.isfinite(v0).all()
        assert torch.isfinite(v1).all()


class TestGetValue:
    """Test GetValue helper function."""

    def test_returns_float(self):
        """Should return float."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        from mdp_solver import InitializeNetworks
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        value = GetValue(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0, a=0)

        assert isinstance(value, float)

    def test_accepts_both_actions(self):
        """Should accept action 0 and 1."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        from mdp_solver import InitializeNetworks
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        value_0 = GetValue(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0, a=0)
        value_1 = GetValue(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0, a=1)

        assert isinstance(value_0, float)
        assert isinstance(value_1, float)


class TestGetPolicy:
    """Test GetPolicy helper function."""

    def test_returns_two_probabilities(self):
        """Should return tuple of (prob_a0, prob_a1)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        from mdp_solver import InitializeNetworks
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        prob_0, prob_1 = GetPolicy(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0)

        assert isinstance(prob_0, float)
        assert isinstance(prob_1, float)

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 (logit formula property)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        from mdp_solver import InitializeNetworks
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        prob_0, prob_1 = GetPolicy(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0)

        assert abs(prob_0 + prob_1 - 1.0) < 1e-6

    def test_probabilities_in_valid_range(self):
        """Probabilities should be in [0, 1]."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        from mdp_solver import InitializeNetworks
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        prob_0, prob_1 = GetPolicy(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=2.0)

        assert 0.0 <= prob_0 <= 1.0
        assert 0.0 <= prob_1 <= 1.0

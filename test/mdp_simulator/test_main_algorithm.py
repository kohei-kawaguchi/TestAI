"""
Unit tests for main SimulateMDP algorithm.

These tests are based solely on the pseudo code specifications:
- Algorithm steps as defined in pseudo code
- Return types and shapes
- Reproducibility properties

Tests do NOT examine internal implementation details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import pytest
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks, MonotonicNetwork


class TestSimulateMDP:
    """Test SimulateMDP procedure based on pseudo code."""

    def test_returns_three_arrays(self):
        """Should return tuple of (Array[M×T], Array[M×T], Array[M×T])."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(rewards, np.ndarray)

    def test_output_shapes(self):
        """All outputs should have shape (M, T)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 15
        T = 25
        s_0 = 2.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert states.shape == (M, T)
        assert actions.shape == (M, T)
        assert rewards.shape == (M, T)

    def test_initial_state_matches_s_0(self):
        """First state should be s_0 for all paths."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 3.5
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        # All paths should start at s_0
        assert np.allclose(states[:, 0], s_0)

    def test_actions_are_binary(self):
        """Actions should be 0 or 1."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert np.all((actions == 0) | (actions == 1))

    def test_actions_are_integers(self):
        """Actions should be integer type."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert np.issubdtype(actions.dtype, np.integer)

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce identical results."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 123

        # Run simulation twice with same seed
        states1, actions1, rewards1 = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        states2, actions2, rewards2 = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert np.allclose(states1, states2)
        assert np.array_equal(actions1, actions2)
        assert np.allclose(rewards1, rewards2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0

        # Run simulation with different seeds
        states1, actions1, rewards1 = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=42
        )

        states2, actions2, rewards2 = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=999
        )

        # Should be different (very unlikely to be identical)
        assert not np.array_equal(actions1, actions2)

    def test_state_transition_follows_formula(self):
        """State should evolve according to s' = (1-gamma)*s + a."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 5
        T = 10
        s_0 = 2.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        # Check state transitions for each path
        for m in range(M):
            for t in range(T - 1):
                s_t = states[m, t]
                a_t = actions[m, t]
                s_next = states[m, t + 1]

                expected_s_next = (1 - gamma) * s_t + a_t
                assert np.isclose(s_next, expected_s_next, atol=1e-6)

    def test_reward_follows_formula(self):
        """Reward should follow r = beta*log(1+s) - a."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 5
        T = 10
        s_0 = 2.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        # Check rewards for each state-action pair
        for m in range(M):
            for t in range(T):
                s_t = states[m, t]
                a_t = actions[m, t]
                r_t = rewards[m, t]

                expected_r_t = beta * np.log(1 + s_t) - a_t
                assert np.isclose(r_t, expected_r_t, atol=1e-6)

    def test_states_are_non_negative(self):
        """States should remain non-negative for s_0 >= 0."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert np.all(states >= 0)

    def test_accepts_all_parameters(self):
        """Should accept all parameters as specified in pseudo code."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

        # Use different parameter values to ensure they're all used
        beta = 0.8
        gamma = 0.15
        delta = 0.9
        M = 8
        T = 15
        s_0 = 1.5
        seed = 999

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        # Should complete without error
        assert states.shape == (M, T)

    def test_single_path_single_period(self):
        """Should work with M=1, T=1."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 1
        T = 1
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert states.shape == (1, 1)
        assert actions.shape == (1, 1)
        assert rewards.shape == (1, 1)
        assert states[0, 0] == s_0

    def test_many_paths_many_periods(self):
        """Should work with large M and T."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 100
        T = 100
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert states.shape == (M, T)
        assert actions.shape == (M, T)
        assert rewards.shape == (M, T)

    def test_all_values_are_finite(self):
        """All returned values should be finite (no inf/nan)."""
        hyperparameters = {'hidden_sizes': [16, 16]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta = 1.0
        gamma = 0.1
        delta = 0.95
        M = 10
        T = 20
        s_0 = 1.0
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            M=M,
            T=T,
            s_0=s_0,
            seed=seed
        )

        assert np.isfinite(states).all()
        assert np.isfinite(actions).all()
        assert np.isfinite(rewards).all()

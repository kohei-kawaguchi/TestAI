"""
Unit tests for main EstimateMDP algorithm.

These tests are based solely on the pseudo code specifications:
- Algorithm steps as defined in pseudo code
- Return types and shapes
- Parameter recovery properties

Tests do NOT examine internal implementation details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import unittest
from mdp_estimator import EstimateMDP, EstimateGamma, EstimateBeta
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks


class TestEstimateMDP(unittest.TestCase):
    """Test EstimateMDP procedure based on pseudo code."""

    def test_returns_four_values(self):
        """Should return tuple of (float, float, Array[K], Array[K])."""
        # Generate small test data
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        # Estimation parameters
        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.8, 1.0, 1.2])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
            states=states,
            actions=actions,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        assert isinstance(gamma_hat, float)
        assert isinstance(beta_hat, float)
        assert isinstance(beta_grid_out, np.ndarray)
        assert isinstance(distances, np.ndarray)

    def test_beta_grid_matches_input(self):
        """Returned beta_grid should match input beta_grid."""
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.7, 0.9, 1.1, 1.3])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
            states=states,
            actions=actions,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        np.testing.assert_array_equal(beta_grid_out, beta_grid)

    def test_distances_shape_matches_beta_grid(self):
        """Distances array should have same length as beta_grid."""
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
            states=states,
            actions=actions,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        assert len(distances) == len(beta_grid)

    def test_beta_hat_in_beta_grid(self):
        """Estimated beta should be one of the candidates in beta_grid."""
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.8, 1.0, 1.2])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
            states=states,
            actions=actions,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        assert beta_hat in beta_grid

    def test_distances_all_positive(self):
        """All distances should be non-negative."""
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.8, 1.0, 1.2])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
            states=states,
            actions=actions,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        assert np.all(distances >= 0)


class TestEstimateGamma(unittest.TestCase):
    """Test EstimateGamma procedure based on pseudo code."""

    def test_returns_float(self):
        """Should return a float value."""
        # Simple test data
        states = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
        actions = np.array([[1, 1, 0], [0, 1, 1]])

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert isinstance(gamma_hat, (float, np.floating))

    def test_gamma_in_valid_range(self):
        """Estimated gamma should be in valid range [0, 1]."""
        states = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
        actions = np.array([[1, 1, 0], [0, 1, 1]])

        gamma_hat = EstimateGamma(states=states, actions=actions)

        assert 0 <= gamma_hat <= 1


class TestEstimateBeta(unittest.TestCase):
    """Test EstimateBeta procedure based on pseudo code."""

    def test_returns_tuple_of_beta_and_distances(self):
        """Should return tuple of (float, Array[K])."""
        hyperparameters = {'hidden_sizes': [8, 8]}
        v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)
        beta_true = 1.0
        gamma_true = 0.1
        delta = 0.95
        M = 50
        T = 30
        state_range = (0.0, 5.0)
        seed = 42

        states, actions, rewards = SimulateMDP(
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta,
            M=M,
            T=T,
            state_range=state_range,
            seed=seed
        )

        N = 20
        state_range = (0.0, 5.0)
        num_epochs = 10
        learning_rate = 1e-3
        beta_grid = np.array([0.8, 1.0, 1.2])
        epsilon_tol = 1e-2
        max_iter = 5
        gamma_E = 0.5772156649015329

        beta_hat, distances = EstimateBeta(
            states=states,
            actions=actions,
            gamma=gamma_true,
            delta=delta,
            N=N,
            state_range=state_range,
            hyperparameters=hyperparameters,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta_grid=beta_grid,
            epsilon_tol=epsilon_tol,
            max_iter=max_iter,
            gamma_E=gamma_E
        )

        assert isinstance(beta_hat, (float, np.floating))
        assert isinstance(distances, np.ndarray)
        assert len(distances) == len(beta_grid)

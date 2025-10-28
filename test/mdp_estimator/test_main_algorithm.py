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
import pytest
from mdp_estimator import EstimateMDP, EstimateGamma, EstimateBeta
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks


@pytest.fixture
def simple_simulation():
    """Generate small test data for estimation."""
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

    return states, actions, hyperparameters


def test_estimate_mdp_returns_four_values(simple_simulation):
    """Should return tuple of (float, float, Array[K], Array[K])."""
    states, actions, hyperparameters = simple_simulation

    gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
        states=states,
        actions=actions,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=np.array([0.8, 1.0, 1.2]),
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    assert isinstance(gamma_hat, float)
    assert isinstance(beta_hat, float)
    assert isinstance(beta_grid_out, np.ndarray)
    assert isinstance(distances, np.ndarray)


def test_estimate_mdp_beta_grid_matches_input(simple_simulation):
    """Returned beta_grid should match input beta_grid."""
    states, actions, hyperparameters = simple_simulation
    beta_grid = np.array([0.7, 0.9, 1.1, 1.3])

    gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
        states=states,
        actions=actions,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=beta_grid,
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    np.testing.assert_array_equal(beta_grid_out, beta_grid)


def test_estimate_mdp_distances_shape_matches_beta_grid(simple_simulation):
    """Distances array should have same length as beta_grid."""
    states, actions, hyperparameters = simple_simulation
    beta_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

    gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
        states=states,
        actions=actions,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=beta_grid,
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    assert len(distances) == len(beta_grid)


def test_estimate_mdp_beta_hat_in_beta_grid(simple_simulation):
    """Estimated beta should be one of the candidates in beta_grid."""
    states, actions, hyperparameters = simple_simulation
    beta_grid = np.array([0.8, 1.0, 1.2])

    gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
        states=states,
        actions=actions,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=beta_grid,
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    assert beta_hat in beta_grid


def test_estimate_mdp_distances_all_positive(simple_simulation):
    """All distances should be non-negative."""
    states, actions, hyperparameters = simple_simulation

    gamma_hat, beta_hat, beta_grid_out, distances = EstimateMDP(
        states=states,
        actions=actions,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=np.array([0.8, 1.0, 1.2]),
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    assert np.all(distances >= 0)


def test_estimate_gamma_returns_float():
    """Should return a float value."""
    states = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    actions = np.array([[1, 1, 0], [0, 1, 1]])

    gamma_hat = EstimateGamma(states=states, actions=actions)

    assert isinstance(gamma_hat, (float, np.floating))


def test_estimate_gamma_in_valid_range():
    """Estimated gamma should be in valid range [0, 1]."""
    states = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    actions = np.array([[1, 1, 0], [0, 1, 1]])

    gamma_hat = EstimateGamma(states=states, actions=actions)

    assert 0 <= gamma_hat <= 1


def test_estimate_beta_returns_tuple(simple_simulation):
    """Should return tuple of (float, Array[K])."""
    states, actions, hyperparameters = simple_simulation

    beta_hat, distances = EstimateBeta(
        states=states,
        actions=actions,
        gamma=0.1,
        delta=0.95,
        N=20,
        state_range=(0.0, 5.0),
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3,
        beta_grid=np.array([0.8, 1.0, 1.2]),
        epsilon_tol=1e-2,
        max_iter=5,
        gamma_E=0.5772156649015329
    )

    assert isinstance(beta_hat, (float, np.floating))
    assert isinstance(distances, np.ndarray)
    assert len(distances) == 3

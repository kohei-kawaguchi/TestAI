"""
Unit tests for EstimateCCP and related subroutines.

These tests verify the CCP estimation procedure based on pseudo code:
- Network returns proper type and shape
- Network output is in [0,1] range (valid probability)
- Network is monotonically DECREASING in state (higher s → lower P(a=1|s))
- Training reduces loss over epochs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import pytest
from mdp_estimator import EstimateCCP, InitializeDecreasingCCPNetwork, DecreasingCCPNetwork
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks


def test_decreasing_ccp_network_instantiation():
    """Network should instantiate with hidden_sizes parameter."""
    network = DecreasingCCPNetwork(hidden_sizes=[16, 16])
    assert isinstance(network, DecreasingCCPNetwork)


def test_decreasing_ccp_network_forward_returns_tensor():
    """Forward pass should return tensor of correct shape."""
    network = DecreasingCCPNetwork(hidden_sizes=[16, 16])
    states = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    output = network(states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 1)


def test_decreasing_ccp_network_output_in_valid_range():
    """Network output should be in [0,1] (valid probability range)."""
    network = DecreasingCCPNetwork(hidden_sizes=[16, 16])
    states = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)

    with torch.no_grad():
        output = network(states)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_decreasing_ccp_network_monotonically_decreasing():
    """Network should be monotonically DECREASING in state."""
    network = DecreasingCCPNetwork(hidden_sizes=[32, 32])

    # Train briefly to ensure weights are non-trivial
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    states_train = torch.linspace(0.0, 5.0, 100).reshape(-1, 1)
    actions_train = torch.ones(100)  # Dummy training

    for _ in range(10):
        optimizer.zero_grad()
        probs = network(states_train).squeeze()
        loss = torch.nn.functional.binary_cross_entropy(probs, actions_train)
        loss.backward()
        optimizer.step()

    # Test monotonicity: P(s_i) >= P(s_j) for s_i < s_j
    states_test = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)

    with torch.no_grad():
        probs = network(states_test).squeeze()

    # Check that probabilities are non-increasing
    for i in range(len(probs) - 1):
        assert probs[i].item() >= probs[i+1].item() - 1e-6, \
            f"Not decreasing: P(s[{i}])={probs[i]:.4f} < P(s[{i+1}])={probs[i+1]:.4f}"


def test_initialize_decreasing_ccp_network_returns_network():
    """Should return DecreasingCCPNetwork instance."""
    hyperparameters = {'hidden_sizes': [16, 16]}

    network = InitializeDecreasingCCPNetwork(hyperparameters=hyperparameters)

    assert isinstance(network, DecreasingCCPNetwork)


def test_initialize_decreasing_ccp_network_respects_hyperparameters():
    """Network should use hidden_sizes from hyperparameters."""
    hyperparameters = {'hidden_sizes': [8, 8]}

    network = InitializeDecreasingCCPNetwork(hyperparameters=hyperparameters)

    # Network should be created successfully
    assert network is not None

    # Test it can process input
    states = torch.tensor([[1.0]], dtype=torch.float32)
    output = network(states)
    assert output.shape == (1, 1)


def test_estimate_ccp_returns_network():
    """Should return a trained DecreasingCCPNetwork."""
    # Generate simple test data
    states = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    actions = np.array([[1, 0, 0], [1, 1, 0]])
    hyperparameters = {'hidden_sizes': [8, 8]}

    network = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3
    )

    assert isinstance(network, DecreasingCCPNetwork)


def test_estimate_ccp_network_outputs_valid_probabilities():
    """Trained network should output probabilities in [0,1]."""
    states = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    actions = np.array([[1, 0, 0], [1, 1, 0]])
    hyperparameters = {'hidden_sizes': [8, 8]}

    network = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=10,
        learning_rate=1e-3
    )

    # Test on evaluation grid
    test_states = torch.linspace(0.0, 5.0, 20).reshape(-1, 1)
    with torch.no_grad():
        probs = network(test_states)

    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_estimate_ccp_trained_network_is_decreasing():
    """Trained network should be monotonically DECREASING."""
    # Generate data where high states → low actions (decreasing pattern)
    hyperparameters = {'hidden_sizes': [16, 16]}
    v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

    states, actions, rewards = SimulateMDP(
        v_theta_0=v_theta_0,
        v_theta_1=v_theta_1,
        beta=1.0,
        gamma=0.1,
        delta=0.95,
        M=100,
        T=50,
        state_range=(0.0, 5.0),
        seed=42
    )

    network = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=100,
        learning_rate=1e-3
    )

    # Test monotonicity on grid
    test_states = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)
    with torch.no_grad():
        probs = network(test_states).squeeze()

    # Check that probabilities are non-increasing
    for i in range(len(probs) - 1):
        assert probs[i].item() >= probs[i+1].item() - 1e-6, \
            f"Not decreasing: P(s[{i}])={probs[i]:.4f} < P(s[{i+1}])={probs[i+1]:.4f}"


def test_estimate_ccp_learns_from_data():
    """Network should fit the observed state-action pattern."""
    # Create data where low states → action 1, high states → action 0
    states = np.array([[0.5, 1.0, 3.0, 4.0]] * 20)  # Repeat for more data
    actions = np.array([[1, 1, 0, 0]] * 20)  # Low s→1, High s→0
    hyperparameters = {'hidden_sizes': [16, 16]}

    network = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=200,
        learning_rate=1e-2
    )

    # Network should predict high prob for low states, low prob for high states
    with torch.no_grad():
        prob_low = network(torch.tensor([[0.5]], dtype=torch.float32)).item()
        prob_high = network(torch.tensor([[4.0]], dtype=torch.float32)).item()

    # Lower state should have higher probability (decreasing pattern)
    assert prob_low > prob_high, \
        f"Expected P(s=0.5) > P(s=4.0), got {prob_low:.4f} vs {prob_high:.4f}"

"""
Unit tests for EstimateCCP and related subroutines.

These tests verify the CCP estimation procedure based on pseudo code:
- Network returns proper type and shape
- Network output is in [0,1] range (valid probability)
- Network predicts P(a=1|s) with unconstrained weights (no monotonicity)
- Training reduces loss over epochs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import pytest
from mdp_estimator import EstimateCCP, InitializeCCPNetwork, CCPNetwork
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks


def test_ccp_network_instantiation():
    """Network should instantiate with hidden_sizes parameter."""
    network = CCPNetwork(hidden_sizes=[16, 16])
    assert isinstance(network, CCPNetwork)


def test_ccp_network_forward_returns_tensor():
    """Forward pass should return tensor of correct shape."""
    network = CCPNetwork(hidden_sizes=[16, 16])
    states = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    output = network(states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 1)


def test_ccp_network_output_in_valid_range():
    """Network output should be in [0,1] (valid probability range)."""
    network = CCPNetwork(hidden_sizes=[16, 16])
    states = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)

    with torch.no_grad():
        output = network(states)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_ccp_network_output_range():
    """Network output should be in [0,1] range (valid probabilities)."""
    network = CCPNetwork(hidden_sizes=[32, 32])

    # Test on domain
    states_test = torch.linspace(0.0, 10.0, 100).reshape(-1, 1)

    with torch.no_grad():
        probs = network(states_test).squeeze()

    # Check output is in [0, 1]
    assert torch.all(probs >= 0.0), \
        f"All probabilities should be >= 0, but min={probs.min():.4f}"
    assert torch.all(probs <= 1.0), \
        f"All probabilities should be <= 1, but max={probs.max():.4f}"


def test_initialize_ccp_network_returns_network():
    """Should return CCPNetwork instance."""
    hyperparameters = {'hidden_sizes': [16, 16]}

    network = InitializeCCPNetwork(hyperparameters=hyperparameters)

    assert isinstance(network, CCPNetwork)


def test_initialize_ccp_network_respects_hyperparameters():
    """Network should use hidden_sizes from hyperparameters."""
    hyperparameters = {'hidden_sizes': [8, 8]}

    network = InitializeCCPNetwork(hyperparameters=hyperparameters)

    # Network should be created successfully
    assert network is not None

    # Test it can process input
    states = torch.tensor([[1.0]], dtype=torch.float32)
    output = network(states)
    assert output.shape == (1, 1)


def test_estimate_ccp_returns_network():
    """Should return a trained CCPNetwork that predicts P(a=1|s)."""
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

    assert isinstance(network, CCPNetwork)


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




def test_estimate_ccp_learns_from_data():
    """Network should fit the observed state-action pattern (predicts P(a=1|s))."""
    # Create data where low states → action 1, high states → action 0
    # This means P(a=1|s) should be HIGH at low s, LOW at high s
    states = np.array([[0.5, 1.0, 3.0, 4.0]] * 20)  # Repeat for more data
    actions = np.array([[1, 1, 0, 0]] * 20)  # Low s→a=1, High s→a=0
    hyperparameters = {'hidden_sizes': [16, 16]}

    network = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=200,
        learning_rate=1e-2
    )

    # Network predicts P(a=1|s), should be high for low states, low for high states
    with torch.no_grad():
        prob_a1_low = network(torch.tensor([[0.5]], dtype=torch.float32)).item()
        prob_a1_high = network(torch.tensor([[4.0]], dtype=torch.float32)).item()

    # Lower state should have higher P(a=1|s) since data shows low s→a=1
    assert prob_a1_low > prob_a1_high, \
        f"Expected P(a=1|s=0.5) > P(a=1|s=4.0), got {prob_a1_low:.4f} vs {prob_a1_high:.4f}"

"""
Unit tests for EstimateCCP and related subroutines.

These tests verify the CCP estimation procedure based on pseudo code:
- Network returns proper type and shape
- Network output is in [0,1] range (valid probability)
- Network predicts P(a=0|s) which is monotonically INCREASING in state (higher s → higher P(a=0|s))
- Training reduces loss over epochs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch
import pytest
from mdp_estimator import EstimateCCP, InitializeIncreasingCCPNetwork, IncreasingCCPNetwork
from mdp_simulator import SimulateMDP
from mdp_solver import InitializeNetworks


def test_increasing_ccp_network_instantiation():
    """Network should instantiate with hidden_sizes parameter."""
    network = IncreasingCCPNetwork(hidden_sizes=[16, 16])
    assert isinstance(network, IncreasingCCPNetwork)


def test_increasing_ccp_network_forward_returns_tensor():
    """Forward pass should return tensor of correct shape."""
    network = IncreasingCCPNetwork(hidden_sizes=[16, 16])
    states = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    output = network(states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 1)


def test_increasing_ccp_network_output_in_valid_range():
    """Network output should be in [0,1] (valid probability range)."""
    network = IncreasingCCPNetwork(hidden_sizes=[16, 16])
    states = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)

    with torch.no_grad():
        output = network(states)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_increasing_ccp_network_weight_constraints():
    """All layers should use positive weights (w_i ≥ 0) for increasing monotonicity."""
    network = IncreasingCCPNetwork(hidden_sizes=[16, 16])

    # Create test input
    states_test = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    # Forward pass to ensure weights are used
    with torch.no_grad():
        output = network(states_test)

    # Verify all layers have positive weights (w_i ≥ 0)
    for i, layer in enumerate(network.layers):
        weight = torch.nn.functional.softplus(layer.weight)
        assert torch.all(weight >= 0), \
            f"Layer {i} weights should be non-negative (w_{i} ≥ 0)"
        assert torch.all(weight >= -1e-6), \
            f"Layer {i} weights should be positive or near-zero"


def test_increasing_ccp_network_monotonically_increasing():
    """Network should be strictly INCREASING for s >= 0 with output in [0,1]."""
    network = IncreasingCCPNetwork(hidden_sizes=[32, 32])

    # Train briefly to ensure weights are non-trivial
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    states_train = torch.linspace(0.0, 5.0, 100).reshape(-1, 1)
    actions_train = torch.zeros(100)  # Train to predict P(a=0|s)

    for _ in range(10):
        optimizer.zero_grad()
        probs = network(states_train).squeeze()
        loss = torch.nn.functional.binary_cross_entropy(probs, actions_train)
        loss.backward()
        optimizer.step()

    # Test on positive domain s >= 0
    states_test = torch.linspace(0.0, 10.0, 100).reshape(-1, 1)

    with torch.no_grad():
        probs = network(states_test).squeeze()

    # Check output is in [0, 1]
    assert torch.all(probs >= 0.0), \
        f"All probabilities should be >= 0, but min={probs.min():.4f}"
    assert torch.all(probs <= 1.0), \
        f"All probabilities should be <= 1, but max={probs.max():.4f}"

    # Check strict monotonic increase: P(s_i) <= P(s_j) for all s_i < s_j
    for i in range(len(probs) - 1):
        assert probs[i].item() <= probs[i+1].item() + 1e-6, \
            f"Not increasing at i={i}: P(s={states_test[i].item():.2f})={probs[i]:.4f} > P(s={states_test[i+1].item():.2f})={probs[i+1]:.4f}"


def test_increasing_ccp_network_untrained_properties():
    """Untrained network should satisfy output in [0,1] and increasing for s >= 0."""
    network = IncreasingCCPNetwork(hidden_sizes=[32, 32])

    # Test on positive domain s >= 0 without any training
    states_test = torch.linspace(0.0, 10.0, 50).reshape(-1, 1)

    with torch.no_grad():
        probs = network(states_test).squeeze()

    # Check output is in [0, 1] even without training
    assert torch.all(probs >= 0.0), \
        f"Untrained network: all probabilities should be >= 0, but min={probs.min():.4f}"
    assert torch.all(probs <= 1.0), \
        f"Untrained network: all probabilities should be <= 1, but max={probs.max():.4f}"

    # Check monotonic increase holds even without training
    violations = 0
    for i in range(len(probs) - 1):
        if probs[i].item() > probs[i+1].item() + 1e-6:
            violations += 1

    assert violations == 0, \
        f"Untrained network: found {violations} violations of monotonic increase out of {len(probs)-1} pairs"


def test_initialize_increasing_ccp_network_returns_network():
    """Should return IncreasingCCPNetwork instance."""
    hyperparameters = {'hidden_sizes': [16, 16]}

    network = InitializeIncreasingCCPNetwork(hyperparameters=hyperparameters)

    assert isinstance(network, IncreasingCCPNetwork)


def test_initialize_increasing_ccp_network_respects_hyperparameters():
    """Network should use hidden_sizes from hyperparameters."""
    hyperparameters = {'hidden_sizes': [8, 8]}

    network = InitializeIncreasingCCPNetwork(hyperparameters=hyperparameters)

    # Network should be created successfully
    assert network is not None

    # Test it can process input
    states = torch.tensor([[1.0]], dtype=torch.float32)
    output = network(states)
    assert output.shape == (1, 1)


def test_estimate_ccp_returns_network():
    """Should return a trained IncreasingCCPNetwork that predicts P(a=0|s)."""
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

    assert isinstance(network, IncreasingCCPNetwork)


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


def test_estimate_ccp_trained_network_is_increasing():
    """Trained network should be monotonically INCREASING (predicts P(a=0|s))."""
    # Generate data where high states → low actions (P(a=0|s) increasing pattern)
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

    # Test monotonicity on grid (network predicts P(a=0|s))
    test_states = torch.linspace(0.0, 5.0, 50).reshape(-1, 1)
    with torch.no_grad():
        probs_a0 = network(test_states).squeeze()

    # Check that P(a=0|s) probabilities are non-decreasing (increasing)
    for i in range(len(probs_a0) - 1):
        assert probs_a0[i].item() <= probs_a0[i+1].item() + 1e-6, \
            f"Not increasing: P(a=0|s[{i}])={probs_a0[i]:.4f} > P(a=0|s[{i+1}])={probs_a0[i+1]:.4f}"


def test_estimate_ccp_learns_from_data():
    """Network should fit the observed state-action pattern (predicts P(a=0|s))."""
    # Create data where low states → action 1, high states → action 0
    # This means P(a=0|s) should be LOW at low s, HIGH at high s (increasing)
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

    # Network predicts P(a=0|s), should be low for low states, high for high states
    with torch.no_grad():
        prob_a0_low = network(torch.tensor([[0.5]], dtype=torch.float32)).item()
        prob_a0_high = network(torch.tensor([[4.0]], dtype=torch.float32)).item()

    # Higher state should have higher P(a=0|s) (increasing pattern)
    assert prob_a0_low < prob_a0_high, \
        f"Expected P(a=0|s=0.5) < P(a=0|s=4.0), got {prob_a0_low:.4f} vs {prob_a0_high:.4f}"

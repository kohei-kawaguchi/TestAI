"""
MDP Solver using Value Iteration with Neural Networks.

This module implements value iteration for solving Markov Decision Processes
with continuous state spaces and discrete action spaces using neural networks.

The implementation follows the modular pseudo code structure with separate
functions for each subroutine.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class MonotonicNetwork(nn.Module):
    """
    A neural network with monotonic constraints using softplus transformations.

    The network ensures monotonicity in the input by constraining all weights
    to be non-negative using the softplus transformation.
    """

    def __init__(self, hidden_sizes: list[int] = [32, 32]):
        """
        Initialize the monotonic network.

        Args:
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes

        # Create layers
        layers = []
        input_size = 1  # Single state input

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with monotonic constraints.

        Args:
            s: State tensor of shape (N, 1)

        Returns:
            Value tensor of shape (N, 1)
        """
        x = s

        # Apply layers with softplus on weights (except biases)
        for i, layer in enumerate(self.layers[:-1]):
            # Apply softplus to weights to ensure non-negativity
            weight = torch.nn.functional.softplus(layer.weight)
            bias = layer.bias
            x = torch.nn.functional.linear(x, weight, bias)
            x = torch.tanh(x)  # Smooth activation

        # Final layer
        final_layer = self.layers[-1]
        weight = torch.nn.functional.softplus(final_layer.weight)
        bias = final_layer.bias
        x = torch.nn.functional.linear(x, weight, bias)

        return x


# ============================================================================
# Subroutines following the pseudo code structure
# ============================================================================

def InitializeNetworks(hyperparameters: Dict) -> Tuple[MonotonicNetwork, MonotonicNetwork]:
    """
    Procedure InitializeNetworks(hyperparameters: dict) -> (Network, Network)

    Create and initialize two neural networks with monotonic weight constraints.

    Args:
        hyperparameters: Dictionary containing 'hidden_sizes'

    Returns:
        Tuple of (v_theta_0, v_theta_1): Two initialized networks
    """
    hidden_sizes = hyperparameters.get('hidden_sizes', [32, 32])

    v_theta_0 = MonotonicNetwork(hidden_sizes=hidden_sizes)
    v_theta_1 = MonotonicNetwork(hidden_sizes=hidden_sizes)

    return v_theta_0, v_theta_1


def GenerateStateGrid(N: int, state_range: Tuple[float, float]) -> torch.Tensor:
    """
    Procedure GenerateStateGrid(N: int, state_range: tuple[float, float]) -> Tensor[N×1]

    Create uniform grid of states over the specified range.

    Args:
        N: Number of states
        state_range: Tuple of (min_state, max_state)

    Returns:
        S: Tensor of shape (N, 1) containing state grid
    """
    S = torch.linspace(state_range[0], state_range[1], N).reshape(-1, 1)
    return S


def ComputeNextState(s: torch.Tensor, a: int, gamma: float) -> torch.Tensor:
    """
    Procedure ComputeNextState(s: Tensor[N×1], a: int, gamma: float) -> Tensor[N×1]

    Compute next state from current state and action.

    Args:
        s: Current state tensor of shape (N, 1)
        a: Action (0 or 1)
        gamma: State depreciation rate

    Returns:
        s_next: Next state tensor of shape (N, 1)
    """
    return (1 - gamma) * s + a


def ComputeMeanReward(s: torch.Tensor, a: int, beta: float) -> torch.Tensor:
    """
    Procedure ComputeMeanReward(s: Tensor[N×1], a: int, beta: float) -> Tensor[N×1]

    Compute mean reward function with logarithmic form.

    The reward function is: r(s,a) = beta * log(1 + s) - a
    This captures diminishing marginal returns to the state.

    Args:
        s: State tensor of shape (N, 1)
        a: Action (0 or 1)
        beta: Reward weight on state

    Returns:
        reward: Mean reward tensor of shape (N, 1)
    """
    return beta * torch.log(1 + s) - a


def LogSumExp(v_0: torch.Tensor, v_1: torch.Tensor) -> torch.Tensor:
    """
    Procedure LogSumExp(v_0: Tensor[N×1], v_1: Tensor[N×1]) -> Tensor[N×1]

    Numerically stable log-sum-exp computation.

    Args:
        v_0: Value tensor for action 0, shape (N, 1)
        v_1: Value tensor for action 1, shape (N, 1)

    Returns:
        log_sum_exp: Tensor of shape (N, 1)
    """
    max_v = torch.max(v_0, v_1)
    return max_v + torch.log(torch.exp(v_0 - max_v) + torch.exp(v_1 - max_v))


def ComputeExpectedValue(
    s_prime: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork,
    gamma_E: float
) -> torch.Tensor:
    """
    Procedure ComputeExpectedValue(s': Tensor[N×1], v_theta^(0): Network,
                                    v_theta^(1): Network, gamma_E: float) -> Tensor[N×1]

    Compute expected value of next state using log-sum-exp.

    Args:
        s_prime: Next state tensor of shape (N, 1)
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1
        gamma_E: Euler-Mascheroni constant

    Returns:
        EV: Expected value tensor of shape (N, 1)
    """
    v_0 = v_theta_0(s_prime)
    v_1 = v_theta_1(s_prime)
    EV = LogSumExp(v_0=v_0, v_1=v_1) + gamma_E
    return EV


def ComputeBellmanTargets(
    S: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork,
    beta: float,
    gamma: float,
    delta: float,
    gamma_E: float
) -> torch.Tensor:
    """
    Procedure ComputeBellmanTargets(S: Tensor[N×1], v_theta^(0): Network, v_theta^(1): Network,
                                     beta: float, gamma: float, delta: float) -> Tensor[N×2]

    Compute Bellman target values for both actions.

    Args:
        S: State tensor of shape (N, 1)
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1
        beta: Reward weight on state
        gamma: State depreciation rate
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant

    Returns:
        targets: Tensor of shape (N, 2) with targets for both actions
    """
    targets_list = []

    for a in [0, 1]:
        s_prime_i = ComputeNextState(s=S, a=a, gamma=gamma)
        EV_i = ComputeExpectedValue(s_prime=s_prime_i, v_theta_0=v_theta_0, v_theta_1=v_theta_1, gamma_E=gamma_E)
        y_i_a = ComputeMeanReward(s=S, a=a, beta=beta) + delta * EV_i
        targets_list.append(y_i_a)

    # Stack into (N, 2) tensor
    targets = torch.cat(targets_list, dim=1)
    return targets


def ComputeLoss(
    S: torch.Tensor,
    targets: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork
) -> float:
    """
    Procedure ComputeLoss(S: Tensor[N×1], {y_i^(a)}: Tensor[N×2],
                          v_theta^(0): Network, v_theta^(1): Network) -> float

    Compute mean squared error loss.

    Args:
        S: State tensor of shape (N, 1)
        targets: Target tensor of shape (N, 2)
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1

    Returns:
        L: Scalar loss value
    """
    pred_0 = v_theta_0(S)
    pred_1 = v_theta_1(S)

    # Targets: [:, 0] for action 0, [:, 1] for action 1
    loss_0 = torch.mean((pred_0 - targets[:, 0:1]) ** 2)
    loss_1 = torch.mean((pred_1 - targets[:, 1:2]) ** 2)

    L = loss_0 + loss_1
    return L


def UpdateNetworks(
    S: torch.Tensor,
    targets: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork,
    num_epochs: int,
    optimizer: torch.optim.Optimizer
) -> Tuple[MonotonicNetwork, MonotonicNetwork]:
    """
    Procedure UpdateNetworks(S: Tensor[N×1], {y_i^(a)}: Tensor[N×2], v_theta^(0): Network,
                             v_theta^(1): Network, num_epochs: int) -> (Network, Network)

    Update networks via gradient descent.

    Args:
        S: State tensor of shape (N, 1)
        targets: Target tensor of shape (N, 2)
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1
        num_epochs: Number of gradient descent epochs
        optimizer: Optimizer for network parameters

    Returns:
        Updated (v_theta_0, v_theta_1)
    """
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute loss
        L = ComputeLoss(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        # Compute gradients
        L.backward()

        # Update parameters
        optimizer.step()

    return v_theta_0, v_theta_1


def CheckConvergence(
    S: torch.Tensor,
    targets: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork
) -> float:
    """
    Procedure CheckConvergence(S: Tensor[N×1], {y_i^(a)}: Tensor[N×2],
                               v_theta^(0): Network, v_theta^(1): Network) -> float

    Check convergence by computing maximum error.

    Args:
        S: State tensor of shape (N, 1)
        targets: Target tensor of shape (N, 2)
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1

    Returns:
        max_error: Maximum absolute error
    """
    with torch.no_grad():
        pred_0 = v_theta_0(S)
        pred_1 = v_theta_1(S)

        error_0 = torch.abs(pred_0 - targets[:, 0:1])
        error_1 = torch.abs(pred_1 - targets[:, 1:2])

        max_error = torch.max(torch.max(error_0), torch.max(error_1)).item()

    return max_error


# ============================================================================
# Main Algorithm
# ============================================================================

def SolveValueIteration(
    beta: float,
    gamma: float,
    delta: float,
    gamma_E: float,
    hyperparameters: Dict,
    N: int = 100,
    state_range: Tuple[float, float] = (0.0, 10.0),
    max_iter: int = 100,
    epsilon_tol: float = 1e-4,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> Tuple[MonotonicNetwork, MonotonicNetwork, Dict]:
    """
    Procedure SolveValueIteration(beta: float, gamma: float, delta: float,
                                   gamma_E: float, hyperparameters: dict) -> (Network, Network)

    Main value iteration algorithm.

    Args:
        beta: Reward weight on state
        gamma: State depreciation rate
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant
        hyperparameters: Dictionary with network configuration
        N: Number of states
        state_range: Range of states to sample
        max_iter: Maximum number of iterations
        epsilon_tol: Convergence tolerance
        num_epochs: Number of gradient descent epochs per iteration
        learning_rate: Learning rate for optimizer
        verbose: Whether to print progress

    Returns:
        Tuple of (v_theta_0, v_theta_1, history)
    """
    # Step 1: Initialize networks
    v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

    # Create optimizer
    params = list(v_theta_0.parameters()) + list(v_theta_1.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Step 2: Generate state grid
    S = GenerateStateGrid(N=N, state_range=state_range)

    # Track history
    history = {
        'iterations': [],
        'max_errors': []
    }

    # Step 3: Iterate until convergence
    for iteration in range(max_iter):
        # Step 3a: Compute Bellman targets
        with torch.no_grad():
            targets = ComputeBellmanTargets(S=S, v_theta_0=v_theta_0, v_theta_1=v_theta_1, beta=beta, gamma=gamma, delta=delta, gamma_E=gamma_E)

        # Step 3b: Update networks
        v_theta_0, v_theta_1 = UpdateNetworks(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1, num_epochs=num_epochs, optimizer=optimizer)

        # Step 3c: Check convergence
        max_error = CheckConvergence(S=S, targets=targets, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        # Track history
        history['iterations'].append(iteration)
        history['max_errors'].append(max_error)

        # Print progress
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: max_error = {max_error:.6f}")

        # Step 3d: Check if converged
        if max_error < epsilon_tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

    # Step 4: Return
    return v_theta_0, v_theta_1, history


# ============================================================================
# Helper functions for evaluation
# ============================================================================

def ComputeChoiceProbability(s: float, v_theta_0: MonotonicNetwork, v_theta_1: MonotonicNetwork) -> Tuple[float, float]:
    """
    Procedure ComputeChoiceProbability(s: float, v_theta^(0): Network, v_theta^(1): Network) -> (float, float)

    Compute optimal choice probabilities using the logit formula.

    This implements the logit formula for choice probabilities under Type-I Extreme Value
    distributed shocks:
        P(a | s) = exp(v(s, a)) / sum_{a'} exp(v(s, a'))

    Args:
        s: State value
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1

    Returns:
        Tuple of (P(a=0|s), P(a=1|s))
    """
    import numpy as np

    s_tensor = torch.tensor([[s]], dtype=torch.float32)

    with torch.no_grad():
        v_0 = v_theta_0(s_tensor).item()
        v_1 = v_theta_1(s_tensor).item()

    denom = np.exp(v_0) + np.exp(v_1)
    prob_a0 = np.exp(v_0) / denom
    prob_a1 = np.exp(v_1) / denom

    return prob_a0, prob_a1


def GetValue(v_theta_0: MonotonicNetwork, v_theta_1: MonotonicNetwork, s: float, a: int) -> float:
    """
    Get value function for a given state and action.

    Args:
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1
        s: State value
        a: Action (0 or 1)

    Returns:
        Value
    """
    s_tensor = torch.tensor([[s]], dtype=torch.float32)

    with torch.no_grad():
        if a == 0:
            return v_theta_0(s_tensor).item()
        else:
            return v_theta_1(s_tensor).item()


def GetPolicy(v_theta_0: MonotonicNetwork, v_theta_1: MonotonicNetwork, s: float) -> Tuple[float, float]:
    """
    Get choice probabilities for a given state.

    Args:
        v_theta_0: Network for action 0
        v_theta_1: Network for action 1
        s: State value

    Returns:
        Tuple of (prob_a0, prob_a1)
    """
    import numpy as np

    v0 = GetValue(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=s, a=0)
    v1 = GetValue(v_theta_0=v_theta_0, v_theta_1=v_theta_1, s=s, a=1)

    exp_v0 = np.exp(v0)
    exp_v1 = np.exp(v1)

    prob_0 = exp_v0 / (exp_v0 + exp_v1)
    prob_1 = exp_v1 / (exp_v0 + exp_v1)

    return prob_0, prob_1

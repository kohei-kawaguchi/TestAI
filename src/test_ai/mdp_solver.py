"""
MDP Solver using Value Iteration with Neural Networks.

This module implements value iteration for solving Markov Decision Processes
with continuous state spaces and discrete action spaces using neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


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
            s: State tensor of shape (batch_size, 1)

        Returns:
            Value tensor of shape (batch_size, 1)
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


class MDPSolver:
    """
    Value iteration solver for MDP with Type-I Extreme Value shocks.
    """

    def __init__(
        self,
        beta: float,
        gamma: float,
        delta: float,
        hidden_sizes: list[int] = [32, 32],
        learning_rate: float = 1e-3,
        euler_mascheroni: float = 0.5772156649015329
    ):
        """
        Initialize the MDP solver.

        Args:
            beta: Reward weight on state
            gamma: State depreciation rate
            delta: Discount factor
            hidden_sizes: Network architecture
            learning_rate: Learning rate for optimizer
            euler_mascheroni: Euler-Mascheroni constant
        """
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.gamma_e = euler_mascheroni

        # Create two networks for binary actions
        self.v_net_0 = MonotonicNetwork(hidden_sizes)
        self.v_net_1 = MonotonicNetwork(hidden_sizes)

        # Optimizer
        params = list(self.v_net_0.parameters()) + list(self.v_net_1.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

    def mean_reward(self, s: torch.Tensor, a: int) -> torch.Tensor:
        """
        Compute mean reward function.

        Args:
            s: State tensor
            a: Action (0 or 1)

        Returns:
            Mean reward
        """
        return self.beta * s - a

    def next_state(self, s: torch.Tensor, a: int) -> torch.Tensor:
        """
        Compute next state.

        Args:
            s: Current state
            a: Action (0 or 1)

        Returns:
            Next state
        """
        return (1 - self.gamma) * s + a

    def expected_value(self, s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute expected value of next state using log-sum-exp.

        Args:
            s_next: Next state tensor

        Returns:
            Expected value
        """
        v0 = self.v_net_0(s_next)
        v1 = self.v_net_1(s_next)

        # Log-sum-exp trick for numerical stability
        max_v = torch.max(v0, v1)
        log_sum_exp = max_v + torch.log(torch.exp(v0 - max_v) + torch.exp(v1 - max_v))

        return log_sum_exp + self.gamma_e

    def compute_targets(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Bellman target values for both actions.

        Args:
            states: Batch of states

        Returns:
            Tuple of (targets_a0, targets_a1)
        """
        targets_0 = []
        targets_1 = []

        for a in [0, 1]:
            s_next = self.next_state(states, a)
            ev = self.expected_value(s_next)
            target = self.mean_reward(states, a) + self.delta * ev

            if a == 0:
                targets_0 = target
            else:
                targets_1 = target

        return targets_0, targets_1

    def fit_iteration(
        self,
        states: torch.Tensor,
        num_epochs: int = 100
    ) -> float:
        """
        Perform one value iteration step.

        Args:
            states: Batch of sampled states
            num_epochs: Number of gradient descent epochs

        Returns:
            Maximum error after update
        """
        # Compute targets
        with torch.no_grad():
            targets_0, targets_1 = self.compute_targets(states)

        # Update networks
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            pred_0 = self.v_net_0(states)
            pred_1 = self.v_net_1(states)

            # Loss
            loss = torch.mean((pred_0 - targets_0) ** 2) + torch.mean((pred_1 - targets_1) ** 2)

            # Backward pass
            loss.backward()
            self.optimizer.step()

        # Compute max error
        with torch.no_grad():
            pred_0 = self.v_net_0(states)
            pred_1 = self.v_net_1(states)
            error_0 = torch.abs(pred_0 - targets_0)
            error_1 = torch.abs(pred_1 - targets_1)
            max_error = torch.max(torch.max(error_0), torch.max(error_1)).item()

        return max_error

    def solve(
        self,
        state_range: Tuple[float, float] = (0.0, 10.0),
        num_states: int = 100,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> dict:
        """
        Solve the MDP using value iteration.

        Args:
            state_range: Range of states to sample
            num_states: Number of states to sample
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            num_epochs: Number of gradient descent epochs per iteration
            verbose: Whether to print progress

        Returns:
            Dictionary with convergence info
        """
        # Generate state samples
        states = torch.linspace(state_range[0], state_range[1], num_states).reshape(-1, 1)

        history = {
            'iterations': [],
            'max_errors': []
        }

        for iteration in range(max_iter):
            max_error = self.fit_iteration(states, num_epochs)

            history['iterations'].append(iteration)
            history['max_errors'].append(max_error)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max_error = {max_error:.6f}")

            if max_error < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

        return history

    def get_value(self, s: float, a: int) -> float:
        """
        Get value function for a given state and action.

        Args:
            s: State value
            a: Action (0 or 1)

        Returns:
            Value
        """
        s_tensor = torch.tensor([[s]], dtype=torch.float32)

        with torch.no_grad():
            if a == 0:
                return self.v_net_0(s_tensor).item()
            else:
                return self.v_net_1(s_tensor).item()

    def get_policy(self, s: float) -> Tuple[float, float]:
        """
        Get choice probabilities for a given state.

        Args:
            s: State value

        Returns:
            Tuple of (prob_a0, prob_a1)
        """
        v0 = self.get_value(s, 0)
        v1 = self.get_value(s, 1)

        exp_v0 = np.exp(v0)
        exp_v1 = np.exp(v1)

        prob_0 = exp_v0 / (exp_v0 + exp_v1)
        prob_1 = exp_v1 / (exp_v0 + exp_v1)

        return prob_0, prob_1

"""
MDP Estimator using Two-Step Revealed Preference Approach.

This module implements parameter estimation for Markov Decision Processes
using observed state-action data. It estimates gamma from state transitions
and beta via revealed preference with CCP estimation.

The implementation follows the modular pseudo code structure with separate
functions for each subroutine, reusing shared functions from mdp_solver.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import sys
sys.path.insert(0, '..')

from mdp_solver import (
    MonotonicNetwork,
    InitializeNetworks,
    GenerateStateGrid,
    ComputeNextState,
    ComputeMeanReward,
    ComputeLoss,
    UpdateNetworks,
    CheckConvergence
)


# ============================================================================
# Main Algorithm
# ============================================================================

def EstimateMDP(
    states: np.ndarray,
    actions: np.ndarray,
    delta: float,
    N: int,
    state_range: Tuple[float, float],
    hyperparameters: Dict,
    num_epochs: int,
    learning_rate: float,
    beta_grid: np.ndarray,
    epsilon_tol: float,
    max_iter: int,
    gamma_E: float
) -> Tuple[float, float]:
    """
    Procedure EstimateMDP(...)

    Main two-step estimation algorithm.

    Args:
        states: Array of shape (M, T) with observed states
        actions: Array of shape (M, T) with observed actions
        delta: Discount factor (known)
        N: Number of grid points for evaluation
        state_range: Tuple of (min_state, max_state)
        hyperparameters: Dict containing 'hidden_sizes'
        num_epochs: Training epochs for networks
        learning_rate: Learning rate for optimization
        beta_grid: Array of candidate beta values
        epsilon_tol: Convergence tolerance
        max_iter: Maximum iterations for value iteration
        gamma_E: Euler-Mascheroni constant

    Returns:
        Tuple of (gamma_hat, beta_hat): Estimated parameters
    """
    # Step 1: Estimate gamma
    gamma_hat = EstimateGamma(states=states, actions=actions)

    # Step 2: Estimate beta
    beta_hat = EstimateBeta(
        states=states,
        actions=actions,
        gamma=gamma_hat,
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

    return gamma_hat, beta_hat


# ============================================================================
# Step 1: Estimate Gamma
# ============================================================================

def EstimateGamma(states: np.ndarray, actions: np.ndarray) -> float:
    """
    Procedure EstimateGamma(states: Array[M×T], actions: Array[M×T]) -> float

    Estimate depreciation parameter from state transitions.

    Uses the deterministic state transition:
        s_{t+1} = (1 - gamma) * s_t + a_t
    Rearranged:
        gamma = 1 - (s_{t+1} - a_t) / s_t

    Args:
        states: Array of shape (M, T) with observed states
        actions: Array of shape (M, T) with observed actions

    Returns:
        gamma_hat: Estimated depreciation parameter
    """
    M, T = states.shape
    gamma_estimates = []

    for m in range(M):
        for t in range(T - 1):
            s_t = states[m, t]
            s_t_plus_1 = states[m, t + 1]
            a_t = actions[m, t]

            # Avoid division by zero
            if s_t != 0:
                gamma_mt = 1 - (s_t_plus_1 - a_t) / s_t
                gamma_estimates.append(gamma_mt)

    # Take mean of all estimates
    gamma_hat = np.mean(gamma_estimates)

    return gamma_hat


# ============================================================================
# Step 2: Estimate Beta via Revealed Preference
# ============================================================================

def EstimateBeta(
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float,
    delta: float,
    N: int,
    state_range: Tuple[float, float],
    hyperparameters: Dict,
    num_epochs: int,
    learning_rate: float,
    beta_grid: np.ndarray,
    epsilon_tol: float,
    max_iter: int,
    gamma_E: float
) -> float:
    """
    Procedure EstimateBeta(...) -> float

    Estimate beta parameter via revealed preference approach.

    Uses nested fixed-point:
    - Outer loop: Search over beta candidates
    - Inner loop: Solve value functions given CCP and beta

    Args:
        states: Array of shape (M, T) with observed states
        actions: Array of shape (M, T) with observed actions
        gamma: Estimated depreciation parameter
        delta: Discount factor
        N: Number of grid points
        state_range: Tuple of (min_state, max_state)
        hyperparameters: Dict containing 'hidden_sizes'
        num_epochs: Training epochs
        learning_rate: Learning rate
        beta_grid: Array of candidate beta values
        epsilon_tol: Convergence tolerance
        max_iter: Maximum iterations
        gamma_E: Euler-Mascheroni constant

    Returns:
        beta_hat: Estimated beta parameter
    """
    # Step 2a: Estimate CCP from data
    P_hat = EstimateCCP(
        states=states,
        actions=actions,
        hyperparameters=hyperparameters,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # Generate state grid for evaluation
    S = GenerateStateGrid(N=N, state_range=state_range)

    # Step 2b-2d: Search over beta grid
    distances = []
    K = len(beta_grid)

    for k in range(K):
        beta_k = beta_grid[k]

        # Solve value functions given CCP and beta_k
        v_theta_0, v_theta_1 = SolveValueFunctionGivenCCP(
            P_hat=P_hat,
            beta=beta_k,
            gamma=gamma,
            delta=delta,
            gamma_E=gamma_E,
            hyperparameters=hyperparameters,
            S=S,
            max_iter=max_iter,
            epsilon_tol=epsilon_tol,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )

        # Compute updated CCP from value functions
        P_updated = ComputeCCPFromValue(S=S, v_theta_0=v_theta_0, v_theta_1=v_theta_1)

        # Evaluate estimated CCP on grid
        with torch.no_grad():
            P_hat_eval = P_hat(S)

        # Compute distance
        distance = ComputeDistance(P_hat_eval=P_hat_eval, P_updated=P_updated)
        distances.append(distance)

    # Find beta that minimizes distance
    k_star = np.argmin(distances)
    beta_hat = beta_grid[k_star]

    return beta_hat


# ============================================================================
# Subroutines for Step 2a: Estimate CCP
# ============================================================================

def EstimateCCP(
    states: np.ndarray,
    actions: np.ndarray,
    hyperparameters: Dict,
    num_epochs: int,
    learning_rate: float
) -> MonotonicNetwork:
    """
    Procedure EstimateCCP(...) -> Network

    Estimate conditional choice probability as monotonic function of state.

    Uses maximum likelihood estimation with binary cross-entropy loss.
    The network outputs P(a=1|s).

    Args:
        states: Array of shape (M, T) with observed states
        actions: Array of shape (M, T) with observed actions
        hyperparameters: Dict containing 'hidden_sizes'
        num_epochs: Training epochs
        learning_rate: Learning rate

    Returns:
        P_hat: Trained monotonic network for CCP
    """
    # Initialize monotonic network
    P_hat = InitializeMonotonicNetwork(hyperparameters=hyperparameters)

    # Create optimizer
    optimizer = torch.optim.Adam(P_hat.parameters(), lr=learning_rate)

    # Convert data to tensors
    M, T = states.shape
    states_flat = states.flatten()
    actions_flat = actions.flatten()

    states_tensor = torch.tensor(states_flat, dtype=torch.float32).reshape(-1, 1)
    actions_tensor = torch.tensor(actions_flat, dtype=torch.float32)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass: predict P(a=1|s)
        p_hat = P_hat(states_tensor).squeeze()

        # Binary cross-entropy loss
        loss = ComputeBinaryCrossEntropy(actions=actions_tensor, probabilities=p_hat)

        # Backward pass
        loss.backward()
        optimizer.step()

    return P_hat


def InitializeMonotonicNetwork(hyperparameters: Dict) -> MonotonicNetwork:
    """
    Initialize a monotonic network for CCP estimation.

    Args:
        hyperparameters: Dict containing 'hidden_sizes'

    Returns:
        network: Initialized monotonic network
    """
    hidden_sizes = hyperparameters.get('hidden_sizes', [32, 32])
    network = MonotonicNetwork(hidden_sizes=hidden_sizes)
    return network


def ComputeBinaryCrossEntropy(actions: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for CCP estimation.

    Loss = -[a * log(p) + (1-a) * log(1-p)]

    Args:
        actions: Tensor of observed actions (0 or 1)
        probabilities: Tensor of predicted P(a=1|s)

    Returns:
        loss: Scalar loss tensor
    """
    # Clamp probabilities to avoid log(0)
    eps = 1e-7
    p = torch.clamp(probabilities, eps, 1 - eps)

    loss = -torch.mean(actions * torch.log(p) + (1 - actions) * torch.log(1 - p))

    return loss


# ============================================================================
# Subroutines for Step 2b: Solve Value Function Given CCP
# ============================================================================

def SolveValueFunctionGivenCCP(
    P_hat: MonotonicNetwork,
    beta: float,
    gamma: float,
    delta: float,
    gamma_E: float,
    hyperparameters: Dict,
    S: torch.Tensor,
    max_iter: int,
    epsilon_tol: float,
    num_epochs: int,
    learning_rate: float
) -> Tuple[MonotonicNetwork, MonotonicNetwork]:
    """
    Procedure SolveValueFunctionGivenCCP(...) -> (Network, Network)

    Solve value functions that are consistent with estimated CCP.

    This is the "critic" step in the actor-critic framework.

    Args:
        P_hat: Estimated CCP network
        beta: Candidate beta value
        gamma: Estimated gamma value
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant
        hyperparameters: Dict containing 'hidden_sizes'
        S: State grid tensor of shape (N, 1)
        max_iter: Maximum iterations
        epsilon_tol: Convergence tolerance
        num_epochs: Training epochs per iteration
        learning_rate: Learning rate

    Returns:
        Tuple of (v_theta_0, v_theta_1): Trained value networks
    """
    # Initialize networks
    v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

    # Create optimizer
    params = list(v_theta_0.parameters()) + list(v_theta_1.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Value iteration loop
    for iteration in range(max_iter):
        # Compute Bellman targets given CCP
        targets = ComputeBellmanTargetsGivenCCP(
            S=S,
            P_hat=P_hat,
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            beta=beta,
            gamma=gamma,
            delta=delta,
            gamma_E=gamma_E
        )

        # Update networks
        v_theta_0, v_theta_1 = UpdateNetworks(
            S=S,
            targets=targets,
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1,
            num_epochs=num_epochs,
            optimizer=optimizer
        )

        # Check convergence
        max_error = CheckConvergence(
            S=S,
            targets=targets,
            v_theta_0=v_theta_0,
            v_theta_1=v_theta_1
        )

        if max_error < epsilon_tol:
            break

    return v_theta_0, v_theta_1


def ComputeBellmanTargetsGivenCCP(
    S: torch.Tensor,
    P_hat: MonotonicNetwork,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork,
    beta: float,
    gamma: float,
    delta: float,
    gamma_E: float
) -> torch.Tensor:
    """
    Procedure ComputeBellmanTargetsGivenCCP(...) -> Tensor[N×2]

    Compute Bellman targets using estimated policy probabilities.

    Instead of using the optimal policy (via LogSumExp), we use the
    estimated policy from the CCP network.

    Args:
        S: State grid tensor of shape (N, 1)
        P_hat: Estimated CCP network
        v_theta_0: Value network for action 0
        v_theta_1: Value network for action 1
        beta: Reward parameter
        gamma: Depreciation parameter
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant

    Returns:
        targets: Tensor of shape (N, 2) with Bellman targets
    """
    targets_list = []

    with torch.no_grad():
        for a in [0, 1]:
            # Next state
            s_prime_i = ComputeNextState(s=S, a=a, gamma=gamma)

            # Estimated policy probability P(a=1|s')
            p_hat_i = P_hat(s_prime_i)

            # Value functions at next state
            v_0 = v_theta_0(s_prime_i)
            v_1 = v_theta_1(s_prime_i)

            # Expected value under estimated policy
            EV_i = (1 - p_hat_i) * v_0 + p_hat_i * v_1 + gamma_E

            # Bellman target
            y_i_a = ComputeMeanReward(s=S, a=a, beta=beta) + delta * EV_i
            targets_list.append(y_i_a)

    # Stack into (N, 2) tensor
    targets = torch.cat(targets_list, dim=1)
    return targets


# ============================================================================
# Subroutines for Step 2c: Compute CCP from Value Functions
# ============================================================================

def ComputeCCPFromValue(
    S: torch.Tensor,
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork
) -> torch.Tensor:
    """
    Procedure ComputeCCPFromValue(...) -> Tensor[N×1]

    Compute choice probabilities from value functions using logit formula.

    P(a=1|s) = exp(v1) / (exp(v0) + exp(v1))

    Args:
        S: State grid tensor of shape (N, 1)
        v_theta_0: Value network for action 0
        v_theta_1: Value network for action 1

    Returns:
        P_updated: Tensor of shape (N, 1) with P(a=1|s)
    """
    with torch.no_grad():
        v_0 = v_theta_0(S)
        v_1 = v_theta_1(S)

        # Logit probability
        P_updated = torch.exp(v_1) / (torch.exp(v_0) + torch.exp(v_1))

    return P_updated


# ============================================================================
# Subroutines for Step 2d: Compute Distance
# ============================================================================

def ComputeDistance(P_hat_eval: torch.Tensor, P_updated: torch.Tensor) -> float:
    """
    Procedure ComputeDistance(...) -> float

    Compute squared distance between estimated and updated CCPs.

    Args:
        P_hat_eval: Estimated CCP on grid, shape (N, 1)
        P_updated: Updated CCP from value functions, shape (N, 1)

    Returns:
        distance: Sum of squared differences
    """
    diff = P_hat_eval - P_updated
    squared_diff = diff * diff
    distance = torch.sum(squared_diff).item()

    return distance

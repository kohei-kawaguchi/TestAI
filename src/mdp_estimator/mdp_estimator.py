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
from concurrent.futures import ThreadPoolExecutor, as_completed

from mdp_solver import (
    MonotonicNetwork,
    InitializeNetworks,
    GenerateStateGrid,
    ComputeMeanReward,
    ComputeNextState
)


# ============================================================================
# CCP Network with Sigmoid Output
# ============================================================================

class IncreasingCCPNetwork(nn.Module):
    """
    Monotonically INCREASING network for CCP estimation with sigmoid output.

    In the capital accumulation model, P(a=1|s) should be DECREASING in state s.
    Instead of directly predicting this, we predict P(a=0|s) which is INCREASING:
    - Higher capital stock → Higher probability of NOT investing
    - Then compute P(a=1|s) = 1 - P(a=0|s)

    Implementation:
    - All layers: positive weights (w_i ≥ 0) via softplus to ensure increasing
    - This is the standard monotonic increasing network architecture
    - Sigmoid output maps to [0,1] probability range
    """

    def __init__(self, hidden_sizes: list = [32, 32]):
        """
        Initialize increasing CCP network.

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
        Forward pass with monotonic increasing constraint and sigmoid output.

        All layers use positive weights (w_i ≥ 0) via softplus to ensure increasing.

        Args:
            s: State tensor of shape (N, 1)

        Returns:
            Probability tensor of shape (N, 1) with values in [0,1],
            monotonically INCREASING in s, representing P(a=0|s)
        """
        x = s

        # All hidden layers: positive weights to ensure increasing (w_i ≥ 0)
        for layer in self.layers[:-1]:
            weight = torch.nn.functional.softplus(layer.weight)
            bias = layer.bias
            x = torch.nn.functional.linear(x, weight, bias)
            x = torch.tanh(x)

        # Final layer: positive weights
        final_layer = self.layers[-1]
        weight = torch.nn.functional.softplus(final_layer.weight)
        bias = final_layer.bias
        logits = torch.nn.functional.linear(x, weight, bias)

        # Apply sigmoid to map to [0,1]
        probabilities = torch.sigmoid(logits)

        return probabilities


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
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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
        Tuple of (gamma_hat, beta_hat, beta_grid, distances):
            gamma_hat: Estimated depreciation parameter
            beta_hat: Estimated reward parameter
            beta_grid: Array of beta candidates searched
            distances: Array of distances for each beta candidate
    """
    # Step 1: Estimate gamma
    gamma_hat = EstimateGamma(states=states, actions=actions)

    # Step 2: Estimate beta
    beta_hat, distances = EstimateBeta(
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

    return gamma_hat, beta_hat, beta_grid, distances


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
) -> Tuple[float, np.ndarray]:
    """
    Procedure EstimateBeta(...) -> (float, Array[K])

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
        Tuple of (beta_hat, distances):
            beta_hat: Estimated beta parameter
            distances: Array of distances for each beta candidate
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

    # Step 2b-2d: Search over beta grid in parallel
    K = len(beta_grid)

    # Evaluate estimated CCP on grid (once, outside loop)
    # P_hat predicts P(a=0|s), convert to P(a=1|s) = 1 - P(a=0|s)
    with torch.no_grad():
        P_hat_eval = 1 - P_hat(S)

    print(f"  Searching over {K} beta candidates in parallel...")

    # Parallel evaluation of beta candidates
    distances = [None] * K

    # Use ThreadPoolExecutor for notebook/Quarto compatibility
    # ProcessPoolExecutor has issues with __main__ module in notebooks
    with ThreadPoolExecutor(max_workers=min(K, 4)) as executor:
        # Submit all beta evaluations
        future_to_k = {
            executor.submit(
                _evaluate_beta_candidate,
                beta_grid[k],
                P_hat,
                P_hat_eval,
                gamma,
                delta,
                gamma_E,
                hyperparameters,
                S,
                max_iter,
                epsilon_tol,
                num_epochs,
                learning_rate
            ): k
            for k in range(K)
        }

        # Collect results as they complete
        for future in as_completed(future_to_k):
            k = future_to_k[future]
            try:
                distance = future.result()
                distances[k] = distance
                print(f"    Completed beta[{k}]={beta_grid[k]:.3f}, distance={distance:.6f}")
            except Exception as exc:
                print(f"    Beta[{k}]={beta_grid[k]:.3f} generated an exception: {exc}")
                distances[k] = np.inf

    # Find beta that minimizes distance
    distances_array = np.array(distances)
    k_star = np.argmin(distances_array)
    beta_hat = beta_grid[k_star]

    print(f"  Best beta: {beta_hat:.4f} (k={k_star}, distance={distances_array[k_star]:.6f})")

    return beta_hat, distances_array


def _evaluate_beta_candidate(
    beta_k: float,
    P_hat: MonotonicNetwork,
    P_hat_eval: torch.Tensor,
    gamma: float,
    delta: float,
    gamma_E: float,
    hyperparameters: Dict,
    S: torch.Tensor,
    max_iter: int,
    epsilon_tol: float,
    num_epochs: int,
    learning_rate: float
) -> float:
    """
    Helper function to evaluate a single beta candidate.

    This function is designed to be called in parallel.

    Args:
        beta_k: Candidate beta value
        P_hat: Estimated CCP network (predicts P(a=0|s))
        P_hat_eval: Pre-computed P(a=1|s) = 1 - P_hat(S) on grid
        gamma: Estimated gamma
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant
        hyperparameters: Network hyperparameters
        S: State grid
        max_iter: Maximum iterations
        epsilon_tol: Convergence tolerance
        num_epochs: Training epochs
        learning_rate: Learning rate

    Returns:
        distance: Distance between estimated and updated CCPs
    """
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

    # Compute distance
    distance = ComputeDistance(P_hat_eval=P_hat_eval, P_updated=P_updated)

    return distance


# ============================================================================
# Subroutines for Step 2a: Estimate CCP
# ============================================================================

def EstimateCCP(
    states: np.ndarray,
    actions: np.ndarray,
    hyperparameters: Dict,
    num_epochs: int,
    learning_rate: float
) -> IncreasingCCPNetwork:
    """
    Procedure EstimateCCP(...) -> Network

    Estimate conditional choice probability by predicting P(a=0|s) as INCREASING function.

    We want P(a=1|s) DECREASING in s. Instead of directly predicting this, we predict
    P(a=0|s) which is INCREASING in s, then compute P(a=1|s) = 1 - P(a=0|s).

    Uses maximum likelihood estimation with binary cross-entropy loss.
    The network outputs P(a=0|s) in [0,1], INCREASING in s (higher capital → higher prob of not investing).

    Args:
        states: Array of shape (M, T) with observed states
        actions: Array of shape (M, T) with observed actions (1 = invest, 0 = not invest)
        hyperparameters: Dict containing 'hidden_sizes'
        num_epochs: Training epochs
        learning_rate: Learning rate

    Returns:
        P_hat_0: Trained monotonic network for P(a=0|s)
    """
    # Initialize monotonic increasing network
    P_hat_0 = InitializeIncreasingCCPNetwork(hyperparameters=hyperparameters)

    # Create optimizer
    optimizer = torch.optim.Adam(P_hat_0.parameters(), lr=learning_rate)

    # Convert data to tensors
    M, T = states.shape
    states_flat = states.flatten()
    actions_flat = actions.flatten()

    states_tensor = torch.tensor(states_flat, dtype=torch.float32).reshape(-1, 1)
    actions_tensor = torch.tensor(actions_flat, dtype=torch.float32)

    # Convert to indicator 1{a=0}
    actions_0_tensor = 1 - actions_tensor

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass: predict P(a=0|s)
        p_hat_0 = P_hat_0(states_tensor).squeeze()

        # Binary cross-entropy loss on P(a=0|s)
        loss = ComputeBinaryCrossEntropy(actions=actions_0_tensor, probabilities=p_hat_0)

        # Backward pass
        loss.backward()
        optimizer.step()

    return P_hat_0


def InitializeIncreasingCCPNetwork(hyperparameters: Dict) -> IncreasingCCPNetwork:
    """
    Initialize a monotonically INCREASING CCP network with sigmoid output.

    The network outputs P(a=0|s) in [0,1] range, monotonically INCREASING in s.
    Higher capital stock → Higher probability of NOT investing (a=0).

    Args:
        hyperparameters: Dict containing 'hidden_sizes'

    Returns:
        network: Initialized increasing CCP network with sigmoid output
    """
    hidden_sizes = hyperparameters.get('hidden_sizes', [32, 32])
    network = IncreasingCCPNetwork(hidden_sizes=hidden_sizes)
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

    Solve value functions via linear system, then fit networks.

    This is the "critic" step in the actor-critic framework.

    Args:
        P_hat: Estimated CCP network (predicts P(a=0|s))
        beta: Candidate beta value
        gamma: Estimated gamma value
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant
        hyperparameters: Dict containing 'hidden_sizes'
        S: State grid tensor of shape (N, 1)
        max_iter: Maximum iterations (unused, kept for compatibility)
        epsilon_tol: Convergence tolerance (unused, kept for compatibility)
        num_epochs: Training epochs for network fitting
        learning_rate: Learning rate for network fitting

    Returns:
        Tuple of (v_theta_0, v_theta_1): Trained value networks
    """
    # Step 1: Solve linear Bellman equation on grid
    v_0, v_1 = SolveLinearBellman(
        P_hat=P_hat,
        beta=beta,
        gamma=gamma,
        delta=delta,
        gamma_E=gamma_E,
        S=S
    )

    # Step 2: Fit networks to grid values via supervised learning
    v_theta_0, v_theta_1 = FitNetworksToValues(
        S=S,
        v_0=v_0,
        v_1=v_1,
        hyperparameters=hyperparameters,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    return v_theta_0, v_theta_1


def SolveLinearBellman(
    P_hat: MonotonicNetwork,
    beta: float,
    gamma: float,
    delta: float,
    gamma_E: float,
    S: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procedure SolveLinearBellman(...) -> (Array[N], Array[N])

    Solve Bellman equation as linear system: (I - delta*T)v = r

    For each action a, we have:
        v^(a) = r^(a) + delta * T^(a) * v
    where v = [(1-p)*v^(0) + p*v^(1)] + gamma_E

    Rearranging:
        v^(a) - delta * T^(a) * [(1-p)*v^(0) + p*v^(1)] = r^(a) + delta * gamma_E

    Args:
        P_hat: Estimated CCP network (predicts P(a=0|s))
        beta: Reward parameter
        gamma: Depreciation parameter
        delta: Discount factor
        gamma_E: Euler-Mascheroni constant
        S: State grid tensor of shape (N, 1)

    Returns:
        Tuple of (v_0, v_1): Value functions on grid as numpy arrays
    """
    N = S.shape[0]

    # Convert S to numpy for indexing
    S_np = S.detach().cpu().numpy().flatten()

    # Evaluate CCP on grid
    # P_hat predicts P(a=0|s), convert to P(a=1|s) = 1 - P(a=0|s)
    with torch.no_grad():
        P_0_eval = P_hat(S).detach().cpu().numpy().flatten()  # P(a=0|s)
        P_eval = 1 - P_0_eval  # Convert to P(a=1|s)

    # Build transition matrices and reward vectors for both actions
    # We'll solve the coupled system for both v^(0) and v^(1) simultaneously

    # Initialize system: [v^(0); v^(1)] of size 2N
    A = np.eye(2 * N)  # Will become (I - delta*T)
    b = np.zeros(2 * N)  # Will become r + delta*gamma_E

    for a in [0, 1]:
        # Row offset for this action
        row_offset = a * N

        for i in range(N):
            s_i_tensor = S[i:i+1]  # Keep as tensor for shared functions

            # Compute reward using shared function
            r_i_tensor = ComputeMeanReward(s=s_i_tensor, a=a, beta=beta)
            r_i = r_i_tensor.item()
            b[row_offset + i] = r_i + delta * gamma_E

            # Compute next state using shared function
            s_prime_i_tensor = ComputeNextState(s=s_i_tensor, a=a, gamma=gamma)
            s_prime_i = s_prime_i_tensor.item()

            # Find nearest grid point (simple nearest neighbor)
            j = np.argmin(np.abs(S_np - s_prime_i))

            # Get transition probability
            p_j = P_eval[j]  # P(a=1|s'_i)

            # Build transition: E[v|s'] = (1-p)*v^(0) + p*v^(1)
            # Subtract delta * T from left side (I - delta*T)
            A[row_offset + i, 0 * N + j] -= delta * (1 - p_j)  # v^(0) component
            A[row_offset + i, 1 * N + j] -= delta * p_j        # v^(1) component

    # Solve linear system: A * v = b
    v_combined = np.linalg.solve(A, b)

    # Extract v^(0) and v^(1)
    v_0 = v_combined[0:N]
    v_1 = v_combined[N:2*N]

    return v_0, v_1


def FitNetworksToValues(
    S: torch.Tensor,
    v_0: np.ndarray,
    v_1: np.ndarray,
    hyperparameters: Dict,
    num_epochs: int,
    learning_rate: float
) -> Tuple[MonotonicNetwork, MonotonicNetwork]:
    """
    Procedure FitNetworksToValues(...) -> (Network, Network)

    Train neural networks to approximate grid-solved value functions.

    Uses supervised learning with MSE loss.

    Args:
        S: State grid tensor of shape (N, 1)
        v_0: Value function for action 0 on grid (numpy array)
        v_1: Value function for action 1 on grid (numpy array)
        hyperparameters: Dict containing 'hidden_sizes'
        num_epochs: Training epochs
        learning_rate: Learning rate

    Returns:
        Tuple of (v_theta_0, v_theta_1): Trained value networks
    """
    # Initialize networks
    v_theta_0, v_theta_1 = InitializeNetworks(hyperparameters=hyperparameters)

    # Convert targets to tensors
    targets = torch.tensor(np.stack([v_0, v_1], axis=1), dtype=torch.float32)

    # Create optimizer
    params = list(v_theta_0.parameters()) + list(v_theta_1.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Training loop (supervised learning)
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        pred_0 = v_theta_0(S)
        pred_1 = v_theta_1(S)
        pred = torch.cat([pred_0, pred_1], dim=1)

        # MSE loss
        loss = torch.mean((pred - targets) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

    return v_theta_0, v_theta_1


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

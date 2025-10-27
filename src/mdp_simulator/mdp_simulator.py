"""
MDP Simulator using Monte Carlo Methods.

This module implements Monte Carlo simulation for analyzing Markov Decision Processes
using trained value function networks. The implementation follows the modular pseudo code
structure and reuses shared functions from mdp_solver.

The simulator draws actions directly from choice probabilities computed by the trained
value functions, rather than drawing preference shocks.
"""

import numpy as np
from typing import Tuple

# Import shared functions from mdp_solver
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mdp_solver'))

from mdp_solver import MonotonicNetwork, ComputeChoiceProbability, ComputeNextState, ComputeMeanReward


# ============================================================================
# Subroutines following the pseudo code structure
# ============================================================================

def DrawAction(P_0: float, P_1: float) -> int:
    """
    Procedure DrawAction(P_0: float, P_1: float) -> int

    Draw action from categorical distribution using inverse CDF method.

    Args:
        P_0: Probability of choosing action 0
        P_1: Probability of choosing action 1

    Returns:
        action: Drawn action (0 or 1)
    """
    u = np.random.uniform(0, 1)
    if u < P_0:
        return 0
    else:
        return 1


def SimulateMDP(
    v_theta_0: MonotonicNetwork,
    v_theta_1: MonotonicNetwork,
    beta: float,
    gamma: float,
    delta: float,
    M: int,
    T: int,
    s_0: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Procedure SimulateMDP(v_theta^(0): Network, v_theta^(1): Network, beta: float,
                          gamma: float, delta: float, M: int, T: int, s_0: float,
                          seed: int) -> (Array[M×T], Array[M×T], Array[M×T])

    Simulate MDP under the optimal policy learned by neural networks.

    This function generates M paths of length T, where actions are drawn from
    choice probabilities computed by the trained value functions.

    Args:
        v_theta_0: Trained network for action 0
        v_theta_1: Trained network for action 1
        beta: Reward weight on state
        gamma: State depreciation rate
        delta: Discount factor (not used in mean rewards, but part of config)
        M: Number of simulation paths
        T: Number of time periods per path
        s_0: Initial state value
        seed: Random seed for reproducibility

    Returns:
        Tuple of (states, actions, rewards):
            - states: Array of shape (M, T) containing state paths
            - actions: Array of shape (M, T) containing action paths
            - rewards: Array of shape (M, T) containing reward paths
    """
    # Set random seed to seed
    np.random.seed(seed)

    # Pre-allocate arrays for storage efficiency
    states = np.zeros((M, T))
    actions = np.zeros((M, T), dtype=int)
    rewards = np.zeros((M, T))

    # Set networks to evaluation mode
    v_theta_0.eval()
    v_theta_1.eval()

    # For each path m = 1 to M
    for m in range(M):
        # Initialize first state
        states[m, 0] = s_0

        # Simulate forward for t = 0 to T-1
        for t in range(T):
            # Get current state
            s_t = states[m, t]

            # Compute choice probabilities using shared function from mdp_solver
            P_a0, P_a1 = ComputeChoiceProbability(
                s=s_t,
                v_theta_0=v_theta_0,
                v_theta_1=v_theta_1
            )

            # Draw action from choice probabilities
            a_t = DrawAction(P_0=P_a0, P_1=P_a1)
            actions[m, t] = a_t

            # Compute mean reward using shared function from mdp_solver
            # Note: ComputeMeanReward expects tensor input, so convert
            import torch
            s_t_tensor = torch.tensor([[s_t]], dtype=torch.float32)
            r_t_tensor = ComputeMeanReward(s=s_t_tensor, a=a_t, beta=beta)
            r_t = r_t_tensor.item()
            rewards[m, t] = r_t

            # Update state if not at terminal period
            if t < T - 1:
                # Compute next state using shared function from mdp_solver
                # Note: ComputeNextState expects tensor input
                s_next_tensor = ComputeNextState(s=s_t_tensor, a=a_t, gamma=gamma)
                s_next = s_next_tensor.item()
                states[m, t + 1] = s_next

    return states, actions, rewards

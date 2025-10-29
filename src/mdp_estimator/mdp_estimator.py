"""
MDP Estimator using Nested Fixed Point Algorithm.

This module implements nested fixed point estimation for Markov Decision Processes
using two-step estimation procedure. The implementation follows the modular pseudo code
structure and reuses shared functions and configurations from mdp_solver.

Step 1: Estimate gamma directly from state transitions using OLS
Step 2: Estimate beta via maximum likelihood, solving MDP at each evaluation

Key Design:
- Reuses SolveValueIteration from mdp_solver (DRY)
- Reuses solver_config from get_solver_config() (shared configuration)
- Only adds estimator-specific optimization_config
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.optimize import minimize

# Import shared functions from mdp_solver
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mdp_solver'))

from mdp_solver import (
    MonotonicNetwork,
    SolveValueIteration,
    ComputeChoiceProbability,
    ComputeNextState,
    ComputeMeanReward
)


# ============================================================================
# Subroutines following the pseudo code structure
# ============================================================================

def EstimateGamma(states: np.ndarray, actions: np.ndarray) -> float:
    """
    Procedure EstimateGamma(states: Array[M×T], actions: Array[M×T]) -> float

    Estimate state transition parameter gamma using ordinary least squares.

    The state transition equation is: s_{t+1} = (1 - gamma) * s_t + a_t
    This can be estimated via OLS since it's linear in gamma.

    Args:
        states: Array of shape (M, T) containing state paths
        actions: Array of shape (M, T) containing action paths

    Returns:
        gamma_hat: Estimated state transition parameter
    """
    # Step 1: Initialize sums
    numer = 0.0
    denom = 0.0

    M, T = states.shape

    # Step 2: For m = 1 to M, for t = 1 to T-1
    for m in range(M):
        for t in range(T - 1):
            s_t = states[m, t]
            a_t = actions[m, t]
            s_t_plus_1 = states[m, t + 1]

            # Accumulate sums for OLS estimator
            numer += s_t * (s_t_plus_1 - a_t)
            denom += s_t ** 2

    # Step 3: Compute OLS estimator
    gamma_hat = 1.0 - numer / denom

    # Step 4: Return
    return gamma_hat


def EstimateBeta(
    states: np.ndarray,
    actions: np.ndarray,
    gamma_hat: float,
    delta: float,
    gamma_E: float,
    solver_config: Dict,
    optimization_config: Dict
) -> float:
    """
    Procedure EstimateBeta(states: Array[M×T], actions: Array[M×T], gamma_hat: float,
                           delta: float, gamma_E: float, solver_config: dict,
                           optimization_config: dict) -> float

    Estimate reward parameter beta via maximum likelihood using nested fixed point.

    For each candidate beta, this solves the MDP using value iteration and evaluates
    the log-likelihood of observed actions given states. The nested structure comes from
    solving the MDP (inner fixed point) within the likelihood evaluation.

    Args:
        states: Array of shape (M, T) containing state paths
        actions: Array of shape (M, T) containing action paths
        gamma_hat: Estimated state transition parameter from Step 1
        delta: Discount factor (known)
        gamma_E: Euler-Mascheroni constant (known)
        solver_config: Dictionary with MDP solver configuration
        optimization_config: Dictionary with optimization configuration

    Returns:
        beta_hat: Estimated reward parameter
    """
    M, T = states.shape

    # Step 1: Define log-likelihood function
    def LogLikelihood(beta: float) -> float:
        """
        Procedure LogLikelihood(beta: float) -> float

        Compute log-likelihood of observed data for given beta.

        This is the inner fixed point: for each beta, we must solve the entire MDP
        to compute choice probabilities.

        Args:
            beta: Candidate value for reward parameter

        Returns:
            log_likelihood: Sum of log probabilities
        """
        # Step 1a: Solve MDP using value iteration (reuses solver from solve_mdp)
        v_theta_0, v_theta_1, _ = SolveValueIteration(
            beta=beta,
            gamma=gamma_hat,
            delta=delta,
            gamma_E=gamma_E,
            hyperparameters=solver_config['hyperparameters'],
            N=solver_config['N'],
            state_range=tuple(solver_config['state_range']),
            max_iter=solver_config['max_iter'],
            epsilon_tol=solver_config['epsilon_tol'],
            num_epochs=solver_config['num_epochs'],
            learning_rate=solver_config['learning_rate'],
            verbose=solver_config.get('verbose', False)
        )

        # Step 1b: Initialize log-likelihood
        log_likelihood = 0.0

        # Step 1c: For each observation (s_mt, a_mt)
        for m in range(M):
            for t in range(T):
                s_mt = states[m, t]
                a_mt = actions[m, t]

                # Compute choice probabilities using shared function from mdp_solver
                P_0, P_1 = ComputeChoiceProbability(
                    s=float(s_mt),
                    v_theta_0=v_theta_0,
                    v_theta_1=v_theta_1
                )

                # Add to log-likelihood
                if a_mt == 0:
                    log_likelihood += np.log(P_0)
                else:
                    log_likelihood += np.log(P_1)

        # Step 1d: Return
        return log_likelihood

    # Step 2: Extract optimization parameters
    beta_bounds = optimization_config['beta_bounds']
    method = optimization_config['method']
    tolerance = optimization_config['tolerance']
    initial_beta = optimization_config['initial_beta']

    # Step 3: Maximize log-likelihood over beta using numerical optimization
    # Note: scipy.optimize.minimize minimizes, so we negate the log-likelihood
    result = minimize(
        fun=lambda beta: -LogLikelihood(beta[0]),  # Negate for maximization
        x0=np.array([initial_beta]),
        method=method,
        bounds=[beta_bounds],
        options={'ftol': tolerance, 'disp': True}
    )

    # Step 4: Extract estimated beta
    beta_hat = result.x[0]

    # Step 5: Return
    return beta_hat


# ============================================================================
# Main Estimation Procedure
# ============================================================================

def EstimateMDP(
    states: np.ndarray,
    actions: np.ndarray,
    delta: float,
    gamma_E: float,
    solver_config: Dict,
    optimization_config: Dict
) -> Tuple[float, float]:
    """
    Procedure EstimateMDP(states: Array[M×T], actions: Array[M×T], delta: float,
                          gamma_E: float, solver_config: dict, optimization_config: dict)
                          -> (float, float)

    Main nested fixed point estimation procedure.

    This implements a two-step estimator:
    1. Estimate gamma directly from state transitions (fast, closed-form)
    2. Estimate beta via MLE, solving MDP at each evaluation (slow, nested)

    Args:
        states: Array of shape (M, T) containing state paths
        actions: Array of shape (M, T) containing action paths
        delta: Discount factor (known)
        gamma_E: Euler-Mascheroni constant (known)
        solver_config: Dictionary with MDP solver configuration
        optimization_config: Dictionary with optimization configuration

    Returns:
        Tuple of (beta_hat, gamma_hat): Estimated parameters
    """
    # Step 1: Estimate gamma from state transitions
    gamma_hat = EstimateGamma(states=states, actions=actions)

    # Step 2: Estimate beta via maximum likelihood
    beta_hat = EstimateBeta(
        states=states,
        actions=actions,
        gamma_hat=gamma_hat,
        delta=delta,
        gamma_E=gamma_E,
        solver_config=solver_config,
        optimization_config=optimization_config
    )

    # Step 3: Return estimated parameters
    return beta_hat, gamma_hat

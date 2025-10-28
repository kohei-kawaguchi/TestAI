"""
Central configuration for MDP solving, simulation, and estimation workflows.

The configuration is structured into four configs:
1. MODEL_CONFIG: MDP model parameters (shared across all workflows)
2. SOLVER_CONFIG: Solution method parameters (shared by solver and estimator)
3. SIMULATOR_CONFIG: Simulation method parameters (simulator-specific)
4. ESTIMATOR_CONFIG: Estimation method parameters (estimator-specific optimization)

Design principle: Only the solver loads from Python config. Downstream steps
(simulator, estimator) load configs from saved outputs of previous steps.

Key for estimator:
- Reuses SOLVER_CONFIG for nested MDP solving (passed to SolveValueIteration)
- Only adds optimization_config for outer loop (beta bounds, method, tolerance)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


# MDP model parameters (the economic model itself)
_MODEL_CONFIG: Dict[str, Any] = {
    "beta": 1.0,        # Reward parameter
    "gamma": 0.1,       # Depreciation rate
    "delta": 0.95,      # Discount factor
    "gamma_E": 0.5772156649015329,  # Euler-Mascheroni constant
}

# Solution method parameters (how to solve the model)
_SOLVER_CONFIG: Dict[str, Any] = {
    "hyperparameters": {
        "hidden_sizes": [32],
    },
    "N": 100,           # Number of grid points
    "state_range": (0.0, 10.0),  # State space range
    "max_iter": 1000,   # Maximum iterations for value iteration
    "epsilon_tol": 1e-4,  # Convergence tolerance
    "num_epochs": 50,   # Training epochs per iteration
    "learning_rate": 1e-3,  # Learning rate for network training
    "verbose": False,   # Print progress during solving (False for nested estimation)
}

# Comparative statics grids
_COMPARATIVE_STATICS: Dict[str, List[float]] = {
    "beta_values": [0.25 * i for i in range(0, 9)],   # 0.00 to 2.00 inclusive
    "gamma_values": [0.25 * i for i in range(0, 5)],  # 0.00 to 1.00 inclusive
}

# Simulation method parameters (how to simulate from solved model)
_SIMULATOR_CONFIG: Dict[str, Any] = {
    "M": 1000,          # Number of simulation paths
    "T": 100,           # Time periods per path
    "seed": 42,         # Random seed for reproducibility
}

# Estimation method parameters (how to estimate from simulated data)
_ESTIMATOR_CONFIG: Dict[str, Any] = {
    # Optimization configuration for beta estimation
    "optimization_config": {
        "beta_bounds": (0.1, 3.0),  # Bounds for beta optimization
        "method": "L-BFGS-B",        # Optimization method (supports bounds)
        "tolerance": 1e-6,           # Convergence tolerance for optimizer
        "initial_beta": 1.0,         # Initial guess for beta
    },
    # Solver configuration is reused from SOLVER_CONFIG
    # (passed to nested fixed point at each likelihood evaluation)
}


def get_model_config() -> Dict[str, Any]:
    """
    Return a deep copy of the MDP model configuration.

    These are the structural parameters of the economic model.
    """
    return deepcopy(_MODEL_CONFIG)


def get_solver_config() -> Dict[str, Any]:
    """
    Return a deep copy of the solver method configuration.

    These are parameters for how to solve the model numerically.
    """
    return deepcopy(_SOLVER_CONFIG)


def get_comparative_statics() -> Dict[str, List[float]]:
    """
    Return the comparative statics grids used in downstream analysis.
    """
    return {key: list(values) for key, values in _COMPARATIVE_STATICS.items()}


def get_simulator_config() -> Dict[str, Any]:
    """
    Return a deep copy of the simulation method configuration.

    These are parameters for how to simulate from the solved model.
    """
    return deepcopy(_SIMULATOR_CONFIG)


def get_estimator_config() -> Dict[str, Any]:
    """
    Return a deep copy of the estimation method configuration.

    These are parameters for how to estimate from simulated data.
    """
    return deepcopy(_ESTIMATOR_CONFIG)


__all__ = [
    "get_model_config",
    "get_solver_config",
    "get_comparative_statics",
    "get_simulator_config",
    "get_estimator_config"
]

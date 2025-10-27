"""
Central configuration for MDP solving and simulation workflows.

The configuration is expressed as pure Python structures so that both the
solver and simulator Quarto documents can import and share the same defaults.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


_SOLVER_CONFIG: Dict[str, Any] = {
    "beta": 1.0,
    "gamma": 0.1,
    "delta": 0.95,
    "gamma_E": 0.5772156649015329,
    "hyperparameters": {
        "hidden_sizes": [32, 32],
    },
    "N": 100,
    "state_range": (0.0, 10.0),
    "max_iter": 1000,
    "epsilon_tol": 1e-4,
    "num_epochs": 50,
    "learning_rate": 1e-3,
}

_COMPARATIVE_STATICS: Dict[str, List[float]] = {
    "beta_values": [0.25 * i for i in range(0, 9)],   # 0.00 to 2.00 inclusive
    "gamma_values": [0.25 * i for i in range(0, 5)],  # 0.00 to 1.00 inclusive
}

_SIMULATOR_CONFIG: Dict[str, Any] = {
    "M": 100,           # Number of simulation paths
    "T": 100,           # Time periods per path
    "s_0": 1.0,         # Initial state value
    "seed": 42,         # Random seed for reproducibility
}


def get_solver_config() -> Dict[str, Any]:
    """
    Return a deep copy of the base solver configuration.
    """
    return deepcopy(_SOLVER_CONFIG)


def get_comparative_statics() -> Dict[str, List[float]]:
    """
    Return the comparative statics grids used in downstream analysis.
    """
    return {key: list(values) for key, values in _COMPARATIVE_STATICS.items()}


def get_simulator_config() -> Dict[str, Any]:
    """
    Return a deep copy of the base simulator configuration.
    """
    return deepcopy(_SIMULATOR_CONFIG)


__all__ = ["get_solver_config", "get_comparative_statics", "get_simulator_config"]

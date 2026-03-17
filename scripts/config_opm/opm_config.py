"""
Central configuration for static oligopoly pricing model workflows.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


_MODEL_CONFIG: Dict[str, Any] = {
    "beta": np.array([1.0, 0.4]),
    "gamma": np.array([0.2, 0.6]),
    "alpha": 1.3,
    "J": 3,
}

_DATA_CONFIG: Dict[str, Any] = {
    "Z": np.array(
        [
            [1.0, 0.2, 0.5],
            [1.5, 0.4, 0.3],
            [1.2, 0.1, 0.7],
        ]
    ),
    "S_X": np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    ),
    "S_W": np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    ),
}

_SOLVER_CONFIG: Dict[str, Any] = {
    "tol": 1e-10,
    "max_iter": 1000,
    "method": "fixed_point",
}

_REPRODUCIBILITY_CONFIG: Dict[str, Any] = {
    "seed": 42,
}

_INITIALIZATION_CONFIG: Dict[str, Any] = {
    "p_init": np.array([1.2, 1.1, 1.0]),
}

_COMPARATIVE_STATICS: Dict[str, Any] = {
    "alpha_values": [0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.2],
    "beta_0_values": [0.3, 0.6, 0.9, 1.0, 1.2, 1.5, 1.8],
    "gamma_1_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.1],
}


def get_model_config() -> Dict[str, Any]:
    return {
        "beta": _MODEL_CONFIG["beta"].copy(),
        "gamma": _MODEL_CONFIG["gamma"].copy(),
        "alpha": float(_MODEL_CONFIG["alpha"]),
        "J": int(_MODEL_CONFIG["J"]),
    }


def get_data_config() -> Dict[str, Any]:
    return {
        "Z": _DATA_CONFIG["Z"].copy(),
        "S_X": _DATA_CONFIG["S_X"].copy(),
        "S_W": _DATA_CONFIG["S_W"].copy(),
    }


def get_solver_config() -> Dict[str, Any]:
    return {
        "tol": float(_SOLVER_CONFIG["tol"]),
        "max_iter": int(_SOLVER_CONFIG["max_iter"]),
        "method": _SOLVER_CONFIG["method"],
    }


def get_reproducibility_config() -> Dict[str, Any]:
    return {
        "seed": int(_REPRODUCIBILITY_CONFIG["seed"]),
    }


def get_initialization_config() -> Dict[str, Any]:
    return {
        "p_init": _INITIALIZATION_CONFIG["p_init"].copy(),
    }


def get_comparative_statics() -> Dict[str, Any]:
    return {
        "alpha_values": list(_COMPARATIVE_STATICS["alpha_values"]),
        "beta_0_values": list(_COMPARATIVE_STATICS["beta_0_values"]),
        "gamma_1_values": list(_COMPARATIVE_STATICS["gamma_1_values"]),
    }


def build_opm_solver_config() -> Dict[str, Any]:
    return {
        "model": get_model_config(),
        "data": get_data_config(),
        "solver": get_solver_config(),
        "reproducibility": get_reproducibility_config(),
        "initialization": get_initialization_config(),
    }


__all__ = [
    "build_opm_solver_config",
    "get_model_config",
    "get_data_config",
    "get_solver_config",
    "get_reproducibility_config",
    "get_initialization_config",
    "get_comparative_statics",
]

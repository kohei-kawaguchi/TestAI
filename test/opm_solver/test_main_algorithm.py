"""
Unit tests for the OPM main solver procedure.

These tests check the pseudocode-level I/O and mathematical implications.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from opm_solver.opm_solver import (
    ComputeFOCResidual,
    ComputeMarginalCost,
    SolveNashEquilibrium,
    ValidateConfig,
)


def make_config() -> dict:
    Z = np.array(
        [
            [1.0, 0.2, 0.5],
            [1.5, 0.4, 0.3],
            [1.2, 0.1, 0.7],
        ]
    )
    S_X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    S_W = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    return {
        "model": {
            "beta": np.array([1.0, 0.4]),
            "gamma": np.array([0.2, 0.6]),
            "alpha": 1.3,
            "J": 3,
        },
        "data": {"Z": Z, "S_X": S_X, "S_W": S_W},
        "solver": {"tol": 1e-10, "max_iter": 1000, "method": "fixed_point"},
        "reproducibility": {"seed": 42},
        "initialization": {"p_init": np.array([1.2, 1.1, 1.0])},
    }


def test_compute_marginal_cost_matches_formula() -> None:
    config = make_config()

    mc = ComputeMarginalCost(config=config)

    W = config["data"]["Z"] @ config["data"]["S_W"]
    expected = W @ config["model"]["gamma"]
    np.testing.assert_allclose(mc, expected)
    assert mc.shape == (3,)


def test_compute_foc_residual_has_expected_shape_and_sign_pattern() -> None:
    config = make_config()
    p = np.array([1.4, 1.3, 1.2])

    residual = ComputeFOCResidual(p=p, config=config)

    assert residual.shape == (3,)
    assert np.all(np.isfinite(residual))


def test_solve_nash_equilibrium_returns_expected_outputs() -> None:
    config = make_config()
    ValidateConfig(config=config)

    p_star, diagnostics = SolveNashEquilibrium(config=config)

    assert isinstance(p_star, np.ndarray)
    assert p_star.shape == (config["model"]["J"],)
    assert set(diagnostics.keys()) == {
        "residual_norm",
        "iterations",
        "converged",
        "solver_status",
    }
    assert diagnostics["iterations"] <= config["solver"]["max_iter"]
    assert diagnostics["residual_norm"] >= 0.0


def test_solved_prices_are_above_marginal_cost() -> None:
    config = make_config()

    p_star, diagnostics = SolveNashEquilibrium(config=config)
    mc = ComputeMarginalCost(config=config)

    assert diagnostics["converged"] is True
    assert np.all(p_star > mc)


def test_solution_satisfies_small_foc_residual() -> None:
    config = make_config()
    tol = config["solver"]["tol"]

    p_star, diagnostics = SolveNashEquilibrium(config=config)
    residual = ComputeFOCResidual(p=p_star, config=config)

    assert np.max(np.abs(residual)) <= max(1e-8, 10.0 * tol)
    assert diagnostics["residual_norm"] <= max(1e-8, 10.0 * tol)

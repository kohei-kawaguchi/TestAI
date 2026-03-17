"""
Unit tests for OPM solver subroutines.

These tests are derived from the pseudocode specification in
scripts/opm_solver/solve_opm.qmd.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from opm_solver.opm_solver import (
    BuildCostShifterIndex,
    BuildDemandShifterIndex,
    BuildDiagnostics,
    ComputeMeanUtility,
    ComputeResidualNorm,
    ComputeShareCoreTerms,
    ComputeShareJacobian,
    ComputeShares,
    ValidateConfig,
)


def make_config() -> dict:
    Z = np.array(
        [
            [1.0, 0.2, 0.5],
            [1.5, 0.4, 0.3],
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
            "beta": np.array([0.7, 0.2]),
            "gamma": np.array([0.3, 0.1]),
            "alpha": 1.1,
            "J": 2,
        },
        "data": {"Z": Z, "S_X": S_X, "S_W": S_W},
        "solver": {"tol": 1e-10, "max_iter": 200, "method": "fixed_point"},
        "reproducibility": {"seed": 123},
        "initialization": {"p_init": np.array([1.3, 1.2])},
    }


def test_build_demand_shifter_index_matches_matrix_product() -> None:
    config = make_config()
    Z = config["data"]["Z"]
    S_X = config["data"]["S_X"]

    X = BuildDemandShifterIndex(Z=Z, S_X=S_X)

    np.testing.assert_allclose(X, Z @ S_X)
    assert X.shape == (2, 2)


def test_build_cost_shifter_index_matches_matrix_product() -> None:
    config = make_config()
    Z = config["data"]["Z"]
    S_W = config["data"]["S_W"]

    W = BuildCostShifterIndex(Z=Z, S_W=S_W)

    np.testing.assert_allclose(W, Z @ S_W)
    assert W.shape == (2, 2)


def test_compute_mean_utility_matches_formula() -> None:
    config = make_config()
    p = np.array([1.4, 1.1])

    delta = ComputeMeanUtility(p=p, config=config)

    X = config["data"]["Z"] @ config["data"]["S_X"]
    expected = X @ config["model"]["beta"] - config["model"]["alpha"] * p
    np.testing.assert_allclose(delta, expected)


def test_compute_share_terms_and_shares_add_up() -> None:
    delta = np.array([-0.4, 0.3, 0.1])

    exp_delta, denom, s_inside = ComputeShareCoreTerms(delta=delta)
    s_inside_2, s_outside = ComputeShares(delta=delta)

    np.testing.assert_allclose(exp_delta, np.exp(delta))
    np.testing.assert_allclose(denom, 1.0 + np.sum(np.exp(delta)))
    np.testing.assert_allclose(s_inside, s_inside_2)
    np.testing.assert_allclose(np.sum(s_inside) + s_outside, 1.0)
    assert 0.0 < s_outside < 1.0


def test_compute_share_jacobian_matches_closed_form() -> None:
    delta = np.array([0.2, -0.1, 0.0])
    alpha = 1.4
    s_inside, _ = ComputeShares(delta=delta)

    jacobian = ComputeShareJacobian(delta=delta, alpha=alpha)
    expected = np.empty((3, 3))
    for j in range(3):
        for k in range(3):
            indicator = 1.0 if j == k else 0.0
            expected[j, k] = -alpha * s_inside[j] * (indicator - s_inside[k])

    np.testing.assert_allclose(jacobian, expected)
    assert np.all(np.diag(jacobian) < 0.0)
    offdiag = jacobian[~np.eye(3, dtype=bool)]
    assert np.all(offdiag > 0.0)


def test_compute_residual_norm_is_max_absolute_value() -> None:
    residual = np.array([0.2, -0.5, 0.49])

    residual_norm = ComputeResidualNorm(residual=residual)

    assert residual_norm == pytest.approx(0.5)


def test_build_diagnostics_has_required_keys() -> None:
    root_result = {
        "solution": np.array([1.0, 2.0]),
        "iterations": 7,
        "converged": True,
        "solver_status": "converged",
        "residual_history": np.array([1e-2, 1e-4, 1e-7]),
    }

    diagnostics = BuildDiagnostics(root_result=root_result, residual_norm=1e-7)

    assert set(diagnostics.keys()) == {
        "residual_norm",
        "iterations",
        "converged",
        "solver_status",
        "residual_history",
    }
    assert diagnostics["iterations"] == 7
    assert diagnostics["converged"] is True
    np.testing.assert_allclose(diagnostics["residual_history"], root_result["residual_history"])


def test_validate_config_accepts_valid_config() -> None:
    config = make_config()
    ValidateConfig(config=config)


def test_validate_config_requires_top_level_keys() -> None:
    config = make_config()
    del config["solver"]

    with pytest.raises(ValueError, match="Missing required config key"):
        ValidateConfig(config=config)


def test_validate_config_checks_dimension_consistency() -> None:
    config = make_config()
    config["model"]["beta"] = np.array([0.5, 0.2, 0.1])

    with pytest.raises(ValueError, match="beta dimension"):
        ValidateConfig(config=config)


def test_validate_config_checks_J_matches_number_of_products() -> None:
    config = make_config()
    config["model"]["J"] = 3

    with pytest.raises(ValueError, match="J must equal"):
        ValidateConfig(config=config)

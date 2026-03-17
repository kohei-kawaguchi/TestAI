"""
Static oligopoly pricing model solver.
"""

from typing import Callable

import numpy as np


def BuildDemandShifterIndex(Z: np.ndarray, S_X: np.ndarray) -> np.ndarray:
    return Z @ S_X


def BuildCostShifterIndex(Z: np.ndarray, S_W: np.ndarray) -> np.ndarray:
    return Z @ S_W


def ComputeMeanUtility(p: np.ndarray, config: dict) -> np.ndarray:
    X = BuildDemandShifterIndex(
        Z=config["data"]["Z"],
        S_X=config["data"]["S_X"],
    )
    return X @ config["model"]["beta"] - config["model"]["alpha"] * p


def ComputeShareCoreTerms(delta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    exp_delta = np.exp(delta)
    denom = 1.0 + np.sum(exp_delta)
    s_inside = exp_delta / denom
    return exp_delta, float(denom), s_inside


def ComputeShares(delta: np.ndarray) -> tuple[np.ndarray, float]:
    _, denom, s_inside = ComputeShareCoreTerms(delta=delta)
    s_outside = 1.0 / denom
    return s_inside, float(s_outside)


def ComputeShareJacobian(delta: np.ndarray, alpha: float) -> np.ndarray:
    _, _, s_inside = ComputeShareCoreTerms(delta=delta)
    J = s_inside.shape[0]
    jacobian = np.empty((J, J))
    for j in range(J):
        for k in range(J):
            indicator = 1.0 if j == k else 0.0
            jacobian[j, k] = -alpha * s_inside[j] * (indicator - s_inside[k])
    return jacobian


def ComputeMarginalCost(config: dict) -> np.ndarray:
    W = BuildCostShifterIndex(
        Z=config["data"]["Z"],
        S_W=config["data"]["S_W"],
    )
    return W @ config["model"]["gamma"]


def ComputeFOCResidual(p: np.ndarray, config: dict) -> np.ndarray:
    delta = ComputeMeanUtility(p=p, config=config)
    s_inside, _ = ComputeShares(delta=delta)
    jacobian = ComputeShareJacobian(
        delta=delta,
        alpha=config["model"]["alpha"],
    )
    mc = ComputeMarginalCost(config=config)
    residual = np.empty_like(p, dtype=float)
    for j in range(p.shape[0]):
        residual[j] = s_inside[j] + (p[j] - mc[j]) * jacobian[j, j]
    return residual


def ComputeResidualNorm(residual: np.ndarray) -> float:
    return float(np.max(np.abs(residual)))


def BuildDiagnostics(root_result: dict, residual_norm: float) -> dict:
    return {
        "residual_norm": float(residual_norm),
        "iterations": int(root_result["iterations"]),
        "converged": bool(root_result["converged"]),
        "solver_status": root_result["solver_status"],
        "residual_history": np.asarray(root_result["residual_history"], dtype=float),
    }


def ValidateConfig(config: dict) -> None:
    required_top = ["model", "data", "solver", "reproducibility", "initialization"]
    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    required_model = ["beta", "gamma", "alpha", "J"]
    required_data = ["Z", "S_X", "S_W"]
    required_solver = ["tol", "max_iter", "method"]
    required_repro = ["seed"]
    required_init = ["p_init"]

    for key in required_model:
        if key not in config["model"]:
            raise ValueError(f"Missing required model key: {key}")
    for key in required_data:
        if key not in config["data"]:
            raise ValueError(f"Missing required data key: {key}")
    for key in required_solver:
        if key not in config["solver"]:
            raise ValueError(f"Missing required solver key: {key}")
    for key in required_repro:
        if key not in config["reproducibility"]:
            raise ValueError(f"Missing required reproducibility key: {key}")
    for key in required_init:
        if key not in config["initialization"]:
            raise ValueError(f"Missing required initialization key: {key}")

    Z = np.asarray(config["data"]["Z"])
    S_X = np.asarray(config["data"]["S_X"])
    S_W = np.asarray(config["data"]["S_W"])
    beta = np.asarray(config["model"]["beta"])
    gamma = np.asarray(config["model"]["gamma"])
    p_init = np.asarray(config["initialization"]["p_init"])
    J = int(config["model"]["J"])
    alpha = float(config["model"]["alpha"])

    if Z.ndim != 2:
        raise ValueError("Z must be a matrix")
    if S_X.ndim != 2:
        raise ValueError("S_X must be a matrix")
    if S_W.ndim != 2:
        raise ValueError("S_W must be a matrix")
    if beta.ndim != 1:
        raise ValueError("beta must be a vector")
    if gamma.ndim != 1:
        raise ValueError("gamma must be a vector")
    if p_init.ndim != 1:
        raise ValueError("p_init must be a vector")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    X = BuildDemandShifterIndex(Z=Z, S_X=S_X)
    W = BuildCostShifterIndex(Z=Z, S_W=S_W)

    if X.shape[1] != beta.shape[0]:
        raise ValueError("beta dimension is inconsistent with Z @ S_X")
    if W.shape[1] != gamma.shape[0]:
        raise ValueError("gamma dimension is inconsistent with Z @ S_W")
    if Z.shape[0] != J:
        raise ValueError("J must equal the number of products in Z")
    if p_init.shape[0] != J:
        raise ValueError("p_init length must equal J")


def SolveNonlinearSystem(
    function: Callable[[np.ndarray, dict], np.ndarray],
    initial_value: np.ndarray,
    function_args: dict,
    method: str,
    tolerance: float,
    max_iterations: int,
) -> dict:
    p = np.array(initial_value, dtype=float)
    residual_history: list[float] = []

    if method != "fixed_point":
        raise ValueError("Only fixed_point method is supported")

    config = function_args["config"]
    alpha = float(config["model"]["alpha"])
    mc = ComputeMarginalCost(config=config)

    for iteration in range(1, max_iterations + 1):
        residual = function(p=p, config=config)
        residual_norm = ComputeResidualNorm(residual=residual)
        residual_history.append(float(residual_norm))
        if residual_norm <= tolerance:
            return {
                "solution": p,
                "iterations": iteration,
                "converged": True,
                "solver_status": "converged",
                "residual_history": np.asarray(residual_history, dtype=float),
            }

        delta = ComputeMeanUtility(p=p, config=config)
        s_inside, _ = ComputeShares(delta=delta)
        p = mc + 1.0 / (alpha * (1.0 - s_inside))

    return {
        "solution": p,
        "iterations": int(max_iterations),
        "converged": False,
        "solver_status": "max_iter_reached",
        "residual_history": np.asarray(residual_history, dtype=float),
    }


def SolveNashEquilibrium(config: dict) -> tuple[np.ndarray, dict]:
    np.random.seed(seed=config["reproducibility"]["seed"])
    ValidateConfig(config=config)

    root_result = SolveNonlinearSystem(
        function=ComputeFOCResidual,
        initial_value=np.asarray(config["initialization"]["p_init"], dtype=float),
        function_args={"config": config},
        method=config["solver"]["method"],
        tolerance=float(config["solver"]["tol"]),
        max_iterations=int(config["solver"]["max_iter"]),
    )
    p_star = np.asarray(root_result["solution"], dtype=float)
    residual = ComputeFOCResidual(p=p_star, config=config)
    residual_norm = ComputeResidualNorm(residual=residual)
    diagnostics = BuildDiagnostics(
        root_result=root_result,
        residual_norm=residual_norm,
    )
    return p_star, diagnostics

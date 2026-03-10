"""Static oligopoly pricing solver package."""

from .opm_solver import (
    BuildCostShifterIndex,
    BuildDemandShifterIndex,
    BuildDiagnostics,
    ComputeFOCResidual,
    ComputeMarginalCost,
    ComputeMeanUtility,
    ComputeResidualNorm,
    ComputeShareCoreTerms,
    ComputeShareJacobian,
    ComputeShares,
    SolveNashEquilibrium,
    SolveNonlinearSystem,
    ValidateConfig,
)

__all__ = [
    "BuildCostShifterIndex",
    "BuildDemandShifterIndex",
    "BuildDiagnostics",
    "ComputeFOCResidual",
    "ComputeMarginalCost",
    "ComputeMeanUtility",
    "ComputeResidualNorm",
    "ComputeShareCoreTerms",
    "ComputeShareJacobian",
    "ComputeShares",
    "SolveNashEquilibrium",
    "SolveNonlinearSystem",
    "ValidateConfig",
]

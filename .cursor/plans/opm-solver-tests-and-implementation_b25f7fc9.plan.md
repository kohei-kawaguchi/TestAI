---
name: opm-solver-tests-and-implementation
overview: Define pseudocode-driven unit tests for the oligopoly pricing solver first, then implement the solver in `src/opm_solver` to satisfy those tests and verify via uv+pytest.
todos:
  - id: extract-spec-contracts
    content: Translate qmd pseudocode into exact Python function contracts and expected invariants
    status: completed
  - id: write-opm-tests-first
    content: Create pytest unit tests in test/opm_solver from pseudocode signatures and mathematical relations
    status: completed
  - id: implement-opm-solver
    content: Implement modular solver and helper procedures in src/opm_solver
    status: completed
  - id: update-public-exports
    content: Switch __init__.py exports to SolveNashEquilibrium API
    status: completed
  - id: run-uv-pytest-and-project-tests
    content: Execute uv pytest for opm tests, then run ./run.sh test and summarize
    status: completed
isProject: false
---

# Implement OPM Solver From Pseudocode

## Scope and decisions

- Public API will follow the pseudocode: expose `SolveNashEquilibrium(config)` and remove the old placeholder API.
- Tests will be specification-first: derived from [scripts/opm_solver/solve_opm.qmd](/Users/koheikawaguchi/Documents/TestAI/scripts/opm_solver/solve_opm.qmd), not current implementation behavior.

## Files to update

- [src/opm_solver/opm_solver.py](/Users/koheikawaguchi/Documents/TestAI/src/opm_solver/opm_solver.py)
- [src/opm_solver/**init**.py](/Users/koheikawaguchi/Documents/TestAI/src/opm_solver/__init__.py)
- [test/opm_solver/test_subroutines.py](/Users/koheikawaguchi/Documents/TestAI/test/opm_solver/test_subroutines.py) (new)
- [test/opm_solver/test_main_algorithm.py](/Users/koheikawaguchi/Documents/TestAI/test/opm_solver/test_main_algorithm.py) (new)

## Implementation plan

- Extract exact formulas and procedure signatures from the pseudocode into concrete function contracts for:
  - `BuildDemandShifterIndex`, `BuildCostShifterIndex`
  - `ComputeMeanUtility`, `ComputeShareCoreTerms`, `ComputeShares`, `ComputeShareJacobian`
  - `ComputeMarginalCost`, `ComputeFOCResidual`, `ComputeResidualNorm`, `BuildDiagnostics`, `ValidateConfig`
  - `SolveNashEquilibrium`
- Write tests first in `test/opm_solver`:
  - Signature and shape tests for all core procedures.
  - Mathematical identity tests implied by pseudocode:
    - share add-up (`sum(s_inside) + s_outside = 1`)
    - Jacobian own/cross derivative signs and formula consistency
    - residual norm equals max absolute component
    - FOC residual behavior for a simple synthetic configuration
  - Main solver output contract tests:
    - returns `(p_star, diagnostics)` with expected shapes/keys
    - diagnostics contains `residual_norm`, `iterations`, `converged`, `solver_status`
- Implement `src/opm_solver/opm_solver.py` to match pseudocode exactly and keep a modular DRY structure.
- Update `src/opm_solver/__init__.py` exports to the new pseudocode API.

## Verification

- Run targeted tests in uv environment with pytest:
  - `uv run pytest test/opm_solver -q`
- Then run broader project tests:
  - `./run.sh test`
- Report test results and any remaining gaps.


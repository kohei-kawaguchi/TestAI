# Problem Set 1: Static Oligopoly Pricing Model

This problem set extends the existing MDP training pipeline to a static oligopoly pricing environment so that juniors internalize each stage—configuration, solving, simulation, and estimation—by rebuilding them for a new economic model.

## Learning Objectives

1. **Model specification**: Translate the familiar dynamic MDP primitives into a differentiated-product Bertrand setup (demand system, cost structure, ownership matrix, conduct assumptions).
2. **Algorithm design**: Replace value iteration with equilibrium price routines (markup fixed points, Newton steps on FOC residuals) while keeping the same level of rigor in documentation, pseudo code, and convergence diagnostics.
3. **Monte Carlo simulation**: Generate repeated markets using the solved pricing game, track equilibrium outcomes, and design diagnostics analogous to the state/action summaries in the MDP simulator.
4. **Estimation workflow**: Implement the demand-GMM-plus-cost-recovery sequence and highlight where the nested fixed point logic from the MDP estimator still applies.

## Required Deliverables

- `problems/config_opm/config_opm.qmd`: Guidance for building an `opm_config.py` module that exposes demand, cost, and algorithm settings as data.
- `problems/solve_opm/solve_opm.qmd`: Instructional solver notebook describing the static pricing equilibrium problem, pseudo code, and validation plan.
- `problems/simulate_opm/simulate_opm.qmd`: Simulation brief explaining how to reuse solver outputs to generate market panels, monitor convergence, and summarize outcomes.
- `problems/estimate_opm/estimate_opm.qmd`: Estimation brief outlining the two-step procedure (demand estimation + cost recovery) and required diagnostics.

Each document is intentionally code-free—juniors must fill in all implementations, figures, and tables. Supervisors can evaluate submissions by checking whether trainees adhered to the structure, reproduced the MDP rigor, and justified modeling choices specific to static oligopoly pricing.

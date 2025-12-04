# TestAI

A research project implementing statistical analysis and dynamic optimization methods, including causal inference methods in R and Markov Decision Process solvers in Python.

## Project Structure

```
TestAI/
├── R/                          # R package source code
│   ├── functions_did.R         # Difference-in-differences functions
│   ├── functions_randomization.R  # Randomization inference functions
│   └── ...                     # Other statistical methods
├── src/                        # Python package source code
│   └── mdp_solver/             # Markov Decision Process solver
│       ├── __init__.py
│       └── mdp_solver.py       # Neural network-based value iteration
├── test/                       # Python unit tests
│   └── mdp_solver/             # MDP solver tests
│       ├── test_subroutines.py
│       └── test_value_iteration.py
├── scripts/
│   ├── analyze_data/           # AI-generated analysis implementations
│   │   ├── analyze_did_multiperiods.Rmd
│   │   └── randomization_fisher_pvalue.Rmd
│   ├── analyze_data_true/      # Ground truth reference implementations
│   │   ├── difference_in_differences_multiperiods.Rmd
│   │   ├── randomization_fisher_pvalue.Rmd
│   │   └── compare_results.R   # Validation script
│   └── solve_mdp/              # MDP solver documentation
│       ├── solve_mdp.qmd       # Quarto document with implementation
│       └── solve_mdp.html      # Rendered HTML report
├── problems/                   # Junior training problem sets
│   ├── solve_opm/              # Static oligopoly solver brief (Quarto scaffold)
│   ├── simulate_opm/           # Static oligopoly simulator brief
│   ├── estimate_opm/           # Static oligopoly estimator brief
│   └── config_opm/             # Configuration guidance for the OPM problem set
├── output/                     # Generated datasets and results
│   ├── difference_in_differences_multiperiods/
│   └── randomization_fisher_pvalue/
└── docs/
    └── conversation/           # Conversation transcripts for training
        ├── did_multiperiod_analysis_conversation_transcript.md
        ├── randomization_inference_conversation_transcript.md
        └── ground_truth_comparison_conversation_transcript.md
```

## Features

### Implemented Methods

1. **Difference-in-Differences (DiD)** (R)
   - Callaway-Sant'Anna estimator for multi-period DiD
   - Support for covariates and multiple treatment periods
   - Robust standard errors with clustered structure

2. **Randomization Inference** (R)
   - Fisher's exact p-value computation
   - Randomization-based hypothesis testing
   - Support for various test statistics

3. **Markov Decision Process Solver** (Python)
   - Neural network-based value function approximation using PyTorch
   - Value iteration algorithm with Bellman operator
   - Type-I extreme value distributed action shocks
   - Continuous state space with discrete binary actions
   - Logarithmic reward function: r(s,a) = β·log(1+s) - a
   - State transition: s' = (1-γ)s + a
   - Monotonic neural network architecture with softplus constraints
   - Comprehensive unit tests covering all subroutines
   - Comparative statics analysis for key parameters (β, γ)

### Datasets

The project includes simulated datasets with:

**DiD Multi-Period Datasets:**
- **df_design_cov**: Panel data with covariates affecting treatment assignment and trends
- **df_design_nocov**: Panel data with unconditional parallel trends
- **df_design_nyt**: Panel data with no never-treated units (all eventually treated)

**Randomization Inference Dataset:**
- **data_realized**: Cross-sectional randomized experiment data with binary treatment assignment

## Usage

### Installation

#### R Package
```r
# Install required packages
install.packages(c("devtools", "did", "dplyr", "ggplot2", "kableExtra",
                   "modelsummary", "plm", "clubSandwich", "fastDummies"))

# Load the package
devtools::load_all()
```

#### Python Package
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch numpy matplotlib pytest
```

### Running Analyses

#### R Analyses
```r
# Run DiD multi-period analysis
rmarkdown::render('scripts/analyze_data/analyze_did_multiperiods.Rmd')

# Run randomization inference analysis
rmarkdown::render('scripts/analyze_data/randomization_fisher_pvalue.Rmd')

# Validate against ground truth
source('scripts/analyze_data_true/compare_results.R')
```

#### MDP Pipeline (Solver → Simulator → Estimator)

Run the complete MDP pipeline sequentially:

```bash
# Run all three steps: solver, simulator, and estimator
./scripts/run_mdp.sh
```

This script will:
1. Solve the MDP using value iteration (saves to `output/solve_mdp/`)
2. Simulate data from the solved model (saves to `output/simulate_mdp/`)
3. Estimate parameters from simulated data (saves to `output/estimate_mdp/`)

Or run individual steps:

**MDP Solver**
```bash
# Run unit tests
uv run pytest test/

# Render solver report
cd scripts/solve_mdp
uv run quarto render solve_mdp.qmd

# Launch live preview (serves at http://127.0.0.1:4200 after port-forwarding)
uv run quarto preview scripts/solve_mdp/solve_mdp.qmd
```

**MDP Simulator**
```bash
# Render simulator report (requires solver output)
cd scripts/simulate_mdp
uv run quarto render simulate_mdp.qmd

# Launch live preview (serves at http://127.0.0.1:4300 after port-forwarding)
uv run quarto preview scripts/simulate_mdp/simulate_mdp.qmd
```

**MDP Estimator**
```bash
# Render estimator report (requires simulator output)
cd scripts/estimate_mdp
uv run quarto render estimate_mdp.qmd

# Launch live preview (serves at http://127.0.0.1:4400 after port-forwarding)
uv run quarto preview scripts/estimate_mdp/estimate_mdp.qmd
```

**Container workflow:**
When working inside a container (e.g., VS Code Dev Containers), forward ports `4200`, `4300`, and `4400` to access the live previews for solver, simulator, and estimator respectively.

**Quarto workflow with cached figures:**
- Always use the project environment: `uv run` or prefix commands with `QUARTO_PYTHON=.venv/bin/python`
- Render once with execution enabled: `uv run quarto render <document.qmd>`
- For previews, reuse the cache: `uv run quarto preview <document.qmd>`
- Avoid `--no-execute` unless you only need text; it strips figure references from the regenerated HTML
- To share a static copy, bundle the HTML with `*_files/` or render with `--self-contained`

## Validation Results

### DiD Implementation Validation

Comparison of AI-generated implementations against manual ground truth:

| Dataset | Ground Truth ATT | AI Implementation ATT | Difference | Status |
|---------|------------------|----------------------|------------|--------|
| Dataset 1 (with covariates) | 0.030235 | 0.030235 | 0.000000 | ✓ Perfect Match |
| Dataset 2 (without covariates) | 0.028018 | 0.028018 | 0.000000 | ✓ Perfect Match |
| Dataset 3 (not-yet-treated) | 0.034936 | 0.036461 | 0.001526 | ✓ Correct (AI includes covariates) |

See [docs/conversation/ground_truth_comparison_conversation_transcript.md](docs/conversation/ground_truth_comparison_conversation_transcript.md) for details.

### MDP Solver Test Results

All 45 unit tests pass, covering:
- Mean reward computation with logarithmic form
- State transition dynamics
- Bellman operator application
- Choice probability calculation with logit structure
- Neural network monotonicity constraints
- Value iteration convergence
- End-to-end value function optimization

The solver successfully produces:
- Converged value functions for both actions (a=0, a=1)
- Optimal state-dependent policies exhibiting threshold structure
- Comparative statics showing intuitive parameter effects on value and policy

## Junior Training Problem Set

The `problems/` directory contains Quarto scaffolds that challenge junior researchers to rebuild the MDP pipeline for a **static oligopoly pricing model**:

- `problems/solve_opm/solve_opm.qmd` documents the expected equilibrium solver for differentiated-product Bertrand competition.
- `problems/simulate_opm/simulate_opm.qmd` specifies how to run market-level Monte Carlo simulations using the oligopoly solver outputs.
- `problems/estimate_opm/estimate_opm.qmd` outlines the demand-and-cost estimation workflow that mirrors the MDP NFXP estimator.
- `problems/config_opm/config_opm.qmd` describes the configuration module that treats demand, cost, and algorithm settings as data.

Each report is intentionally code-free; it states the learning goals, required components, and deliverables so juniors can translate the existing MDP materials into the static oligopoly setting.

## Documentation

### Conversation Transcripts

Conversation transcripts documenting the development process are available in [docs/conversation/](docs/conversation/):
- **DiD Multi-Period Analysis**: Development of Callaway-Sant'Anna estimator implementation
- **Randomization Inference**: Fisher's exact p-value computation
- **Ground Truth Comparison**: Validation methodology and results

### Presentation

A presentation on using AI as a research assistant for economic analysis is available at [docs/slide/use_ai_as_ra.md](docs/slide/use_ai_as_ra.md). This Marp-format slide deck covers:
- Best practices for supervising AI research assistants
- Workflow for solving and simulating economic models
- Model-free analysis and data cleaning approaches
- Tool recommendations and practical tips

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

## License

This is a research project for educational and training purposes.

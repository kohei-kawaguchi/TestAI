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

#### MDP Solver
```bash
# Run unit tests
uv run pytest test/

# Launch live preview (serves at http://127.0.0.1:4200 after port-forwarding)
uv run quarto preview scripts/solve_mdp/solve_mdp.qmd

# Render Quarto documentation with results
cd scripts/solve_mdp
uv run quarto render solve_mdp.qmd

# View the rendered HTML report
# Open scripts/solve_mdp/solve_mdp.html in your browser
```

When working inside a container (e.g., VS Code Dev Containers), forward port `4200` and open `http://127.0.0.1:4200` in the VS Code Simple Browser or your local browser. The live server command above reuses cached computations (`_freeze/`) unless you edit the document code, so previewing is lightweight.

**Quarto workflow with cached figures**
- Always use the project environment: `source .venv/bin/activate` or prefix commands with `QUARTO_PYTHON=.venv/bin/python`.
- Render once with execution enabled: `quarto render scripts/solve_mdp/solve_mdp.qmd --no-browser --no-watch-inputs`.
- For previews, reuse the cache: `QUARTO_PYTHON=.venv/bin/python quarto preview scripts/solve_mdp/solve_mdp.qmd --no-browser --no-watch-inputs`.
- Avoid `--no-execute` unless you only need text; it strips figure references from the regenerated HTML.
- To share a static copy, bundle the HTML with `solve_mdp_files/` or render with `--self-contained`.

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

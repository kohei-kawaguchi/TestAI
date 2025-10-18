# TestAI

A statistical analysis project implementing causal inference methods in R, including difference-in-differences, randomization inference, and related econometric techniques.

## Project Structure

```
TestAI/
├── R/                          # R package source code
│   ├── functions_did.R         # Difference-in-differences functions
│   ├── functions_randomization.R  # Randomization inference functions
│   └── ...                     # Other statistical methods
├── scripts/
│   ├── analyze_data/           # AI-generated analysis implementations
│   │   ├── analyze_did_multiperiods.Rmd
│   │   └── randomization_fisher_pvalue.Rmd
│   └── analyze_data_true/      # Ground truth reference implementations
│       ├── difference_in_differences_multiperiods.Rmd
│       ├── randomization_fisher_pvalue.Rmd
│       └── compare_results.R   # Validation script
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

1. **Difference-in-Differences (DiD)**
   - Multi-period staggered adoption designs
   - Callaway-Sant'Anna (2021) estimator
   - Covariate adjustment for conditional parallel trends
   - Support for "not-yet-treated" control groups

2. **Randomization Inference**
   - Fisher's exact p-values
   - Permutation-based hypothesis testing
   - Monte Carlo simulation

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

```r
# Install required packages
install.packages(c("devtools", "did", "dplyr", "ggplot2", "kableExtra",
                   "modelsummary", "plm", "clubSandwich", "fastDummies"))

# Load the package
devtools::load_all()
```

### Running Analyses

```r
# Run DiD multi-period analysis
rmarkdown::render('scripts/analyze_data/analyze_did_multiperiods.Rmd')

# Run randomization inference analysis
rmarkdown::render('scripts/analyze_data/randomization_fisher_pvalue.Rmd')

# Validate against ground truth
source('scripts/analyze_data_true/compare_results.R')
```

## Validation Results

Comparison of AI-generated implementations against manual ground truth:

| Dataset | Ground Truth ATT | AI Implementation ATT | Difference | Status |
|---------|------------------|----------------------|------------|--------|
| Dataset 1 (with covariates) | 0.030235 | 0.030235 | 0.000000 | ✓ Perfect Match |
| Dataset 2 (without covariates) | 0.028018 | 0.028018 | 0.000000 | ✓ Perfect Match |
| Dataset 3 (not-yet-treated) | 0.034936 | 0.036461 | 0.001526 | ✓ Correct (AI includes covariates) |

See [docs/conversation/ground_truth_comparison_conversation_transcript.md](docs/conversation/ground_truth_comparison_conversation_transcript.md) for details.

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
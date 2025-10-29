#!/usr/bin/env bash

# Run MDP solver, simulator, and estimator sequentially
# Usage: ./scripts/run_mdp.sh

set -e  # Exit on error

echo "========================================="
echo "Running MDP Pipeline"
echo "========================================="

# Navigate to project root
cd "$(dirname "$0")/.."

echo ""
echo "[1/3] Running MDP Solver..."
echo "-----------------------------------------"
cd scripts/solve_mdp
uv run quarto render solve_mdp.qmd
cd ../..
echo "✓ Solver completed"

echo ""
echo "[2/3] Running MDP Simulator..."
echo "-----------------------------------------"
cd scripts/simulate_mdp
uv run quarto render simulate_mdp.qmd
cd ../..
echo "✓ Simulator completed"

echo ""
echo "[3/3] Running MDP Estimator..."
echo "-----------------------------------------"
cd scripts/estimate_mdp
uv run quarto render estimate_mdp.qmd
cd ../..
echo "✓ Estimator completed"

echo ""
echo "========================================="
echo "MDP Pipeline completed successfully!"
echo "========================================="
echo ""
echo "Output locations:"
echo "  Solver:     output/solve_mdp/"
echo "  Simulator:  output/simulate_mdp/"
echo "  Estimator:  output/estimate_mdp/"

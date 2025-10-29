# Leveraging AI as a Research Assistant: Inverse Reinforcement Learning Implementation for MDPs

## Introduction

In our [previous article]([mdp_solver_en.md](https://note.com/mixingale/n/nd1d7f06e9d5d)), we demonstrated how AI can implement Markov Decision Process (MDP) solvers for sequential decision-making problems. This time, we complete the pipeline by adding **simulation and inverse reinforcement learning** to create a full end-to-end workflow.

The key is completing the **solve → simulate → estimate → recover** cycle. We generate data with known parameters, estimate those parameters from the data, and verify we can accurately recover the original values. This is **the most comprehensive validation of correct implementation**.

## Parameter Recovery as the Most Comprehensive Validation

### Why Parameter Recovery Matters

Unit tests verify "does this function work correctly?" Comparative statics verify "do results align with theory when parameters change?" But these alone are insufficient.

Parameter recovery is **end-to-end system validation**. We solve the model with true parameters, generate data, and estimate parameters from that data. The entire pipeline must function correctly for accurate parameter recovery.

**A bug anywhere breaks parameter recovery.**

- Solver bug → incorrect equilibrium → incorrect data → incorrect estimates
- Simulator bug → data inconsistent with model → biased estimates
- Estimator bug → incorrect estimates even from correct data

Thus, successful parameter recovery proves that **the solver, simulator, and estimator are all correctly implemented and working consistently together**.

### It Won't Succeed on the First Try

An important point: **parameter recovery won't succeed on the first attempt**.

In this implementation, we discovered and fixed several bugs. A staged approach (validating each component independently before integration) is essential. When we finally confirmed the estimator's objective function was optimized around true values, we gained confidence in our implementation.

## Pipeline Management for Data, Configuration, and Functions

In multi-step pipelines, **consistent management of data, configuration, and functions** is essential. This is the foundation for successful parameter recovery.

### Treating Configuration as Data

The most important design principle in this project: **"treat configuration as data, not code"**.

Some people hardcode parameters in code. The parameters used in the solver get rewritten in the simulator, and again in the estimator. This breeds configuration mismatch bugs.

This project enforces these rules:

1. **Define all parameters in one place**
   The configuration module is the single source of truth

2. **Save configuration with results**
   The solver saves configuration as JSON, and subsequent steps load it

3. **Automatically validate configuration consistency**
   Verify loaded configuration matches current code configuration. Error on mismatch

This approach makes it impossible for "the simulator ran with different parameters" or "the estimator used wrong settings" to occur.

### DRY Principle: Don't Repeat Yourself

The **DRY (Don't Repeat Yourself) principle** is also crucial.

For example, implementing the reward function separately in the solver, simulator, and estimator means the same code exists in three places. When changing the reward function, all three must be updated, and missing one causes bugs.

This project **consolidates all functions in shared modules**. Reward functions, state transitions, visualization: everything exists in only one place. Changes happen in one location and automatically propagate to all steps.

This design makes it impossible for "the solver and estimator had different reward function definitions" to occur.

### Pipeline Design

The data flow is linear:

- **Solver** → saves trained model + configuration
- **Simulator** → loads model and configuration, saves data + configuration
- **Estimator** → loads data and configuration, estimates parameters

Each step runs independently and depends only on the previous step's output. This design allows re-running any step and facilitates debugging.

## Bug Isolation Through Diagnostic Analysis

When parameter recovery fails, **identifying where the bug exists** is the biggest challenge.

### Problem Isolation

There are three possible sources for parameter recovery failure:

1. **Solver bug**: computing incorrect equilibrium
2. **Simulator bug**: generating data inconsistent with model
3. **Estimator bug**: producing incorrect estimates even from correct data

But looking at estimates alone doesn't reveal where the problem lies.

### Inserting Diagnostic Analysis

Therefore, we **insert diagnostic analysis into the pipeline**.

The ultimate verification is **plotting likelihood profiles around true parameters**. We solve the model at true parameters and compute the likelihood of simulation data using that value function.

When this fails, we need to isolate whether the problem is in the solver, simulator, or estimator. If we've validated the solver and simulator before implementing the estimator, we can focus on debugging the estimator. Without that validation, we face inefficient investigation across vast areas.

For estimator debugging, we start from data and configuration loading, then examine algorithm-level issues, then implementation-level issues sequentially. When meaningful routines are modularized as functions, reusability increases and debugging becomes easier. If code isn't organized this way, problem isolation and diagnostics fail, so code cleanup must come first.

## Summary: Most Comprehensive Validation Through Parameter Recovery

This implementation case study demonstrates a methodology for having AI implement the complete **solve → simulate → estimate → recover** computational pipeline. Parameter recovery doesn't succeed on the first attempt. Through iterative bug isolation and fixing, we ultimately confirm implementation correctness.

### Core Principles

1. **Parameter recovery is the most comprehensive validation**
   Unit tests and comparative statics alone are insufficient. Only by generating data with true parameters, estimating, and recovering parameters can we claim the entire pipeline is correct.

2. **Treat configuration as data**
   Don't hardcode in code. Save configuration as JSON and load/validate at each step. Auto-detect configuration mismatches to prevent bugs. This is the foundation for successful parameter recovery.

3. **Enforce DRY principle**
   Don't write the same code twice. Reward functions, state transitions, visualization: consolidate everything in shared modules. Changes in one place propagate to all steps, fundamentally preventing implementation inconsistencies.

4. **Bug isolation through diagnostic analysis**
   When parameter recovery fails, we must identify where the bug exists. Modularize functions, insert diagnostic analysis, plot likelihood profiles at true parameters to isolate whether the problem is in the solver, simulator, or estimator.

### Extension to More Complex Problems

The same methodology applies to more complex problems. Build the solve → simulate → estimate → recover pipeline, share configuration and functions at each step, and validate through parameter recovery. This principle applies to any computational economics project.

### Complete Public Implementation

Detailed conversation history, all code, and rendered HTML reports are available in the [GitHub repository](https://github.com/kohei-kawaguchi/TestAI). In particular, [`docs/conversation/mdp_simulator_estimator_conversation_transcript.md`](../../conversation/mdp_simulator_estimator_conversation_transcript.md) contains the conversation history with decision criteria recorded at each stage.

### Rendered Reports

Detailed analysis results from each step are available in these HTML reports:

- [MDP Solver](https://kohei-kawaguchi.github.io/TestAI/solve_mdp.html): Equilibrium computation via value function approximation
- [MDP Simulator](https://kohei-kawaguchi.github.io/TestAI/simulate_mdp.html): Agent behavior simulation using equilibrium value function
- [MDP Estimator](https://kohei-kawaguchi.github.io/TestAI/estimate_mdp.html): Parameter estimation from simulation data and recovery validation

# Leveraging AI as a Research Assistant: A Case Study in Reinforcement Learning Implementation

## Introduction

In our [previous article]([causal_inference_en.md](https://note.com/mixingale/n/nb7dfcc3ea18a)), we demonstrated how AI can effectively implement causal inference methods using an "oral exam" approach. This time, we tackle another fundamental task in data science: **implementing reinforcement learning algorithms**.

Specifically, we had AI implement a solver for **Markov Decision Processes (MDPs)**, the mathematical foundation of reinforcement learning. MDPs model sequential decision-making under uncertainty—from training game-playing agents to optimizing recommendation systems, from inventory management to autonomous control. Understanding how to implement MDP solvers is essential for data scientists working with any form of sequential decision problems.

More importantly, understanding this basic MDP framework opens the door to advanced reinforcement learning techniques: Q-learning, policy gradients, actor-critic methods, and deep reinforcement learning. The methodology remains consistent: **mathematical formulation → algorithm design → implementation → verification → validation**.

## Making Mathematical Formulas the Single Source of Truth

We take the oral exam approach from the previous article one step further. **We explicitly document the model setup and solution algorithm using mathematical formulas and pseudocode, treating them as the single source of truth for all implementation work**.

### Why Mathematical Formulas Are Essential

For equilibrium computation, "what to compute" is not obvious to AI. Bellman equations, value functions, optimal policies—if these mathematical definitions remain ambiguous, AI will head in the wrong direction.

Therefore, before implementation, we clarify:

1. **Complete mathematical formulation**: Describe states, actions, rewards, and transitions with formulas
2. **Pseudocode for the solution algorithm**: Express computational steps in an intermediate language between math and programming
3. **Use pseudocode as the implementation specification**: AI writes code following this blueprint
4. **Generate unit tests from pseudocode**: Verify that each step is correctly implemented

The greatest advantage of this approach is **easy change management**. For example, when modifying the reward function:

1. First, update the formulas (documentation level)
2. Next, update the pseudocode (specification level)
3. Then, update the implementation (code level)
4. Finally, update the tests (verification: does it work as specified?)

Following this sequence prevents documentation-implementation divergence. Explicitly instructing AI "do not change the code yet" is also crucial.

## Pseudocode Quality Determines Success

While you could have AI write code directly from formulas, this eliminates intermediate verification. Therefore, **we place "pseudocode" as a bridge between formulas and code**.

Pseudocode combines elements of mathematics and programming. It is mathematically rigorous while being written in a form that is implementable programmatically.

### Quality Requirements for Pseudocode

AI's initial pseudocode often contains ambiguous descriptions—placeholders like `// Implementation details` or `// TODO: Compute targets`.

These must **all be eliminated**. Pseudocode is not a memo for "what to implement later," but a specification that contains "enough information for AI to implement."

Specifically: zero placeholders, explicit type information, small function decomposition, and verb-based naming (ComputeX, FitY, SolveZ). All function calls must use named arguments: `func(beta=0.5, gamma=0.1)`.

Parameter order bugs occur frequently for both AI and humans. Making this a project rule (in CLAUDE.md) ensures consistency.

### Generating Implementation and Tests from Pseudocode

Once pseudocode is complete, implementation and tests can be mechanically derived. Each pseudocode procedure corresponds to a Python function and a test class. Test content is also derived from pseudocode. If the specification says "return r(s,a) = β·log(1+s) - a," we test whether the function output matches the value computed by that formula.

In this project, we created 45 unit tests and confirmed all passed. This guarantees that each computational step operates according to specification.

## The Importance of Staged Change Management

During implementation, we needed to change the reward function form—from simple linear (β·s - a) to logarithmic with diminishing returns (β·log(1+s) - a).

The critical point here is **the order of changes**. First, update the mathematical documentation, add economic interpretation, update the Bellman equation, and update the pseudocode. Then, explicitly instruct AI: "do not change the implementation yet." Only after obtaining user permission do we update the implementation, followed by updating the tests.

AI will rewrite code immediately if instructed, but this breaks synchronization with documentation. Explicitly saying "wait" enables staged updates.

## Validation Through Comparative Statics

In model implementation verification, **comparative statics analysis is indispensable**.

Unit tests verify "whether individual functions work correctly." However, this alone is insufficient. We need to confirm that the model as a whole "produces economically meaningful results."

Therefore, we systematically vary key parameters and verify that results are consistent with theoretical predictions.

### Why Comparative Statics Is Essential

In our MDP problem, we focused on two parameters: β (reward weight) and γ (depreciation rate). Systematically varying these revealed that increasing β raises the value function and produces state-dependent threshold policies, while increasing γ reduces the value function and shifts toward replacement-type policies.

These results are economically sound. If we observed the opposite (β increasing reduces value, etc.), there would be an implementation error somewhere.

Comparative statics discovers **model-level logic errors that unit tests cannot catch**. Even if individual functions are correct, incorrect combinations will produce anomalous comparative statics results.

### The Importance of Visualization

We visualized comparative statics results by overlaying value functions and policies for multiple parameter values. Using color gradients (black→blue, black→red as parameters increase) makes continuous changes intuitively understandable.

Beyond quantitative numbers, visually confirming patterns deepens confidence in implementation validity.

## Recording Conversation History for Post-Hoc Learning

As before, we completely recorded the conversation history with AI and published it as [`docs/conversation/mdp_solver_conversation_transcript.md`](../../conversation/mdp_solver_conversation_transcript.md).

Conversation history has value beyond mere documentation. It ensures project reproducibility (what instructions produced these results), enables knowledge sharing (what questions to ask at each stage), and functions as a self-improvement tool (have AI evaluate conversation history and git commit history to improve supervision methods).

This conversation history is structured into 11 phases. Each phase has a "Key Learning" section recording why certain choices were made (e.g., why enforce named arguments).

## Summary: Methodology for AI-Driven Model Implementation

This implementation case study advances beyond the previous causal inference work to demonstrate a methodology for having AI implement **equilibrium computation**.

### Core Principles

1. **Mathematical formulas as single source of truth**
   Create a consistent flow: mathematical formulation → pseudocode → implementation → tests. Always start changes upstream (formulas/pseudocode).

2. **Pseudocode quality control**
   Eliminate placeholders, specify types explicitly, decompose into small functions. Pseudocode is not a "to-do" memo but an "AI-implementable specification."

3. **Staged change management**
   AI rewrites code immediately when instructed. Therefore, explicitly say "wait" and maintain documentation→implementation order.

4. **Validation through comparative statics**
   Unit tests guarantee function-level correctness. Comparative statics verify model-wide economic validity. Both are necessary.

5. **Named arguments and project rules**
   Parameter order errors are frequent. Codify this in CLAUDE.md and enforce it with AI.

6. **Recording conversation history**
   Not just for reproducibility, but as a self-improvement tool. Have AI evaluate conversation history and git commits to improve supervision methods.

### Extension to More Complex Problems

The individual decision problem we addressed is the basic form of dynamic optimization. However, the same methodology applies to more complex problems:

- **Game theory**: Strategic interactions among multiple players
- **Incomplete information**: Signals and belief updating
- **Heterogeneous agents**: Equilibria across diverse decision-makers

The concept remains the same: formulate the problem with formulas, design the solution with pseudocode, implement in stages, and validate with comparative statics.

### Complete Public Implementation

All materials are available in the [GitHub repository](https://github.com/kohei-kawaguchi/TestAI). In particular, `docs/conversation/mdp_solver_conversation_transcript.md` contains the complete conversation history across 11 phases, with decision criteria recorded as "Key Learning" at each stage.

You can explore the live implementation results on [GitHub Pages](https://kohei-kawaguchi.github.io/TestAI/):
- [MDP Solver Report](https://kohei-kawaguchi.github.io/TestAI/solve_mdp.html): Interactive HTML report with value function plots and comparative statics analysis
- [Difference-in-Differences Analysis](https://kohei-kawaguchi.github.io/TestAI/analyze_did_multiperiods.html): Multi-period DiD implementation from the previous article
- [Neyman's Average Treatment Effect](https://kohei-kawaguchi.github.io/TestAI/randomization_fisher_pvalue.html): Randomized experiment analysis

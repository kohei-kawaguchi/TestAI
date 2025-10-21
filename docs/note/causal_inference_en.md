# Leveraging AI as a Research Assistant: A Case Study in Causal Inference Implementation

## Introduction

Many data scientists are familiar with these challenges: waiting days to receive results after assigning analysis tasks to a junior analyst, or requiring multiple rounds of communication to fix code bugs.

This repository "[TestAI](https://github.com/kohei-kawaguchi/TestAI)" demonstrates that **on-demand data analysis tasks can be completely replaced by AI assistants**. We validated this possibility by having AI implement Staggered Difference-in-Differences (DiD) analysis, an advanced causal inference method used to measure treatment effects when interventions are rolled out at different times.

## Why AI Can Effectively Replace Manual Analysis Work

On-demand data analysis tasks are particularly well-suited for AI assistance for three reasons. First, the tasks can be clearly defined using statistical methods and mathematical formulas. Second, since you are the one using the results, extensive customization is not necessary. Third, for exploratory or one-time analyses, long-term code maintainability is less critical.

If you provide instructions as carefully as you would when supervising a junior analyst, AI will deliver the expected results. Moreover, with faster iteration and the ability to run multiple tasks in parallel, your analysis workflow becomes dramatically more efficient.

## The Key: "Oral Exam" Approach

Before requesting AI to implement anything, there is a critical step you must take: conduct an **"oral exam"** to verify the AI's understanding.

If you immediately instruct "Please implement Staggered DiD," there's a possibility that the AI doesn't correctly understand the statistical method or its assumptions. Complex analytical concepts—such as parallel trends assumptions, treatment effect heterogeneity, or multiple testing corrections—may not be immediately clear to the AI. Therefore, just as you would conduct an oral exam with a junior analyst, you should test the AI's knowledge first.

Example of actual conversation:

- "In a panel data setting with staggered treatment timing, do you know how to estimate causal effects?"
- AI responds: "We can use the Callaway and Sant'Anna (2021) method"
- "Then, please find the R package, read the documentation, and summarize how to use it"
- AI reads the package documentation and returns an explanation

By going through this oral exam process and having the AI document the results as an analysis plan, you create a single source of truth that ensures the AI has the correct statistical foundation before beginning implementation. If you proceed with incorrect understanding, later corrections become difficult and time-consuming.

## Implementing Staggered DiD in Practice

In this repository, we implemented the Callaway and Sant'Anna (2021) Staggered DiD method. This is a technically complex causal inference method that estimates treatment effects in longitudinal data where different groups receive interventions at different times—common in real-world scenarios like product feature rollouts, policy implementations, or A/B test deployments.

We tested the implementation using three datasets from the difference-in-differences chapter of [The Econometrician's Guide to Causal Inference](https://github.com/keisemi/EconometriciansGuide_CausalInference):

We validated the approach across three different scenarios:

1. **With control variables, control group never receives treatment**: The most standard setting in observational studies
2. **Without control variables**: Requires special specification `xformla = ~1` in the implementation
3. **All units eventually receive treatment**: Requires changing the setting to `control_group = "notyettreated"` to handle the absence of a pure control group

After confirming the method's understanding through the oral exam, the AI generated appropriate code for all three scenarios.

## Documenting the AI Interaction Process

An important feature of this repository is that **key points from the conversation history with AI are recorded and published**. The `docs/conversation/` directory contains detailed records of what instructions were given, how the AI responded, and where oral exams were conducted.

These records have value beyond mere documentation. They concretely demonstrate what types of prompts promote correct AI understanding and at which stages verification should be performed. You can also have AI evaluate your conversation history and Git commit history to retrospectively identify areas for improvement in how you guide the AI.

## The Importance of Project Management

The key to effectively utilizing AI lies in appropriate project management and workflow structure.

This repository adopts a well-organized folder structure, Git version control, and a "documentation → implementation → verification" workflow. The `R/` directory contains reusable functions, `scripts/analyze_data/` contains AI-generated implementations, `scripts/analyze_data_true/` contains reference implementations, and `docs/` contains conversation history and presentation materials.

The crucial point is recognizing that **when AI doesn't function as expected, it's not an AI problem but rather an issue with how instructions are given or the workflow is structured**. Just like with junior analysts, if you provide clear instructions and a verification process, AI can deliver the expected results.

## Future Developments

While we've demonstrated the use case of "implementing causal inference," data science encompasses diverse analytical tasks.

We plan to introduce the following use cases in the future:

- **Reinforcement learning**: Having AI implement policy optimization and value function estimation algorithms
- **Inverse reinforcement learning**: Requesting AI to implement reward inference from observed behavior
- **Data preprocessing and feature engineering**: Conducting data cleaning and transformation work using the oral exam approach

In all cases, the oral exam approach and documenting the interaction process will play important roles.

## Summary

This repository demonstrates that on-demand data analysis tasks can be effectively automated using AI assistants under proper management. By combining oral exams for knowledge verification, staged implementation, and rigorous validation processes through visualization and statistical checks, we achieved reliable implementation even for complex Staggered DiD analysis.

The key takeaways for data scientists:
- Conduct oral exams to verify AI's understanding of statistical methods before implementation
- Document the interaction process for reproducibility and learning
- Structure your project with clear workflows and version control
- Treat AI assistance like supervising a junior analyst—provide clear guidance and verify outputs

Detailed conversation history and slide materials are available in the `docs/` directory, and all code is published in the [GitHub repository](https://github.com/kohei-kawaguchi/TestAI).

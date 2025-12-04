# Problems Directory

This directory hosts training problem sets that mirror the main MDP workflow while forcing juniors to rebuild each step for new economic environments (e.g., the static oligopoly pricing model).

## General Instructions for Juniors

1. **AI-only implementation**: All coding must be executed by directing approved AI coding tools. Do not hand-write code yourself; instead, provide clear instructions to the AI, review its output, and iterate until the implementation satisfies the specification.
2. **Respect the structure**: Follow the repository conventions: function and class definitions live in `src/` Python modules, notebooks call into those modules, configs live under `scripts/config_*`, and outputs land in `output/`. Do not invent ad-hoc folders.
3. **Mirror the workflow**: Execute the steps in the same order as the baseline MDP pipeline—configure → solve → simulate → estimate—using the `uv` environment where applicable. Document how data flows between stages (configs saved as data, solver outputs consumed by simulator, etc.).
4. **Document decisions**: Every Quarto scaffold lists required components; expand them with pseudo code, validation plans, and figures/tables that match the MDP standard. Use the scaffolds as contracts—fill them in rather than inventing new structures.
5. **Supervisor review**: Turn in both the rendered reports and the AI prompt history so supervisors can audit how you leveraged AI. Highlight any deviations from the prescribed workflow.
6. **Issue-driven development**: Track work through Git issues (or tasks) and version all changes via Git. Create branches per issue, keep commits focused on the problem set task, and document how each change closes or advances the tracked issue.

Problem sets such as `problem_set_1.md` describe the economic goals and deliverables for each track. Complete all referenced scaffolds before moving to the next problem set.

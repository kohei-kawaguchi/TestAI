# Guidelines for Claude

## Git Operations

**Do not git add/commit without explicit user request.**

- Only run `git add` and `git commit` when the user explicitly asks to "commit changes" or similar
- Do not automatically commit after completing tasks
- Do not batch multiple changes into commits without permission
- Wait for explicit confirmation before pushing to remote repositories

## Function Call Convention

**Always use named arguments when calling functions.**

- All function calls must specify argument names explicitly (e.g., `func(x=value, y=value)`)
- Never rely on positional argument order (e.g., avoid `func(value1, value2)`)
- This applies to both implementation code and test code
- This prevents parameter order issues and makes code more maintainable
- Example:
  - ✅ Good: `ComputeNextState(s=state, a=action, gamma=gamma)`
  - ❌ Bad: `ComputeNextState(state, action, gamma)`

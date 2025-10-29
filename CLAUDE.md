# Claude Code Project Guidelines

**Read the CRITICAL CHECKLIST section of CLAUDE.md every time the user gives an instruction.**

---

## 🚨 CRITICAL CHECKLIST

```
□ REPORT THE ISSUE BEFORE FIXING, EDITING, OR EXECUTING ANYTHING
□ NO GIT without explicit "commit" request
□ NO TIMESTAMPS (❌ file_20250101.py → ✅ file.py)
□ NO VERSIONS (❌ file_v2.py, _old.py → ✅ file.py)
□ NO BACKWARD COMPATIBILITY
□ TEMPORARY → scripts/temporary/
□ NAMED ARGUMENTS (❌ f(x) → ✅ f(value=x))
□ EDIT don't CREATE
□ PLAN → WAIT → "proceed"
□ TEST → show results → "done"
□ USE run.sh wrapper
□ DELETE wrong code
```

---

## 🔄 WORKFLOW

### Planning
1. Create plan → Show → **WAIT** for "proceed"
2. "continue" ≠ "execute"

### Testing
After code change: `./run.sh test` → show results → "done"

Before expensive: `pipeline-test` (100) → verify S3 → `pipeline-release` (1000)

### Commands
`./run.sh test` NOT `pytest`
`./run.sh solve` NOT `python scripts/...`

---

## 🔧 REFACTORING

1. Grep ALL old names
2. Update ALL (source + tests) in same commit
3. Never change VALUES, only names
4. Grep verify none remain
5. Test

**Wrong code:** Delete (use git for history)

---

## 📊 REPRODUCIBILITY

### DRY
Single implementation, parameterize differences

### Config = Data
Save with models, load automatically, no hardcoding

### Stable Paths
`output/eq/` NOT `output/eq_20250927/`
`file.py` NOT `file_v2.py`

---

## 🧪 TESTING

Test properties (monotonicity, bounds), not exact values

Test fails → Fix code, not test

---

## 📝 SPECIFICATION

1. Read spec first
2. Implement exactly (don't improvise)
3. Validate vs theory
4. Test on dummy data
5. Verify before expensive compute

**APIs:** Grep verify names exist, read implementation

---

## 🎯 DECISION POINTS

"plan this" → Create, show, **WAIT** for "proceed"
"continue" → Ask: explaining or execute?
"Did you test?" → Run tests now
Code "WRONG" → Delete

---

## 🚫 NEVER

1. Git without "commit" request
2. Execute without "proceed"
3. "done" without test results
4. Timestamps/versions in filenames
5. Positional arguments
6. Backward compatibility
7. Feature flags for wrong code

---

## 🎓 PRINCIPLES

1. **Complete verification** - Grep ALL, test everything
2. **Config = data** - Save/load, no hardcoding
3. **Reproducibility** - Stable paths, DRY
4. **Trust user** - Follow exactly
5. **Test first** - Before claiming done, before expensive compute

---

## 📖 MORE

[README.md](README.md) - Project structure, S3, HPC4
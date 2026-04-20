# AGENTS.md

## General Guidelines

### Planning

Use `$plan-review` for every non-trivial implementation plan. This is a required step before presenting a final plan.

For non-trivial plans, follow this sequence:
1. Read the rubric in `$plan-review` before drafting.
2. Draft the implementation plan with rubric criteria in mind.
3. Write a narrative reflection that engages with the plan across all five criteria.
4. Revise weak areas identified by the reflection.
5. Repeat narrative reflection as needed.
6. Present the final plan with explicit tradeoffs where relevant.

### ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in .agents/PLANS.md) from design to implementation.

### Code Style & Typing
- **Formatting**: Strict `ruff` enforcement. All PRs must pass `ruff check --fix .`
- **Typing**: Explicit types preferred
  - **OK**: `cast(...)`, `assert ...` for type narrowing
  - **SOMETIMES OK**: Untyped args for simple cases (e.g., prompt handlers)
  - **NOT OK**: `# type: ignore` without strong justification

### Naming Conventions
- **Methods**: snake_case
- **Classes**: PascalCase (e.g., `LocalREPL`, `PortkeyClient`)
- **Variables**: snake_case
- **Constants**: UPPER_CASE (e.g., `_SAFE_BUILTINS`, `RLM_SYSTEM_PROMPT`)

Do NOT use `_` prefix for private methods unless explicitly requested.

### Error Handling Philosophy
- **Fail fast, fail loud** - No defensive programming or silent fallbacks
- **Minimize branching** - Prefer single code paths; every `if`/`try` needs justification
- **Example**: Missing API key → immediate `ValueError`, not graceful fallback

### Dependencies
- Avoid new core dependencies
- Use optional extras for non-essential features (e.g., `modal` extra)
- Exception: tiny deps that simplify widely-used code

### Testing
- `uv run pytest` with discovery under `tests/`
- Write simple, deterministic unit tests
- Update tests when changing functionality
- For isolated environments, mock external services

### Documentation
- Keep concise and actionable
- Update README when behavior changes
- Avoid content duplication

### Scope
- Small, focused diffs
- One change per PR
- Backward compatibility is only desirable if it can be done without introducing excessive maintenance burden
- Delete dead code (don't guard it)

### Checklist

Before a PR/task completion:

```bash
# Run style + lint checks:
uv run ruff check --fix .
uv run ruff format .
uv run ty check
uv run pre-commit run --all-files # only if there is a pre-commit config

# Run tests:
uv run pytest # try uv run -m pytest if `uv run pytest` fails
```

If you find failures beyond the task at hand, report them, but do not immediately fix them.

Ensure docs and tests are updated if necessary, and dead code is deleted. Strive for minimal, surgical diffs.

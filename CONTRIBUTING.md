# Contributing to HVAC RL

Thank you for helping improve this project! This document describes our branching strategy, commit conventions, and review process.

---

## Branching Strategy

We follow a lightweight GitFlow model:

```
main          ← production-ready; only merged from dev via PR
dev           ← integration branch; all feature branches merge here
feature/*     ← new features (e.g. feature/double-q-learning)
fix/*         ← bug fixes   (e.g. fix/reward-clipping)
hotfix/*      ← urgent production fixes merged directly to main + dev
experiment/*  ← exploratory work that may never merge
```

### Rules

| Branch | Protected | Who can push | Merges via |
|--------|-----------|-------------|------------|
| `main` | ✅ Yes | Nobody directly | PR from `dev` |
| `dev`  | ✅ Yes | Nobody directly | PR from `feature/*` |
| `feature/*` | ❌ No | Author | PR to `dev` |
| `hotfix/*`  | ❌ No | Author | PR to `main` + `dev` |

---

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]
[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

**Examples:**
```
feat(agent): add Double Q-Learning variant
fix(env): correct reward clipping bounds
docs(readme): add K8s deployment section
ci(github-actions): enable scheduled retraining
```

---

## Pull Request Process

1. Branch from `dev`: `git checkout -b feature/my-feature dev`
2. Keep PRs small and focused — one feature or fix per PR
3. Make sure `pytest tests/` passes locally before opening a PR
4. Fill in the PR template (description, testing done, screenshots if UI)
5. Request at least one reviewer
6. Squash-merge into `dev`; the commit message becomes the conventional commit

---

## Local Dev Setup

```bash
git clone https://github.com/<org>/hvac-rl.git
cd hvac-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/          # all tests must pass
pre-commit install     # optional but recommended
```

---

## Code Style

- Formatter: **Black** (line length 88)
- Import order: **isort**
- Linter: **flake8** (max line 120)

The CI pipeline enforces all three. Run locally:
```bash
black .
isort .
flake8 .
```

---

## Adding a New RL Agent

1. Subclass `_BaseAgent` in `env.py`
2. Implement `update(...)` following the Q-Learning or SARSA pattern
3. Register the new type in `build_agent()` in `env.py`
4. Add tests in `tests/test_all.py`
5. Document the algorithm in `README.md` under *Models*

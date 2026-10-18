# iaa

One-line description of what this project does.

## Architecture

> Keep this section current — it is the living architecture doc the `spec-first`
> skill reads before non-trivial changes and updates after. Describe the real
> system, not aspirations.

- **Entry points:**
- **Data flow:**
- **Interface contracts / invariants:**
- **Failure modes:**
- **External services & how credentials are handled:**

## Conventions

- Python 3.12, `uv`-managed venv. Deps pinned in a lockfile before first deploy.
- Secrets via `.env` (chmod 600) or Keychain — never literals in source.
- New endpoints authenticate by default; outbound calls set explicit `timeout=`.
- Quality gates: `pre-commit` (ruff + gitleaks) locally, `quality.yml` in CI
  (ruff + gitleaks secrets + pytest). Run `pre-commit install` on a fresh clone.
- Deploying to Fly.io? Follow the `flyio-deploy` skill — pre-flight checklist, `fly.toml` template, and lessons learned.

## Run / test

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q
```

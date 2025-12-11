# Repository Guidelines

## What to Touch
- Don’t hand-edit generated artifacts (e.g., `doc.html`); focus on `src/`, `tests/`, and config files.
- Avoid destructive git operations (`reset --hard`, `checkout --`) unless explicitly requested.
- Prefer `rg` for searching the codebase.

## Setup
- Create/update the dev env: `uv sync --all-groups --extra test` (add `--extra docs` if needed). Use `uv run <cmd>` or activate `.venv`.

## Must-Run Commands Before Hand-off
- `uv run task format_check`
- `uv run task lint_check`
- `uv run task types_check`
- `uv run task tests` (or the focused subset below when offline)
- For docs changes, also `uv run task docs_build`

## Testing
- Full suite: `uv run task tests`.
- Offline/limited network: `uv run pytest tests -k "not OpenAIStore and not ingest"` (skips API and scraping).
- Always run the relevant tests yourself and report results; don’t defer to the user.

## Style & API
- Python, 4-space indent; snake_case for functions/modules, PascalCase for classes; type hints on public APIs; prefer `dataclass` for structured payloads.
- Let Ruff handle formatting.

## Commits & PRs
- After every set of changes, emit a draft commit message. If you are asked for revisions, when you're done, emit
  an updated draft commit message.
- Never run modifying or destructive git commands unless explicitly asked. I.e., never stage, unstage, commit, etc. But inspecting the current git status is fine.
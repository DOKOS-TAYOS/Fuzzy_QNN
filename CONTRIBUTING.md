# Contributing

Thanks for contributing.

## Workflow

1. Bootstrap the template before major development.
2. Work in small changes and prefer test-first development.
3. Keep public APIs minimal and document new behavior.
4. Update `CHANGELOG.md` whenever something meaningful changes.
5. Before handing off work, run:
   - `ruff check . --fix`
   - `ruff format .`
   - `pytest`
   - `pyright`

## Development Environment

- Use a local `.venv`.
- Install editable dependencies with `python -m pip install -e .[dev]`.
- Use the wrappers in `bin/` or the CLI commands directly.

## Pull Request Checklist

- tests added or updated first when behavior changed
- docs updated where needed
- AI docs updated if workflow, architecture, or conventions changed
- license inventory refreshed if dependencies changed

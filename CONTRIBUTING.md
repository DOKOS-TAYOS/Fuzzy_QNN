# Contributing

Thanks for contributing.

## Workflow

1. Work in small changes and prefer test-first development.
2. Keep public APIs minimal and document new behavior.
3. Update `CHANGELOG.md` whenever something meaningful changes.
4. Before handing off work, run `ruff check . --fix`, `ruff format .`, `pytest`, and `pyright`.

## Development Environment

- Use a local `.venv`.
- Install editable dependencies with `python -m pip install -e .[dev]`.
- Use the wrappers in `bin/` or the CLI commands directly.

## Pull Request Checklist

- tests added or updated first when behavior changed
- docs updated where needed
- license inventory refreshed if dependencies changed

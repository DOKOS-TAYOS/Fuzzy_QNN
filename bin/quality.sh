#!/usr/bin/env sh
set -eu
PYTHON_EXE="./.venv/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
  PYTHON_EXE="python"
fi
"$PYTHON_EXE" -m ruff check . --fix
"$PYTHON_EXE" -m ruff format .
"$PYTHON_EXE" -m pytest

#!/usr/bin/env sh
set -eu
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -rf .pytest_cache .ruff_cache

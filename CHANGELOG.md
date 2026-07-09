# Changelog

## [Unreleased]

### Fixed

- Repaired GitHub Actions workflows so CI and security jobs no longer call missing helper scripts.
- Fixed Pyright CI configuration and several static typing hotspots so type checking no longer references the removed `examples` path and is less brittle around scikit-learn and PennyLane boundaries.
- Removed leftover setup-template references from repository documentation and helper scripts.

## [0.2.0] - 2026-07-08

### Added

- Full FQNN implementation under `src/fuzzy_qnn` with typed config, fuzzy, quantum, model, data, training, evaluation, artifact, and baseline modules.
- CLI scripts for GPU validation, training, benchmarking, checkpoint evaluation, noise evaluation, experiment batches, and sample inspection.
- Ready-to-run YAML configs for Iris, Breast Cancer, Wine, Synthetic, and CPU debug workflows.
- CPU-first configs for Iris, Breast Cancer, Wine, and Synthetic so the project can be run comfortably on Windows without PennyLane GPU support.
- Run artifact generation including histories, plots, checkpoints, rules, fuzzy parameters, and device metadata.
- Pytest coverage for fuzzy layers, quantum circuit, runtime fallback, GPU-optional execution, training smoke, and CLI smoke.
- `requirements.txt` and `requirements-gpu.txt` for clean environment setup.
- CLI support for `--dry-run`, `--max-epochs`, and `--experiment-name` overrides, plus extra config validation tests for missing YAML keys.

### Changed

- Focused the repository workflow on CLI-driven FQNN experiments.
- Rewrote the README around installation, GPU validation, training, benchmarking, evaluation, and outputs.
- Improved terminal UX with clearer startup summaries, more readable final test output, better YAML validation errors, and clearer `lightning.gpu` fallback messaging.
- Changed the implicit runtime preference to CPU when a config omits `runtime.prefer_gpu`, avoiding accidental GPU attempts on unsupported systems.

### Removed

- Legacy setup, clean, and tooling package modules from the previous repository layout.
- Old tests, examples, and extra Markdown documentation that no longer described the project.

# Changelog

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

- Replaced the previous template-first repository flow with a CLI-first FQNN workflow focused on executable experiments.
- Rewrote the README around installation, GPU validation, training, benchmarking, evaluation, and outputs.
- Improved terminal UX with clearer startup summaries, more readable final test output, better YAML validation errors, and clearer `lightning.gpu` fallback messaging.
- Changed the implicit runtime preference to CPU when a config omits `runtime.prefer_gpu`, avoiding accidental GPU attempts on unsupported systems.

### Removed

- Template bootstrap, clean, and tooling package modules.
- Template-specific tests, examples, and extra Markdown documentation that no longer described the actual project.

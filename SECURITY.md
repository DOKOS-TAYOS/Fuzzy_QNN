# Security Policy

## Supported Versions

Security support applies to the latest maintained branch of this repository.

## Reporting a Vulnerability

Use a private reporting channel defined by the maintainers. Do not open a
public issue for a suspected secret leak or security defect until the
maintainers confirm the handling plan.

## Template Security Baseline

This repository keeps these tracked security defaults:

- Dependabot version updates for Python dependency metadata and GitHub Actions.
- A Security workflow with dependency review and `pip-audit`.
- Read-only GitHub Actions token permissions unless a job explicitly needs more.
- Checkout steps that do not persist credentials after the repository is fetched.

## GitHub Settings to Keep Enabled

Some GitHub security features are repository settings, not tracked files. Keep
these enabled in GitHub when available:

- Dependabot alerts and Dependabot security updates.
- Dependency graph.
- Code scanning alerts with CodeQL default setup.
- Secret scanning and push protection.

If a project stores credentials or handles sensitive data, add project-specific threat modeling and environment hardening before production use.

Do not add a CodeQL advanced workflow while GitHub CodeQL default setup is enabled. GitHub rejects those uploads because the two setup modes are mutually exclusive for the same repository.

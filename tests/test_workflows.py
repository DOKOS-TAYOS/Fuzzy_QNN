from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PYTHON_SCRIPT_PATTERN = re.compile(r"""(?<!\S)(?:python|python3)\s+["']?([^"'\s]+\.py)\b""")


def _iter_local_python_scripts(run_command: str) -> list[Path]:
    script_paths: list[Path] = []

    for match in PYTHON_SCRIPT_PATTERN.finditer(run_command):
        candidate = Path(match.group(1))
        if candidate.is_absolute():
            continue
        script_paths.append(REPO_ROOT / candidate)

    return script_paths


def test_workflows_only_reference_existing_repo_python_scripts() -> None:
    missing_references: list[str] = []

    for workflow_path in WORKFLOW_DIR.glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        jobs = workflow.get("jobs", {})

        for job_name, job_config in jobs.items():
            steps = job_config.get("steps", [])

            for step_index, step in enumerate(steps, start=1):
                run_command = step.get("run")
                if not isinstance(run_command, str):
                    continue

                step_name = step.get("name", f"step-{step_index}")
                for script_path in _iter_local_python_scripts(run_command):
                    if script_path.exists():
                        continue
                    missing_references.append(
                        f"{workflow_path.name}:{job_name}:{step_name} -> "
                        f"{script_path.relative_to(REPO_ROOT).as_posix()}"
                    )

    assert not missing_references, "Workflow steps reference missing repo scripts:\n" + "\n".join(
        missing_references
    )

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cicd_contract_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_cicd.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CI/CD validation passed" in result.stdout


def test_docker_context_excludes_local_credentials_and_state() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert ".env" in patterns
    assert ".venv" in patterns
    assert ".git" in patterns
    assert "var" in patterns
    assert "gha-creds-*.json" in patterns

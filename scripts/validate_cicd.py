"""Validate security and reproducibility invariants of the CI/CD configuration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
ACTION_USE = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)
IMMUTABLE_ACTION = re.compile(r"^[^/\s]+/[^@\s]+@[0-9a-f]{40}$")
LOCKED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==[^=\s]+$")
DEV_ONLY_PACKAGES = {
    "coverage",
    "iniconfig",
    "pluggy",
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "ruff",
}


def _requirements(path: Path) -> set[str]:
    packages: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = LOCKED_REQUIREMENT.fullmatch(stripped)
        if match is None:
            raise ValueError(f"{path.name} contains an unlocked requirement: {stripped}")
        packages.add(match.group(1).casefold())
    return packages


def validate() -> None:
    errors: list[str] = []
    workflow_paths = sorted(WORKFLOWS.glob("*.yml"))
    if not workflow_paths:
        errors.append("no GitHub Actions workflows found")

    for path in workflow_paths:
        text = path.read_text(encoding="utf-8")
        if "pull_request_target:" in text:
            errors.append(f"{path.name}: pull_request_target is forbidden")
        if "permissions:" not in text:
            errors.append(f"{path.name}: explicit permissions are required")
        for action in ACTION_USE.findall(text):
            if action.startswith("./"):
                continue
            if not IMMUTABLE_ACTION.fullmatch(action):
                errors.append(f"{path.name}: action is not pinned to a commit SHA: {action}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for forbidden in ("TELEGRAM_BOT_TOKEN=", "OPENAI_API_KEY=", "COPY .env"):
        if forbidden in dockerfile:
            errors.append(f"Dockerfile contains forbidden secret material: {forbidden}")
    if "USER 10001:10001" not in dockerfile:
        errors.append("Dockerfile must run as the dedicated non-root user")
    if dockerfile.count("python:3.11-slim@sha256:") != 2:
        errors.append("both Docker stages must pin the base image by digest")
    if 'CMD ["python", "-m", "app.main"]' not in dockerfile:
        errors.append("Dockerfile must use the application composition root")

    all_packages = _requirements(ROOT / "requirements.lock")
    runtime_packages = _requirements(ROOT / "requirements.runtime.lock")
    expected_runtime = all_packages - DEV_ONLY_PACKAGES
    if runtime_packages != expected_runtime:
        missing = sorted(expected_runtime - runtime_packages)
        extra = sorted(runtime_packages - expected_runtime)
        errors.append(f"runtime lock drift: missing={missing}, extra={extra}")

    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    release_contracts = (
        "packages: write",
        "attestations: write",
        "provenance: mode=max",
        "sbom: true",
    )
    for contract in release_contracts:
        if contract not in release:
            errors.append(f"release.yml is missing: {contract}")

    promote = (WORKFLOWS / "promote.yml").read_text(encoding="utf-8")
    if "environment:" not in promote or "docker buildx imagetools create" not in promote:
        errors.append("promote.yml must use a protected environment and digest promotion")

    if errors:
        raise SystemExit("CI/CD validation failed:\n- " + "\n- ".join(errors))
    print(
        f"CI/CD validation passed: {len(workflow_paths)} workflows, immutable actions, locked image"
    )


if __name__ == "__main__":
    validate()

"""Fail CI when versioned or untracked source files contain credential signatures."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def main() -> None:
    candidates = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    findings: list[str] = []
    scanned = 0
    for name in candidates:
        if not name:
            continue
        path = Path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        if any(pattern.search(text) for pattern in PATTERNS):
            findings.append(name)
    if findings:
        raise SystemExit("possible secrets in tracked files: " + ", ".join(findings))
    print(f"secret scan passed: {scanned} source files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export docs/presentation.md to PDF (requires pandoc)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "presentation.md"
PDF = ROOT / "docs" / "presentation.pdf"


def main() -> int:
    if not MD.exists():
        print(f"Missing {MD}")
        return 1
    if not shutil.which("pandoc"):
        print("pandoc not found; install with: brew install pandoc")
        return 1

    cmd = [
        "pandoc",
        str(MD),
        "-o",
        str(PDF),
        "--resource-path=docs",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"Wrote {PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared tsc --noEmit checker for dataset generation and validation."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TS_CHECKER = ROOT / "eval" / "ts-checker"
CHECK_FILE = TS_CHECKER / "check.tsx"


def check_typescript(code: str) -> tuple[bool, str]:
    """Write code to check.tsx and run tsc --noEmit."""
    CHECK_FILE.write_text(code, encoding="utf-8")
    try:
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=TS_CHECKER,
        )
        if result.returncode != 0:
            errors = [
                line for line in result.stdout.splitlines()
                if "error TS" in line
            ]
            return False, "; ".join(errors[:3]) if errors else "tsc error"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "tsc timeout"
    finally:
        CHECK_FILE.unlink(missing_ok=True)

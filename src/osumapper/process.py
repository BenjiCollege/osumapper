from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from osumapper.errors import DependencyError


def run_checked(
    args: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in args]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Required executable was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DependencyError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "no diagnostic output").strip()
        raise DependencyError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}\n{details}"
        ) from exc

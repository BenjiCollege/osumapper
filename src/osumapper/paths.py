from __future__ import annotations

import os
from pathlib import Path

from osumapper.errors import DependencyError


def project_root() -> Path:
    override = os.environ.get("OSUMAPPER_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "v7.0").is_dir():
            return root
        raise DependencyError(f"OSUMAPPER_PROJECT_ROOT does not contain v7.0: {root}")

    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "v7.0").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise DependencyError(
        "Could not locate the osumapper project root. Set OSUMAPPER_PROJECT_ROOT."
    )


def legacy_v7_root() -> Path:
    return project_root() / "v7.0"


def modern_models_root() -> Path:
    return project_root() / "models"

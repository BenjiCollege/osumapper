from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from osumapper.errors import DependencyError


def _running_under_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _windows_path_from_wsl(path: Path) -> str:
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DependencyError(f"Could not translate WSL path for osu!lazer: {path}") from exc
    translated = result.stdout.strip()
    if not translated:
        raise DependencyError(f"WSL returned an empty Windows path for {path}.")
    return translated


def find_lazer_executable() -> Path | None:
    override = os.environ.get("OSU_LAZER_PATH")
    if override and Path(override).expanduser().is_file():
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = Path(local) / "osulazer" / "current" / "osu!.exe"
            if candidate.is_file():
                return candidate
    elif sys.platform == "darwin":
        candidate = Path("/Applications/osu!.app/Contents/MacOS/osu!")
        if candidate.is_file():
            return candidate
    elif _running_under_wsl():
        users = Path("/mnt/c/Users")
        if users.is_dir():
            candidates = users.glob("*/AppData/Local/osulazer/current/osu!.exe")
            for candidate in sorted(candidates, key=lambda item: str(item).casefold()):
                if candidate.is_file():
                    return candidate
    for executable in ("osu!", "osu-lazer", "osu"):
        found = shutil.which(executable)
        if found:
            return Path(found)
    return None


def open_in_lazer(package: Path) -> None:
    executable = find_lazer_executable()
    if executable is None:
        raise DependencyError(
            "osu!lazer was not found. Set OSU_LAZER_PATH or import the .osz manually."
        )
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    package_argument = (
        _windows_path_from_wsl(package)
        if _running_under_wsl() and executable.suffix.casefold() == ".exe"
        else str(package.resolve())
    )
    try:
        subprocess.Popen(
            [str(executable), package_argument],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise DependencyError(f"Could not open {package} with {executable}: {exc}") from exc

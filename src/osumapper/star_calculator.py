from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from osumapper.beatmap import BeatmapDocument
from osumapper.difficulty import calculate_standard_stars
from osumapper.errors import DependencyError, InputError
from osumapper.lazer import find_lazer_executable
from osumapper.paths import project_root


class StandardStarCalculator(Protocol):
    name: str
    version: str

    def calculate(self, source: Path | BeatmapDocument) -> float: ...

    def close(self) -> None: ...


class RosuStarCalculator:
    """Published rosu-pp fallback retained for reproducibility and non-lazer systems."""

    name = "rosu-pp-py-4.0.2-legacy"
    version = "4.0.2"

    def calculate(self, source: Path | BeatmapDocument) -> float:
        return calculate_standard_stars(source)

    def close(self) -> None:
        return None


def _running_under_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _windows_path(path: Path) -> str:
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path.expanduser().resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DependencyError(f"Could not translate WSL path for .NET: {path}") from exc
    translated = result.stdout.strip()
    if not translated:
        raise DependencyError(f"WSL returned an empty Windows path for {path}.")
    return translated


def _dotnet_executable() -> Path:
    found = shutil.which("dotnet") or shutil.which("dotnet.exe")
    if not found:
        raise DependencyError(
            "The installed osu!lazer star calculator requires .NET 8. Install the .NET 8 SDK "
            "on Windows, or use --star-calculator rosu for the legacy approximation."
        )
    return Path(found).expanduser().resolve()


def _uses_windows_dotnet(dotnet: Path) -> bool:
    return _running_under_wsl() and dotnet.suffix.casefold() == ".exe"


def _dotnet_path(path: Path, dotnet: Path) -> str:
    return _windows_path(path) if _uses_windows_dotnet(dotnet) else str(path.resolve())


def _helper_fingerprint(lazer_directory: Path) -> str:
    helper_root = _helper_source_root()
    digest = hashlib.sha256()
    for path in (
        helper_root / "OsuMapper.LazerDifficulty.csproj",
        helper_root / "Program.cs",
        lazer_directory / "osu.Game.dll",
        lazer_directory / "osu.Game.Rulesets.Osu.dll",
    ):
        try:
            stat = path.stat()
        except OSError as exc:
            raise DependencyError(f"Required star-calculator file is missing: {path}") from exc
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        if path.suffix.casefold() in {".cs", ".csproj"}:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _helper_source_root() -> Path:
    checkout = project_root() / "tools" / "lazer-difficulty"
    bundled = Path(__file__).resolve().parent / "_lazer_difficulty"
    for candidate in (checkout, bundled):
        if (candidate / "OsuMapper.LazerDifficulty.csproj").is_file():
            return candidate
    raise DependencyError("The bundled osu!lazer difficulty bridge source is missing.")


def _build_helper(dotnet: Path, lazer_directory: Path) -> Path:
    root = project_root()
    helper_root = _helper_source_root()
    project = helper_root / "OsuMapper.LazerDifficulty.csproj"
    config = helper_root / "NuGet.Config"
    cache = root / ".bootstrap" / "lazer-difficulty" / _helper_fingerprint(lazer_directory)
    output = cache / "OsuMapper.LazerDifficulty.dll"
    if output.is_file():
        return output

    cache.mkdir(parents=True, exist_ok=True)
    dotnet_home = root / ".bootstrap" / "dotnet-home"
    appdata = root / ".bootstrap" / "dotnet-appdata"
    packages = root / ".bootstrap" / "dotnet-packages"
    for path in (dotnet_home, appdata, packages):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "DOTNET_CLI_HOME": _dotnet_path(dotnet_home, dotnet),
            "APPDATA": _dotnet_path(appdata, dotnet),
            "NUGET_PACKAGES": _dotnet_path(packages, dotnet),
            "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
            "DOTNET_NOLOGO": "1",
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        }
    )
    commands = (
        [
            str(dotnet),
            "restore",
            _dotnet_path(project, dotnet),
            "--configfile",
            _dotnet_path(config, dotnet),
            "--verbosity",
            "quiet",
        ],
        [
            str(dotnet),
            "build",
            _dotnet_path(project, dotnet),
            "--configuration",
            "Release",
            "--output",
            _dotnet_path(cache, dotnet),
            "--no-restore",
            "--nologo",
            "--verbosity",
            "quiet",
        ],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            diagnostic = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)
            raise DependencyError(
                f"Could not build the local osu!lazer difficulty bridge: {diagnostic.strip()}"
            ) from exc
        if result.stderr.strip() and not output.exists():
            raise DependencyError(result.stderr.strip())
    if not output.is_file():
        raise DependencyError(f"The osu!lazer difficulty bridge did not produce {output}.")
    return output


class LazerStarCalculator:
    """Calculate stars with the exact osu!standard ruleset installed with osu!lazer."""

    name = "osu!lazer-installed"

    def __init__(self) -> None:
        executable = find_lazer_executable()
        if executable is None:
            raise DependencyError(
                "osu!lazer was not found. Set OSU_LAZER_PATH, or use "
                "--star-calculator rosu for the legacy approximation."
            )
        self._lazer_directory = executable.parent.resolve()
        self._dotnet = _dotnet_executable()
        self._helper = _build_helper(self._dotnet, self._lazer_directory)
        self._process: subprocess.Popen[str] | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="osumapper-lazer-stars-")
        self._measurement = 0
        self.version = "installed"

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        command = [
            str(self._dotnet),
            _dotnet_path(self._helper, self._dotnet),
            _dotnet_path(self._lazer_directory, self._dotnet),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=creationflags,
            )
        except OSError as exc:
            raise DependencyError(f"Could not start the osu!lazer star calculator: {exc}") from exc
        return self._process

    def calculate(self, source: Path | BeatmapDocument) -> float:
        if isinstance(source, BeatmapDocument):
            self._measurement += 1
            path = Path(self._temporary.name) / f"measurement-{self._measurement}.osu"
            path.write_text(source.text, encoding="utf-8", newline="\r\n")
        else:
            path = source.expanduser().resolve()
        process = self._start()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(f"{_dotnet_path(path, self._dotnet)}\n")
            process.stdin.flush()
            response = process.stdout.readline()
        except (OSError, BrokenPipeError) as exc:
            raise DependencyError(
                f"The osu!lazer star calculator stopped unexpectedly: {exc}"
            ) from exc
        if not response:
            diagnostic = process.stderr.read().strip() if process.stderr is not None else ""
            raise DependencyError(
                "The osu!lazer star calculator returned no result. "
                f"{diagnostic or 'Check the installed lazer files and .NET 8 runtime.'}"
            )
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise DependencyError(
                f"The osu!lazer star calculator returned invalid output: {response.strip()}"
            ) from exc
        if payload.get("error"):
            raise InputError(f"osu!lazer could not calculate this beatmap: {payload['error']}")
        self.version = str(payload.get("game_version", "installed"))
        try:
            return float(payload["stars"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DependencyError(f"osu!lazer returned no numeric star rating: {payload}") from exc

    def close(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                with suppress(BrokenPipeError, OSError):
                    self._process.stdin.close()
            if self._process.poll() is None:
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    self._process.wait(timeout=3)
            self._process = None
        self._temporary.cleanup()


def resolve_star_calculator(
    preference: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> StandardStarCalculator:
    normalized = preference.strip().casefold()
    if normalized == "rosu":
        return RosuStarCalculator()
    if normalized not in {"auto", "lazer"}:
        raise InputError("Star calculator must be auto, lazer, or rosu.")
    try:
        return LazerStarCalculator()
    except DependencyError as exc:
        if normalized == "lazer":
            raise
        if progress is not None:
            progress(
                "Current osu!lazer calculator unavailable; falling back to legacy rosu-pp "
                f"measurement ({exc})."
            )
        return RosuStarCalculator()

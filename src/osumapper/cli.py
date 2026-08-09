from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from osumapper import (
    __maintainer__,
    __original_author__,
    __original_project__,
    __version__,
)
from osumapper.config import GameMode, GenerationConfig
from osumapper.engine import generate_document
from osumapper.errors import OsumapperError
from osumapper.lazer import find_lazer_executable, open_in_lazer
from osumapper.models import (
    legacy_model_paths,
    migrate_all,
    migrate_model,
    verify_all,
    verify_model,
)
from osumapper.package import generated_map_name, write_osz
from osumapper.paths import project_root
from osumapper.presets import get_preset, preset_names
from osumapper.stable import scan_stable_maps, write_maplist
from osumapper.workspace import SUPPORTED_AUDIO, prepare_source


class ConsoleProgress:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.step = 0

    def __call__(self, message: str) -> None:
        self.step += 1
        if not self.quiet:
            print(f"[{self.step:02d}] {message}", flush=True)


def _mode(value: str) -> GameMode | None:
    try:
        return GameMode.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osumapper",
        description="Generate importable osu!lazer beatmap packages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a beatmap and package it as .osz")
    generate.add_argument("source", type=Path, help="input .osz, .osu, or audio file")
    generate.add_argument("--audio", type=Path, help="audio for an explicitly selected .osu")
    generate.add_argument("--output", "-o", type=Path, help="destination .osz")
    generate.add_argument("--preset", choices=preset_names(), default="default")
    generate.add_argument(
        "--mode", type=_mode, default=None, help="auto, standard, taiko, catch, or mania"
    )
    generate.add_argument(
        "--difficulty", help="difficulty-name substring when an .osz has several maps"
    )
    generate.add_argument("--seed", type=int, default=2026)
    generate.add_argument(
        "--flow-engine", choices=("auto", "legacy", "deterministic"), default="auto"
    )
    generate.add_argument("--bpm", type=float, help="tempo for audio-only input")
    generate.add_argument("--offset", type=int, help="timing offset in milliseconds")
    generate.add_argument(
        "--keys", type=int, default=4, help="mania key count for audio-only input"
    )
    generate.add_argument("--open", action="store_true", dest="open_in_lazer")
    generate.add_argument("--quiet", action="store_true")

    subparsers.add_parser("doctor", help="inspect the local runtime and installation")
    subparsers.add_parser("presets", help="list generation presets")
    subparsers.add_parser("ui", help="open the local drag-and-drop interface")
    subparsers.add_parser("credits", help="show original-project attribution")

    models = subparsers.add_parser("models", help="migrate or verify legacy rhythm models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    migrate = models_sub.add_parser(
        "migrate", help="convert HDF5 models to .keras with parity checks"
    )
    migrate.add_argument("--preset", help="migrate only one legacy model directory")
    migrate.add_argument("--seed", type=int, default=2026)
    migrate.add_argument("--tolerance", type=float, default=1e-5)
    verify = models_sub.add_parser("verify", help="verify model and parity-manifest hashes")
    verify.add_argument("--preset", help="verify only one legacy model directory")
    verify.add_argument("--tolerance", type=float, default=1e-5)

    stable = subparsers.add_parser(
        "stable-scan", help="build a maplist from an osu!stable Songs folder"
    )
    stable.add_argument("root", type=Path, help="osu!stable root or Songs directory")
    stable.add_argument("--output", "-o", type=Path, default=Path("maplist.txt"))
    stable.add_argument("--mode", type=_mode, default=None)
    return parser


def _default_output(source: Path, preset: str) -> Path:
    output_dir = Path.cwd() / "output"
    return output_dir / f"{source.stem}-osumapper-{preset}.osz"


def _generate(args: argparse.Namespace) -> int:
    progress = ConsoleProgress(args.quiet)
    source_path = args.source.expanduser().resolve()
    output = (args.output or _default_output(source_path, args.preset)).expanduser().resolve()
    if output == source_path:
        raise OsumapperError("Output must not overwrite the input package.")
    config = GenerationConfig(
        preset=args.preset,
        mode=args.mode,
        seed=args.seed,
        difficulty=args.difficulty,
        output=output,
        open_in_lazer=args.open_in_lazer,
        flow_engine=args.flow_engine,
    )
    progress("Opening input in an isolated workspace")
    with prepare_source(
        source_path,
        audio=args.audio,
        difficulty=args.difficulty,
        mode=args.mode,
        bpm=args.bpm,
        offset_ms=args.offset,
        key_count=args.keys,
    ) as workspace:
        source_mode = args.mode or workspace.document.mode
        preset = get_preset(args.preset, source_mode)
        generated = generate_document(
            workspace.document,
            workspace.audio,
            workspace.root,
            preset,
            config,
            progress,
        )
        generated_path = workspace.root / generated_map_name(generated, preset.name)
        generated_path.write_text(generated.text, encoding="utf-8", newline="\r\n")
        excluded = {
            workspace.root / "mapthis.json",
            workspace.root / "mapthis.npz",
            workspace.root / "rhythm_data.npz",
        }
        if source_path.suffix.casefold() in SUPPORTED_AUDIO:
            excluded.add(workspace.document.path)
        progress("Building deterministic .osz package")
        write_osz(workspace.root, output, exclude=excluded)
    if args.open_in_lazer:
        progress("Opening package in osu!lazer")
        open_in_lazer(output)
    if not args.quiet:
        print(f"Created: {output}")
    return 0


def _doctor() -> int:
    root = project_root()
    checks: dict[str, Any] = {
        "osumapper": __version__,
        "project_root": str(root),
        "python": platform.python_version(),
        "python_supported": sys.version_info[:2] == (3, 12),
        "lazer": str(find_lazer_executable() or "not found"),
        "legacy_models": len(legacy_model_paths()),
        "dependencies": {
            name: bool(importlib.util.find_spec(name))
            for name in (
                "numpy",
                "librosa",
                "sklearn",
                "matplotlib",
                "tensorflow",
                "tkinterdnd2",
            )
        },
    }
    print(json.dumps(checks, indent=2))
    return 0 if checks["python_supported"] else 1


def _find_model(name: str) -> Path:
    matches = [
        path for path in legacy_model_paths() if path.parent.name.casefold() == name.casefold()
    ]
    if not matches:
        available = ", ".join(path.parent.name for path in legacy_model_paths())
        raise OsumapperError(f"Unknown model directory {name!r}. Available: {available}")
    return matches[0]


def _models(args: argparse.Namespace) -> int:
    if args.models_command == "migrate":
        results = (
            [migrate_model(_find_model(args.preset), seed=args.seed, tolerance=args.tolerance)]
            if args.preset
            else migrate_all(seed=args.seed, tolerance=args.tolerance)
        )
    else:
        results = (
            [verify_model(_find_model(args.preset), tolerance=args.tolerance)]
            if args.preset
            else verify_all(tolerance=args.tolerance)
        )
    for result in results:
        print(
            f"{Path(result.source).parent.name}: parity max error={result.max_absolute_error:.8g}"
        )
    return 0


def _stable_scan(args: argparse.Namespace) -> int:
    paths = scan_stable_maps(args.root, args.mode)
    output = write_maplist(paths, args.output)
    print(f"Wrote {len(paths)} beatmap paths to {output}")
    return 0


def _credits() -> int:
    print(f"osumapper {__version__}")
    print(f"Original creator: {__original_author__}")
    print(f"Original project: {__original_project__}")
    print(f"2026 modernization: {__maintainer__} and contributors")
    print("This package is a maintained derivative of the original project.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return _generate(args)
        if args.command == "doctor":
            return _doctor()
        if args.command == "presets":
            print("\n".join(preset_names()))
            return 0
        if args.command == "credits":
            return _credits()
        if args.command == "models":
            return _models(args)
        if args.command == "stable-scan":
            return _stable_scan(args)
        if args.command == "ui":
            from osumapper.ui import launch

            launch()
            return 0
    except (OsumapperError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unhandled command: {args.command}")
    return 2

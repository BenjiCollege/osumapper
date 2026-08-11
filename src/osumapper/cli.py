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
from osumapper.config import GameMode
from osumapper.errors import OsumapperError
from osumapper.lazer import find_lazer_executable
from osumapper.models import (
    legacy_model_paths,
    migrate_all,
    migrate_model,
    verify_all,
    verify_model,
)
from osumapper.paths import project_root
from osumapper.pipeline import GenerationRequest, generate_package
from osumapper.presets import preset_names
from osumapper.stable import scan_stable_maps, write_maplist
from osumapper.training.analysis import analyze_map
from osumapper.training.calibration import calibrate_threshold
from osumapper.training.config import (
    AudioFeatureConfig,
    GridConfig,
    QualityConfig,
    RhythmTrainingConfig,
)
from osumapper.training.dataset import (
    dataset_statistics,
    import_osz_dataset,
    rate_map,
    rate_maps,
    scan_dataset,
)
from osumapper.training.evaluation import evaluate_rhythm
from osumapper.training.features import extract_dataset_features
from osumapper.training.loader import make_tf_dataset
from osumapper.training.placement import analyze_placement
from osumapper.training.placement_learning import (
    PlacementTrainingConfig,
    evaluate_placement,
    train_placement,
)
from osumapper.training.review import generate_review_packages
from osumapper.training.splits import load_split, split_dataset
from osumapper.training.trainer import train_rhythm


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
        "--flow-engine",
        choices=("auto", "legacy", "deterministic", "placement"),
        default="auto",
    )
    generate.add_argument("--rhythm-engine", choices=("legacy", "modern"), default="legacy")
    generate.add_argument(
        "--modern-model", type=Path, help="modern rhythm model directory or .keras file"
    )
    generate.add_argument(
        "--placement-model", type=Path, help="Placement-v1 model directory or .keras file"
    )
    generate.add_argument("--rhythm-threshold", type=float, help="modern hit-probability threshold")
    generate.add_argument(
        "--target-density", type=float, help="target objects per second for modern rhythm"
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

    dataset = subparsers.add_parser("dataset", help="build and manage local training data")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_scan = dataset_sub.add_parser(
        "scan", help="scan an osu!stable Songs directory for standard maps"
    )
    dataset_scan.add_argument("songs_root", type=Path)
    dataset_scan.add_argument("--data-root", type=Path)
    dataset_scan.add_argument("--min-objects", type=int, default=25)
    dataset_scan.add_argument("--min-duration-ms", type=int, default=5_000)
    dataset_scan.add_argument("--max-duration-ms", type=int, default=30 * 60 * 1_000)
    dataset_scan.add_argument("--min-bpm", type=float, default=20.0)
    dataset_scan.add_argument("--max-bpm", type=float, default=1_000.0)
    dataset_scan.add_argument("--allow-converted", action="store_true")
    dataset_scan.add_argument("--append", action="store_true")

    dataset_stats = dataset_sub.add_parser("stats", help="summarize the local dataset")
    dataset_stats.add_argument("--data-root", type=Path)

    dataset_rate = dataset_sub.add_parser("rate", help="manually rate a beatmap")
    dataset_rate.add_argument("map", type=Path)
    dataset_rate.add_argument("rating", choices=("good", "bad", "ignore"))
    dataset_rate.add_argument("--data-root", type=Path)

    dataset_rate_folder = dataset_sub.add_parser(
        "rate-folder", help="rate every .osu file under a deliberately selected folder"
    )
    dataset_rate_folder.add_argument("folder", type=Path)
    dataset_rate_folder.add_argument("rating", choices=("good", "bad", "ignore"))
    dataset_rate_folder.add_argument("--data-root", type=Path)

    dataset_import = dataset_sub.add_parser(
        "import-osz", help="safely stage .osz packages and append them to the dataset"
    )
    dataset_import.add_argument("source", type=Path)
    dataset_import.add_argument("--data-root", type=Path)
    dataset_import.add_argument(
        "--rating",
        choices=("good", "bad", "ignore"),
        help="apply only when you have reviewed/trust every imported map",
    )

    dataset_split = dataset_sub.add_parser(
        "split", help="create deterministic song-level train/validation/test splits"
    )
    dataset_split.add_argument("--data-root", type=Path)
    dataset_split.add_argument("--seed", type=int, default=2026)
    dataset_split.add_argument("--train-ratio", type=float, default=0.8)
    dataset_split.add_argument("--validation-ratio", type=float, default=0.1)
    dataset_split.add_argument("--test-ratio", type=float, default=0.1)
    dataset_split.add_argument("--include-unrated", action="store_true")

    dataset_features = dataset_sub.add_parser(
        "features", help="extract and cache deterministic per-song audio features"
    )
    dataset_features.add_argument("--data-root", type=Path)
    dataset_features.add_argument("--sample-rate", type=int, default=22_050)
    dataset_features.add_argument("--hop-length", type=int, default=512)
    dataset_features.add_argument("--n-fft", type=int, default=2_048)
    dataset_features.add_argument("--n-mels", type=int, default=128)
    dataset_features.add_argument("--force", action="store_true")

    dataset_windows = dataset_sub.add_parser(
        "windows", help="precompute deterministic float16 training-window shards"
    )
    dataset_windows.add_argument("--data-root", type=Path)
    dataset_windows.add_argument("--sequence-length", type=int, default=512)
    dataset_windows.add_argument("--audio-context-radius", type=int, default=0)
    dataset_windows.add_argument("--rebuild", action="store_true")

    train = subparsers.add_parser("train", help="train or evaluate modern local models")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_rhythm_parser = train_sub.add_parser("rhythm", help="train the modern rhythm model")
    train_rhythm_parser.add_argument("--data-root", type=Path)
    train_rhythm_parser.add_argument("--epochs", type=int, default=50)
    train_rhythm_parser.add_argument("--batch-size", type=int, default=16)
    train_rhythm_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_rhythm_parser.add_argument("--seed", type=int, default=2026)
    train_rhythm_parser.add_argument("--resume", action="store_true")
    train_rhythm_parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    train_rhythm_parser.add_argument(
        "--precision",
        choices=("auto", "float32", "mixed-float16"),
        default="auto",
        help="numeric policy (auto uses mixed float16 on an NVIDIA GPU)",
    )
    train_rhythm_parser.add_argument(
        "--xla", choices=("auto", "on", "off"), default="auto", help="XLA compilation mode"
    )
    train_rhythm_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="stop after this many epochs without validation PR-AUC improvement; 0 disables",
    )
    train_rhythm_parser.add_argument("--lr-patience", type=int, default=3)
    train_rhythm_parser.add_argument("--lr-factor", type=float, default=0.5)
    train_rhythm_parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    train_rhythm_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_rhythm_parser.add_argument(
        "--window-cache",
        choices=("auto", "off", "rebuild"),
        default="auto",
        help="reuse, disable, or rebuild deterministic float16 training shards",
    )
    train_rhythm_parser.add_argument(
        "--no-balance-songs",
        action="store_true",
        help="disable inverse-window song weighting",
    )
    train_rhythm_parser.add_argument("--output", type=Path)
    train_rhythm_parser.add_argument("--sequence-length", type=int, default=256)
    train_rhythm_parser.add_argument("--threshold", type=float, default=0.5)
    train_rhythm_parser.add_argument(
        "--architecture",
        choices=("transformer-v1", "conformer-v2", "conformer-v3"),
        default="conformer-v3",
    )
    train_rhythm_parser.add_argument(
        "--audio-context-radius",
        type=int,
        help="feature frames on each side (default: 4 for v2, 0 for v1/v3)",
    )

    train_placement_parser = train_sub.add_parser(
        "placement", help="train learned standard-mode flow and object-type placement"
    )
    train_placement_parser.add_argument("--data-root", type=Path)
    train_placement_parser.add_argument("--output", type=Path)
    train_placement_parser.add_argument("--epochs", type=int, default=50)
    train_placement_parser.add_argument("--batch-size", type=int, default=32)
    train_placement_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_placement_parser.add_argument("--sequence-length", type=int, default=256)
    train_placement_parser.add_argument("--seed", type=int, default=2026)
    train_placement_parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    train_placement_parser.add_argument(
        "--precision", choices=("auto", "float32", "mixed-float16"), default="auto"
    )
    train_placement_parser.add_argument("--xla", choices=("auto", "on", "off"), default="auto")
    train_placement_parser.add_argument("--early-stopping-patience", type=int, default=8)
    train_placement_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_placement_parser.add_argument("--no-balance-songs", action="store_true")
    train_placement_parser.add_argument("--resume", action="store_true")

    train_evaluate = train_sub.add_parser(
        "evaluate", help="evaluate a model against held-out test songs only"
    )
    train_evaluate.add_argument("model_kind", choices=("rhythm", "placement"))
    train_evaluate.add_argument("--data-root", type=Path)
    train_evaluate.add_argument("--model", type=Path)
    train_evaluate.add_argument("--threshold", type=float)
    train_evaluate.add_argument("--output", type=Path)

    train_calibrate = train_sub.add_parser(
        "calibrate", help="select a prediction threshold using validation songs only"
    )
    train_calibrate.add_argument("model_kind", choices=("rhythm",))
    train_calibrate.add_argument("--data-root", type=Path)
    train_calibrate.add_argument("--model", type=Path)
    train_calibrate.add_argument("--output", type=Path)

    train_review = train_sub.add_parser(
        "review", help="generate deterministic packages from held-out test songs"
    )
    train_review.add_argument("model_kind", choices=("rhythm",))
    train_review.add_argument("--data-root", type=Path)
    train_review.add_argument("--model", type=Path)
    train_review.add_argument("--output", type=Path)
    train_review.add_argument("--count", type=int, default=5)
    train_review.add_argument("--seed", type=int, default=2026)
    train_review.add_argument("--threshold", type=float)
    train_review.add_argument("--open", action="store_true", dest="open_in_lazer")

    analyze = subparsers.add_parser(
        "analyze", help="compare one human standard map with the modern rhythm model"
    )
    analyze.add_argument("map", type=Path)
    analyze.add_argument("--model", type=Path)
    analyze.add_argument("--data-root", type=Path)
    analyze.add_argument("--threshold", type=float)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--format", choices=("json", "csv", "both"), default="json")

    placement = subparsers.add_parser(
        "placement", help="inspect standard-map flow, bounds, spacing, and playability heuristics"
    )
    placement_sub = placement.add_subparsers(dest="placement_command", required=True)
    placement_analyze = placement_sub.add_parser("analyze", help="analyze one .osu map")
    placement_analyze.add_argument("map", type=Path)
    placement_analyze.add_argument("--output", type=Path)

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
    generate_package(
        GenerationRequest(
            source=source_path,
            output=output,
            preset=args.preset,
            mode=args.mode,
            difficulty=args.difficulty,
            seed=args.seed,
            flow_engine=args.flow_engine,
            rhythm_engine=args.rhythm_engine,
            modern_model=args.modern_model,
            placement_model=args.placement_model,
            rhythm_threshold=args.rhythm_threshold,
            target_density=args.target_density,
            audio=args.audio,
            bpm=args.bpm,
            offset_ms=args.offset,
            key_count=args.keys,
            open_in_lazer=args.open_in_lazer,
        ),
        progress=progress,
    )
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


def _dataset(args: argparse.Namespace) -> int:
    if args.dataset_command == "scan":
        quality = QualityConfig(
            min_objects=args.min_objects,
            min_duration_ms=args.min_duration_ms,
            max_duration_ms=args.max_duration_ms,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            reject_converted=not args.allow_converted,
        )
        result = scan_dataset(
            args.songs_root,
            dataset_root=args.data_root,
            quality=quality,
            append=args.append,
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.dataset_command == "stats":
        print(json.dumps(dataset_statistics(dataset_root=args.data_root), indent=2))
        return 0
    if args.dataset_command == "rate":
        result = rate_map(args.map, args.rating, dataset_root=args.data_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.dataset_command == "rate-folder":
        folder = args.folder.expanduser().resolve()
        if not folder.is_dir():
            raise OsumapperError(f"Folder does not exist: {folder}")
        maps = sorted(folder.rglob("*.osu"), key=lambda path: str(path).casefold())
        result = rate_maps(maps, args.rating, dataset_root=args.data_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.dataset_command == "import-osz":
        result = import_osz_dataset(
            args.source,
            dataset_root=args.data_root,
            rating=args.rating,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.dataset_command == "split":
        result = split_dataset(
            dataset_root=args.data_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            include_unrated=args.include_unrated,
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.dataset_command == "features":
        config = AudioFeatureConfig(
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            fmax=args.sample_rate / 2,
        )
        result = extract_dataset_features(
            dataset_root=args.data_root,
            config=config,
            force=args.force,
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.dataset_command == "windows":
        if args.sequence_length <= 0 or not 0 <= args.audio_context_radius <= 32:
            raise OsumapperError("Sequence length must be positive and context must be 0-32.")
        summaries: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            rows = load_split(split, dataset_root=args.data_root)
            if not rows:
                continue
            _, summary = make_tf_dataset(
                rows,
                dataset_root=args.data_root,
                grid_config=GridConfig(sequence_length=args.sequence_length),
                batch_size=1,
                seed=2026,
                shuffle=False,
                audio_context_radius=args.audio_context_radius,
                cache_split=split,
                cache_mode="rebuild" if args.rebuild else "auto",
                balance_songs=split == "train",
            )
            summaries[split] = {
                "windows": summary.windows,
                "songs": summary.songs,
                "cache": summary.cache_path,
            }
        print(json.dumps(summaries, indent=2))
        return 0
    raise OsumapperError(f"Unhandled dataset command: {args.dataset_command}")


def _train(args: argparse.Namespace) -> int:
    if args.train_command == "rhythm":
        if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
            raise OsumapperError("Epochs, batch size, and learning rate must be positive.")
        if not 0 < args.threshold < 1:
            raise OsumapperError("Prediction threshold must be between 0 and 1.")
        context_radius = (
            args.audio_context_radius
            if args.audio_context_radius is not None
            else (4 if args.architecture == "conformer-v2" else 0)
        )
        if context_radius < 0 or context_radius > 32:
            raise OsumapperError("Audio context radius must be between 0 and 32.")
        config = RhythmTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            sequence_length=args.sequence_length,
            prediction_threshold=args.threshold,
            architecture=args.architecture,
            audio_context_radius=context_radius,
            precision=args.precision,
            xla=args.xla,
            early_stopping_patience=args.early_stopping_patience,
            learning_rate_patience=args.lr_patience,
            learning_rate_factor=args.lr_factor,
            minimum_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            window_cache=args.window_cache,
            balance_songs=not args.no_balance_songs,
        )
        result = train_rhythm(
            dataset_root=args.data_root,
            output=args.output,
            config=config,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "model": str(result.model),
                    "best_checkpoint": str(result.best_checkpoint),
                    "epochs_completed": result.epochs_completed,
                    "train_windows": result.train.windows,
                    "validation_windows": result.validation.windows,
                },
                indent=2,
            )
        )
        return 0
    if args.train_command == "placement":
        if (
            args.epochs <= 0
            or args.batch_size <= 0
            or args.learning_rate <= 0
            or args.sequence_length <= 0
        ):
            raise OsumapperError(
                "Placement epochs, batch size, rate, and sequence must be positive."
            )
        report = train_placement(
            dataset_root=args.data_root,
            output=args.output,
            config=PlacementTrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,
                seed=args.seed,
                device=args.device,
                precision=args.precision,
                xla=args.xla,
                early_stopping_patience=args.early_stopping_patience,
                weight_decay=args.weight_decay,
                balance_songs=not args.no_balance_songs,
            ),
            resume=args.resume,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.train_command == "evaluate" and args.model_kind == "rhythm":
        report = evaluate_rhythm(
            dataset_root=args.data_root,
            model_root=args.model,
            threshold=args.threshold,
            output=args.output,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.train_command == "evaluate" and args.model_kind == "placement":
        report = evaluate_placement(
            dataset_root=args.data_root,
            model_root=args.model,
            output=args.output,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.train_command == "calibrate" and args.model_kind == "rhythm":
        report = calibrate_threshold(
            dataset_root=args.data_root,
            model_root=args.model,
            output=args.output,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.train_command == "review" and args.model_kind == "rhythm":
        report = generate_review_packages(
            dataset_root=args.data_root,
            model_root=args.model,
            output=args.output,
            count=args.count,
            seed=args.seed,
            threshold=args.threshold,
            open_in_lazer=args.open_in_lazer,
        )
        print(json.dumps(report, indent=2))
        return 0
    raise OsumapperError(f"Unhandled training command: {args.train_command}")


def _analyze(args: argparse.Namespace) -> int:
    report = analyze_map(
        args.map,
        model_root=args.model,
        dataset_root=args.data_root,
        threshold=args.threshold,
        output=args.output,
        timeline_format=args.format,
    )
    print(json.dumps(report, indent=2))
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
        if args.command == "dataset":
            return _dataset(args)
        if args.command == "train":
            return _train(args)
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "placement" and args.placement_command == "analyze":
            print(json.dumps(analyze_placement(args.map, output=args.output), indent=2))
            return 0
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

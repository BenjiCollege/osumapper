from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from osumapper.difficulty import STAR_DIFFICULTY_FEATURES
from osumapper.errors import InputError
from osumapper.package import safe_filename
from osumapper.pipeline import GenerationRequest, generate_package
from osumapper.training.config import DatasetPaths, prediction_threshold
from osumapper.training.splits import load_split
from osumapper.training.storage import read_json, write_json
from osumapper.training.trainer import default_model_root


def select_review_maps(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    count: int = 5,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise InputError("Review-map count must be positive.")
    paths = DatasetPaths.at(dataset_root)
    root = (model_root or default_model_root()).expanduser().resolve()
    if root.suffix.casefold() == ".keras":
        root = root.parent
    model_manifest = read_json(root / "dataset_manifest.json", default=None)
    current_manifest = read_json(paths.splits / "manifest.json", default=None)
    if model_manifest is None or current_manifest is None:
        raise InputError("Held-out review requires model and split manifests.")
    if model_manifest.get("split_manifest") != current_manifest:
        raise InputError("The current split differs from the model split; review would be unsafe.")
    rows = load_split("test", dataset_root=paths.root)
    by_song: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_song.setdefault(str(row["song_id"]), []).append(row)
    song_ids = sorted(by_song)
    random.Random(seed).shuffle(song_ids)
    selected: list[dict[str, Any]] = []
    for song_id in song_ids[: min(count, len(song_ids))]:
        # Prefer the median-density map so the review is not biased toward only easy
        # or only extreme difficulties when a set contains many maps.
        candidates = sorted(
            by_song[song_id],
            key=lambda row: (float(row["objects_per_second"]), str(row["map_path"])),
        )
        selected.append(candidates[len(candidates) // 2])
    return selected


def generate_review_packages(
    *,
    dataset_root: Path | None = None,
    model_root: Path | None = None,
    output: Path | None = None,
    count: int = 5,
    seed: int = 2026,
    threshold: float | None = None,
    open_in_lazer: bool = False,
    progress: Any = print,
) -> dict[str, Any]:
    root = (model_root or default_model_root()).expanduser().resolve()
    if root.suffix.casefold() == ".keras":
        root = root.parent
    config = read_json(root / "config.json", default=None)
    if config is None:
        raise InputError(f"Modern model configuration is missing under {root}.")
    chosen_threshold = prediction_threshold(config, threshold)
    star_conditioned = tuple(config.get("difficulty_features", ())) == STAR_DIFFICULTY_FEATURES
    destination = (output or (Path.cwd() / "output" / "held-out-review")).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = select_review_maps(dataset_root=dataset_root, model_root=root, count=count, seed=seed)
    generated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        package = destination / safe_filename(
            f"{index:02d}-{row['artist']} - {row['title']} [{row['version']}]", ".osz"
        )
        progress(f"Review map {index}/{len(rows)}: {row['map_path']}")
        generate_package(
            GenerationRequest(
                source=Path(str(row["map_path"])),
                output=package,
                preset="default",
                seed=seed,
                flow_engine="deterministic",
                rhythm_engine="modern",
                modern_model=root,
                rhythm_threshold=chosen_threshold,
                difficulty_tier=str(row["difficulty_tier"]) if star_conditioned else None,
                target_stars=float(row["star_rating"]) if star_conditioned else None,
                open_in_lazer=open_in_lazer,
            ),
            progress=lambda message, number=index: progress(f"[{number}] {message}"),
        )
        generated.append(
            {
                "map_id": row["map_id"],
                "song_id": row["song_id"],
                "source_map": row["map_path"],
                "source_objects_per_second": row["objects_per_second"],
                "source_star_rating": row.get("star_rating"),
                "difficulty_tier": row.get("difficulty_tier"),
                "package": str(package),
            }
        )
    report = {
        "version": 1,
        "source_split": "test",
        "selection": "seeded_song_sample_median_density_map",
        "seed": seed,
        "threshold": chosen_threshold,
        "model": str(root / "model.keras"),
        "packages": generated,
    }
    manifest = destination / "review_manifest.json"
    write_json(manifest, report)
    report["output"] = str(destination)
    report["manifest"] = str(manifest)
    return report

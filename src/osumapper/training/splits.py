from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osumapper.errors import InputError
from osumapper.training.config import DatasetPaths
from osumapper.training.storage import read_parquet, write_json, write_parquet

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitSummary:
    seed: int
    songs: dict[str, int]
    maps: dict[str, int]
    root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "songs": self.songs,
            "maps": self.maps,
            "root": str(self.root),
        }


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_counts(group_count: int, validation_ratio: float, test_ratio: float) -> tuple[int, int]:
    if group_count <= 1:
        return 0, 0
    validation = round(group_count * validation_ratio)
    test = round(group_count * test_ratio)
    if group_count >= 3 and validation_ratio > 0:
        validation = max(1, validation)
    if group_count >= 3 and test_ratio > 0:
        test = max(1, test)
    while validation + test >= group_count:
        if test >= validation and test > 0:
            test -= 1
        elif validation > 0:
            validation -= 1
    return validation, test


def split_dataset(
    *,
    dataset_root: Path | None = None,
    seed: int = 2026,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    include_unrated: bool = False,
) -> SplitSummary:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value < 0 for value in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise InputError("Split ratios must be non-negative and sum to 1.0.")
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    rows = read_parquet(paths.dataset)
    selected = [
        row for row in rows if row["eligible"] and (include_unrated or row["rating"] == "good")
    ]
    if not selected:
        qualifier = "eligible maps" if include_unrated else "maps rated GOOD"
        raise InputError(
            f"No {qualifier} are available for splitting. Rate maps or use --include-unrated."
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["song_id"])].append(row)
    song_ids = sorted(grouped)
    random.Random(seed).shuffle(song_ids)
    validation_count, test_count = _split_counts(len(song_ids), validation_ratio, test_ratio)
    test_songs = set(song_ids[:test_count])
    validation_songs = set(song_ids[test_count : test_count + validation_count])
    assignments: dict[str, str] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for song_id in song_ids:
        split = (
            "test"
            if song_id in test_songs
            else "validation"
            if song_id in validation_songs
            else "train"
        )
        assignments[song_id] = split
        split_rows[split].extend(grouped[song_id])

    for name in SPLIT_NAMES:
        split_rows[name].sort(key=lambda row: (str(row["song_id"]), str(row["map_path"])))
        write_parquet(split_rows[name], paths.splits / f"{name}.parquet")
    songs = {
        name: sum(assignment == name for assignment in assignments.values()) for name in SPLIT_NAMES
    }
    maps = {name: len(split_rows[name]) for name in SPLIT_NAMES}
    manifest = {
        "version": 1,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "include_unrated": include_unrated,
        "dataset_sha256": _dataset_hash(paths.dataset),
        "songs": songs,
        "maps": maps,
        "assignments": dict(sorted(assignments.items())),
    }
    write_json(paths.splits / "manifest.json", manifest)
    return SplitSummary(seed=seed, songs=songs, maps=maps, root=paths.splits)


def load_split(split: str, *, dataset_root: Path | None = None) -> list[dict[str, Any]]:
    normalized = split.casefold()
    if normalized == "val":
        normalized = "validation"
    if normalized not in SPLIT_NAMES:
        raise InputError(f"Unknown split {split!r}; expected {', '.join(SPLIT_NAMES)}")
    paths = DatasetPaths.at(dataset_root)
    manifest_path = paths.splits / "manifest.json"
    if not manifest_path.is_file():
        raise InputError("Dataset has not been split. Run `osumapper dataset split`.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Invalid split manifest: {manifest_path}") from exc
    if manifest.get("dataset_sha256") != _dataset_hash(paths.dataset):
        raise InputError("Dataset changed after splitting; run `osumapper dataset split` again.")
    return read_parquet(paths.splits / f"{normalized}.parquet")

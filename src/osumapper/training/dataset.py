from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osumapper.errors import InputError
from osumapper.training import DATASET_SCHEMA_VERSION
from osumapper.training.beatmaps import (
    BeatmapDataError,
    ParsedTrainingBeatmap,
    file_sha256,
    parse_standard_beatmap,
)
from osumapper.training.config import DatasetPaths, QualityConfig
from osumapper.training.storage import read_json, read_parquet, write_json, write_parquet
from osumapper.workspace import safe_extract_osz

Progress = Callable[[str], None]
RATINGS = {"good", "bad", "ignore"}


@dataclass(frozen=True, slots=True)
class ScanSummary:
    discovered: int
    indexed: int
    candidates: int
    good: int
    rejected: int
    skipped: int
    dataset: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "indexed": self.indexed,
            "candidates": self.candidates,
            "good": self.good,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "dataset": str(self.dataset),
        }


def _ratings_document(paths: DatasetPaths) -> dict[str, Any]:
    value = read_json(paths.ratings, default=None)
    if value is None:
        return {"version": 1, "ratings": {}}
    if not isinstance(value, dict) or not isinstance(value.get("ratings"), dict):
        raise InputError(f"Malformed ratings file: {paths.ratings}")
    return value


def _rating_for(document: dict[str, Any], map_sha: str, map_path: Path) -> str | None:
    ratings = document["ratings"]
    record = ratings.get(map_sha) or ratings.get(str(map_path).casefold())
    if isinstance(record, str):
        return record if record in RATINGS else None
    if isinstance(record, dict) and record.get("rating") in RATINGS:
        return str(record["rating"])
    return None


def _quality_reasons(parsed: ParsedTrainingBeatmap, config: QualityConfig) -> list[str]:
    stats = parsed.statistics
    reasons: list[str] = []
    if stats["object_count"] < config.min_objects:
        reasons.append(f"object_count_below_{config.min_objects}")
    if stats["map_duration_ms"] < config.min_duration_ms:
        reasons.append(f"duration_below_{config.min_duration_ms}ms")
    if stats["map_duration_ms"] > config.max_duration_ms:
        reasons.append(f"duration_above_{config.max_duration_ms}ms")
    if stats["bpm_min"] < config.min_bpm or stats["bpm_max"] > config.max_bpm:
        reasons.append("bpm_out_of_range")
    if stats["objects_per_second"] > 20:
        reasons.append("implausible_object_density")
    first_timing_ms = min(point.time_ms for point in parsed.timing_points if point.uninherited)
    if parsed.hit_objects and first_timing_ms > stats["map_end_ms"]:
        reasons.append("timing_points_after_objects")
    if parsed.is_converted and config.reject_converted:
        reasons.append("converted_map")
    return reasons


def _quality_state(rating: str | None, reasons: list[str]) -> tuple[str, bool]:
    if rating == "bad":
        return "rated_bad", False
    if rating == "ignore":
        return "rated_ignore", False
    if reasons:
        return "rejected", False
    if rating == "good":
        return "good", True
    return "candidate_unrated", True


def _row(
    parsed: ParsedTrainingBeatmap,
    map_json_path: Path,
    rating: str | None,
    quality_status: str,
    eligible: bool,
    reasons: list[str],
) -> dict[str, Any]:
    metadata = parsed.metadata
    difficulty = parsed.difficulty
    stats = parsed.statistics
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "map_id": parsed.map_id,
        "map_sha256": parsed.map_sha256,
        "map_path": str(parsed.map_path),
        "map_json_path": str(map_json_path),
        "audio_path": str(parsed.audio_path),
        "song_id": parsed.song_id,
        "mapset_key": parsed.mapset_key,
        "beatmap_id": metadata["beatmap_id"],
        "beatmapset_id": metadata["beatmapset_id"],
        "artist": metadata["artist"],
        "artist_unicode": metadata["artist_unicode"],
        "title": metadata["title"],
        "title_unicode": metadata["title_unicode"],
        "creator": metadata["creator"],
        "version": metadata["version"],
        "source": metadata["source"],
        "tags": metadata["tags"],
        "mode": metadata["mode"],
        "is_converted": parsed.is_converted,
        "hp": difficulty["hp"],
        "cs": difficulty["cs"],
        "od": difficulty["od"],
        "ar": difficulty["ar"],
        "slider_multiplier": difficulty["slider_multiplier"],
        "slider_tick_rate": difficulty["slider_tick_rate"],
        "map_start_ms": stats["map_start_ms"],
        "map_end_ms": stats["map_end_ms"],
        "map_duration_ms": stats["map_duration_ms"],
        "object_count": stats["object_count"],
        "circle_count": stats["circle_count"],
        "slider_count": stats["slider_count"],
        "spinner_count": stats["spinner_count"],
        "objects_per_second": stats["objects_per_second"],
        "slider_percentage": stats["slider_percentage"],
        "average_spacing": stats["average_spacing"],
        "median_jump_distance": stats["median_jump_distance"],
        "max_jump_distance": stats["max_jump_distance"],
        "average_rhythm_interval": stats["average_rhythm_interval"],
        "median_rhythm_interval": stats["median_rhythm_interval"],
        "bpm_min": stats["bpm_min"],
        "bpm_max": stats["bpm_max"],
        "bpm_mean": stats["bpm_mean"],
        "timing_change_count": stats["timing_change_count"],
        "burst_count": stats["burst_count"],
        "stream_count": stats["stream_count"],
        "rating": rating or "unrated",
        "quality_status": quality_status,
        "eligible": eligible,
        "quality_reasons": reasons,
    }


def _write_skips(path: Path, skips: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in skips
        )
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _disambiguate_multi_audio_song_ids(rows: list[dict[str, Any]]) -> int:
    """Separate malformed mapsets that reuse one ID for unrelated audio files."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["song_id"]), []).append(row)
    changed = 0
    for base_song_id, group in grouped.items():
        audio_paths = {str(row["audio_path"]).casefold() for row in group}
        if len(audio_paths) <= 1:
            continue
        for row in group:
            audio_identity = str(row["audio_path"]).replace("\\", "/").casefold()
            suffix = hashlib.sha256(audio_identity.encode("utf-8")).hexdigest()[:12]
            song_id = f"{base_song_id}-audio-{suffix}"
            row["song_id"] = song_id
            detail_path = Path(str(row["map_json_path"]))
            detail = read_json(detail_path, default={})
            detail["song_id"] = song_id
            write_json(detail_path, detail)
            changed += 1
    return changed


def scan_dataset(
    songs_root: Path,
    *,
    dataset_root: Path | None = None,
    quality: QualityConfig | None = None,
    progress: Progress = print,
    append: bool = False,
) -> ScanSummary:
    root = songs_root.expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Songs directory does not exist: {root}")
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    config = quality or QualityConfig()
    ratings = _ratings_document(paths)
    discovered = sorted(root.rglob("*.osu"), key=lambda path: str(path).casefold())
    progress(f"Discovered {len(discovered)} .osu files")
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    seen_hashes: dict[str, Path] = {}

    for index, map_path in enumerate(discovered, start=1):
        try:
            map_sha = file_sha256(map_path)
        except OSError as exc:
            skips.append({"path": str(map_path), "reason": "unreadable_map", "detail": str(exc)})
            continue
        if map_sha in seen_hashes:
            skips.append(
                {
                    "path": str(map_path),
                    "reason": "duplicate_osu",
                    "duplicate_of": str(seen_hashes[map_sha]),
                }
            )
            continue
        seen_hashes[map_sha] = map_path
        try:
            parsed = parse_standard_beatmap(map_path, root)
            reasons = _quality_reasons(parsed, config)
            rating = _rating_for(ratings, parsed.map_sha256, parsed.map_path)
            quality_status, eligible = _quality_state(rating, reasons)
            detail = parsed.detail_dict()
            detail["rating"] = rating or "unrated"
            detail["quality_status"] = quality_status
            detail["eligible"] = eligible
            detail["quality_reasons"] = reasons
            map_json_path = paths.maps / f"{parsed.map_id}-{parsed.map_sha256[:12]}.json"
            write_json(map_json_path, detail)
            rows.append(_row(parsed, map_json_path, rating, quality_status, eligible, reasons))
            if not eligible:
                skips.append(
                    {
                        "path": str(map_path),
                        "reason": quality_status,
                        "detail": reasons,
                    }
                )
        except BeatmapDataError as exc:
            skips.append({"path": str(map_path), "reason": exc.code, "detail": str(exc)})
        except (OSError, ValueError, OverflowError) as exc:
            skips.append({"path": str(map_path), "reason": "malformed_map", "detail": str(exc)})
        if index % 250 == 0:
            progress(f"Scanned {index}/{len(discovered)} maps")

    previous_config = read_json(paths.config, default={}) if append else {}
    if append and paths.dataset.is_file():
        existing = read_parquet(paths.dataset)
        combined = {str(row["map_sha256"]): row for row in existing}
        combined.update({str(row["map_sha256"]): row for row in rows})
        rows = list(combined.values())
    disambiguated = _disambiguate_multi_audio_song_ids(rows)
    if disambiguated:
        progress(
            f"Separated {disambiguated} maps whose BeatmapSetID references multiple audio files"
        )
    rows.sort(key=lambda row: str(row["map_path"]).casefold())
    write_parquet(rows, paths.dataset)
    _write_skips(paths.skipped, skips)
    counts = Counter(str(row["quality_status"]) for row in rows)
    summary = ScanSummary(
        discovered=len(discovered),
        indexed=len(rows),
        candidates=counts["candidate_unrated"],
        good=counts["good"],
        rejected=sum(
            count
            for name, count in counts.items()
            if name in {"rejected", "rated_bad", "rated_ignore"}
        ),
        skipped=len(skips),
        dataset=paths.dataset,
    )
    write_json(
        paths.config,
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "songs_root": str(root),
            "songs_roots": sorted(
                {
                    str(root),
                    *[str(value) for value in previous_config.get("songs_roots", [])],
                    *(
                        [str(previous_config["songs_root"])]
                        if previous_config.get("songs_root")
                        else []
                    ),
                }
            ),
            "quality": config.as_dict(),
            "last_scan_utc": datetime.now(UTC).isoformat(),
            "summary": summary.as_dict(),
        },
    )
    progress(
        f"Indexed {summary.indexed} standard maps; "
        f"{summary.skipped} skipped/rejected records logged"
    )
    return summary


def rate_map(map_path: Path, rating: str, *, dataset_root: Path | None = None) -> dict[str, Any]:
    result = rate_maps([map_path], rating, dataset_root=dataset_root)
    return result["maps"][0]


def rate_maps(
    map_paths: list[Path], rating: str, *, dataset_root: Path | None = None
) -> dict[str, Any]:
    normalized = rating.casefold()
    if normalized not in RATINGS:
        raise InputError(f"Rating must be one of: {', '.join(sorted(RATINGS))}")
    resolved = [path.expanduser().resolve() for path in map_paths]
    missing = next((path for path in resolved if not path.is_file()), None)
    if missing is not None:
        raise InputError(f"Beatmap does not exist: {missing}")
    if not resolved:
        raise InputError("No .osu beatmaps were selected for rating.")
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    document = _ratings_document(paths)
    timestamp = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    identities: dict[str, Path] = {}
    for path in resolved:
        map_sha = file_sha256(path)
        identities[map_sha] = path
        record = {"rating": normalized, "path": str(path), "updated_utc": timestamp}
        document["ratings"][map_sha] = record
        document["ratings"][str(path).casefold()] = record
        results.append(
            {
                "path": str(path),
                "sha256": map_sha,
                "rating": normalized,
                "rows_updated": 0,
            }
        )
    write_json(paths.ratings, document)

    updated = 0
    if paths.dataset.is_file():
        rows = read_parquet(paths.dataset)
        result_lookup = {str(result["sha256"]): result for result in results}
        for row in rows:
            map_sha = str(row["map_sha256"])
            if map_sha not in identities:
                continue
            row["rating"] = normalized
            status, eligible = _quality_state(normalized, list(row["quality_reasons"] or []))
            row["quality_status"] = status
            row["eligible"] = eligible
            updated += 1
            result_lookup[map_sha]["rows_updated"] += 1
        write_parquet(rows, paths.dataset)
    return {
        "rating": normalized,
        "selected": len(resolved),
        "rows_updated": updated,
        "maps": results,
    }


def import_osz_dataset(
    source: Path,
    *,
    dataset_root: Path | None = None,
    rating: str | None = None,
    quality: QualityConfig | None = None,
    progress: Progress = print,
) -> dict[str, Any]:
    """Safely stage one package or a folder of packages and append them to the dataset."""
    selected = source.expanduser().resolve()
    if selected.is_file() and selected.suffix.casefold() == ".osz":
        packages = [selected]
    elif selected.is_dir():
        packages = sorted(selected.rglob("*.osz"), key=lambda path: str(path).casefold())
    else:
        raise InputError(f"Expected an .osz package or folder containing packages: {selected}")
    if not packages:
        raise InputError(f"No .osz packages were found under {selected}")
    paths = DatasetPaths.at(dataset_root)
    paths.create()
    imported_root = paths.root / "imported_osz"
    imported_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    reused = 0
    maps: list[Path] = []
    for index, package in enumerate(packages, start=1):
        package_hash = file_sha256(package)
        destination = imported_root / package_hash[:20]
        if destination.is_dir() and any(destination.rglob("*.osu")):
            reused += 1
        else:
            with tempfile.TemporaryDirectory(prefix=".osumapper-osz-", dir=imported_root) as name:
                temporary = Path(name)
                safe_extract_osz(package, temporary)
                os.replace(temporary, destination)
            extracted += 1
        maps.extend(destination.rglob("*.osu"))
        if index % 25 == 0:
            progress(f"Imported {index}/{len(packages)} packages")
    if rating is not None:
        rate_maps(sorted(set(maps)), rating, dataset_root=paths.root)
    summary = scan_dataset(
        imported_root,
        dataset_root=paths.root,
        quality=quality,
        progress=progress,
        append=True,
    )
    return {
        "packages": len(packages),
        "extracted": extracted,
        "reused": reused,
        "maps_in_packages": len(set(maps)),
        "rating": rating or "unrated",
        "dataset": summary.as_dict(),
        "import_root": str(imported_root),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(finite) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return finite[lower]
        return finite[lower] * (upper - position) + finite[upper] * (position - lower)

    return {
        "count": len(finite),
        "min": finite[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def dataset_statistics(*, dataset_root: Path | None = None) -> dict[str, Any]:
    paths = DatasetPaths.at(dataset_root)
    rows = read_parquet(paths.dataset)
    quality_counts = Counter(str(row["quality_status"]) for row in rows)
    rating_counts = Counter(str(row["rating"]) for row in rows)
    versions = Counter(str(row["version"]) for row in rows)
    return {
        "songs": len({row["song_id"] for row in rows}),
        "mapsets": len({row["mapset_key"] for row in rows}),
        "difficulties": len(rows),
        "total_hit_objects": sum(int(row["object_count"]) for row in rows),
        "quality_status": dict(sorted(quality_counts.items())),
        "ratings": dict(sorted(rating_counts.items())),
        "bpm": _distribution([float(row["bpm_mean"]) for row in rows]),
        "approach_rate": _distribution([float(row["ar"]) for row in rows]),
        "overall_difficulty": _distribution([float(row["od"]) for row in rows]),
        "circle_size": _distribution([float(row["cs"]) for row in rows]),
        "map_duration_seconds": _distribution(
            [float(row["map_duration_ms"]) / 1000.0 for row in rows]
        ),
        "objects_per_second": _distribution([float(row["objects_per_second"]) for row in rows]),
        "difficulty_names": dict(versions.most_common(50)),
    }

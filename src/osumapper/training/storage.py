from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from osumapper.errors import DependencyError, InputError


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DependencyError("Dataset storage requires pyarrow. Run `uv sync --locked`.") from exc
    return pa, pq


def dataset_schema() -> Any:
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("schema_version", pa.int16()),
            ("map_id", pa.string()),
            ("map_sha256", pa.string()),
            ("map_path", pa.string()),
            ("map_json_path", pa.string()),
            ("audio_path", pa.string()),
            ("song_id", pa.string()),
            ("mapset_key", pa.string()),
            ("beatmap_id", pa.int64()),
            ("beatmapset_id", pa.int64()),
            ("artist", pa.string()),
            ("artist_unicode", pa.string()),
            ("title", pa.string()),
            ("title_unicode", pa.string()),
            ("creator", pa.string()),
            ("version", pa.string()),
            ("source", pa.string()),
            ("tags", pa.string()),
            ("mode", pa.int8()),
            ("is_converted", pa.bool_()),
            ("star_rating", pa.float32()),
            ("difficulty_tier", pa.string()),
            ("hp", pa.float32()),
            ("cs", pa.float32()),
            ("od", pa.float32()),
            ("ar", pa.float32()),
            ("slider_multiplier", pa.float32()),
            ("slider_tick_rate", pa.float32()),
            ("map_start_ms", pa.float64()),
            ("map_end_ms", pa.float64()),
            ("map_duration_ms", pa.float64()),
            ("object_count", pa.int32()),
            ("circle_count", pa.int32()),
            ("slider_count", pa.int32()),
            ("spinner_count", pa.int32()),
            ("objects_per_second", pa.float32()),
            ("slider_percentage", pa.float32()),
            ("average_spacing", pa.float32()),
            ("median_jump_distance", pa.float32()),
            ("max_jump_distance", pa.float32()),
            ("average_rhythm_interval", pa.float32()),
            ("median_rhythm_interval", pa.float32()),
            ("bpm_min", pa.float32()),
            ("bpm_max", pa.float32()),
            ("bpm_mean", pa.float32()),
            ("timing_change_count", pa.int32()),
            ("burst_count", pa.int32()),
            ("stream_count", pa.int32()),
            ("rating", pa.string()),
            ("quality_status", pa.string()),
            ("eligible", pa.bool_()),
            ("quality_reasons", pa.list_(pa.string())),
        ]
    )


def write_parquet(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    pa, pq = _require_pyarrow()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=dataset_schema())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_parquet(path: Path) -> list[dict[str, Any]]:
    _, pq = _require_pyarrow()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise InputError(f"Dataset does not exist: {path}. Run `osumapper dataset scan`.")
    return pq.read_table(path).to_pylist()


def write_json(path: Path, value: Any) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Could not read JSON file {path}: {exc}") from exc

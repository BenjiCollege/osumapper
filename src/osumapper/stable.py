from __future__ import annotations

from pathlib import Path

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode
from osumapper.errors import InputError


def find_songs_directory(root: Path) -> Path:
    root = root.expanduser().resolve()
    songs = root if root.name.casefold() == "songs" else root / "Songs"
    if not songs.is_dir():
        raise InputError(f"osu!stable Songs directory not found under: {root}")
    return songs


def scan_stable_maps(root: Path, mode: GameMode | None = None) -> list[Path]:
    songs = find_songs_directory(root)
    results: list[Path] = []
    for path in sorted(songs.rglob("*.osu")):
        try:
            document = BeatmapDocument.read(path)
        except InputError:
            continue
        if mode is None or document.mode is mode:
            results.append(path)
    return results


def write_maplist(paths: list[Path], output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8", newline="\n")
    return output

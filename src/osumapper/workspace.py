from __future__ import annotations

import re
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode
from osumapper.errors import InputError, PackageSafetyError
from osumapper.timing import create_timed_beatmap, estimate_timing

MAX_ARCHIVE_FILES = 20_000
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
SUPPORTED_AUDIO = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac", ".opus"}
_QUOTED_ASSET = re.compile(r'"(?P<name>[^"\r\n]+)"')


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _add_default_background(document: BeatmapDocument, root: Path) -> BeatmapDocument:
    filename = "osumapper-background.png"
    width, height = 1280, 720
    pixel = bytes((11, 16, 32))
    raw = (b"\x00" + pixel * width) * height
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    (root / filename).write_bytes(png)
    return document.replace_section(
        "Events",
        ["//Background and Video events", f'0,0,"{filename}",0,0'],
    )


def _copy_explicit_map_assets(source: Path, root: Path, document: BeatmapDocument) -> None:
    names = {
        match.group("name")
        for line in document.sections().get("Events", [])
        for match in _QUOTED_ASSET.finditer(line)
    }
    names.update(
        path.name
        for path in source.parent.iterdir()
        if path.is_file() and path.suffix.casefold() == ".osb"
    )
    for name in sorted(names, key=str.casefold):
        relative = _safe_relative_path(name.replace("\\", "/"))
        source_asset = source.parent / relative
        if not source_asset.is_file():
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, target)


def _safe_relative_path(name: str) -> Path:
    if "\\" in name:
        raise PackageSafetyError(f"Archive member uses unsafe separators: {name}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PackageSafetyError(f"Unsafe archive member path: {name}")
    if ":" in pure.parts[0]:
        raise PackageSafetyError(f"Archive member contains a drive path: {name}")
    return Path(*pure.parts)


def safe_extract_osz(archive: Path, destination: Path) -> None:
    try:
        package = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise InputError(f"Invalid .osz package: {archive}") from exc
    with package:
        members = package.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise PackageSafetyError(
                f"Package contains {len(members)} files; limit is {MAX_ARCHIVE_FILES}."
            )
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_ARCHIVE_BYTES:
            raise PackageSafetyError(
                f"Package expands to {total_size} bytes; limit is {MAX_ARCHIVE_BYTES}."
            )
        for member in members:
            relative = _safe_relative_path(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PackageSafetyError(
                    f"Symbolic links are not allowed in .osz: {member.filename}"
                )
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def select_beatmap(paths: list[Path], difficulty: str | None) -> BeatmapDocument:
    if not paths:
        raise InputError("The input does not contain a .osu difficulty.")
    documents = [BeatmapDocument.read(path) for path in sorted(paths)]
    if difficulty:
        needle = difficulty.casefold()
        matches = [doc for doc in documents if needle in doc.version_name.casefold()]
        if not matches:
            available = ", ".join(document.version_name for document in documents)
            raise InputError(f"No difficulty matches {difficulty!r}. Available: {available}")
        if len(matches) > 1:
            raise InputError(f"Difficulty selector {difficulty!r} matched multiple files.")
        return matches[0]
    return documents[0]


@dataclass(slots=True)
class SourceWorkspace:
    _temporary: tempfile.TemporaryDirectory[str]
    root: Path
    document: BeatmapDocument
    audio: Path

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> SourceWorkspace:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def prepare_source(
    source: Path,
    *,
    audio: Path | None = None,
    difficulty: str | None = None,
    mode: GameMode | None = None,
    bpm: float | None = None,
    offset_ms: int | None = None,
    key_count: int = 4,
) -> SourceWorkspace:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise InputError(f"Input does not exist: {source}")
    temporary = tempfile.TemporaryDirectory(prefix="osumapper-")
    root = Path(temporary.name)
    try:
        if source.suffix.casefold() == ".osz":
            safe_extract_osz(source, root)
            document = select_beatmap(list(root.rglob("*.osu")), difficulty)
        elif source.suffix.casefold() == ".osu":
            target = root / source.name
            shutil.copy2(source, target)
            document = BeatmapDocument.read(target)
            _copy_explicit_map_assets(source, root, document)
        elif source.suffix.casefold() in SUPPORTED_AUDIO:
            selected_mode = mode or GameMode.STANDARD
            target_audio = root / source.name
            shutil.copy2(source, target_audio)
            estimate = None if bpm is not None else estimate_timing(target_audio)
            chosen_bpm = bpm if bpm is not None else estimate.bpm
            chosen_offset = (
                offset_ms if offset_ms is not None else (estimate.offset_ms if estimate else 0)
            )
            text = create_timed_beatmap(
                target_audio.name,
                mode=selected_mode,
                bpm=chosen_bpm,
                offset_ms=chosen_offset,
                key_count=key_count,
                title=source.stem,
            )
            target_map = root / f"{source.stem} [Auto-timed].osu"
            target_map.write_text(text, encoding="utf-8", newline="")
            document = BeatmapDocument.read(target_map)
            document = _add_default_background(document, root)
        else:
            raise InputError("Input must be a .osz, .osu, or supported audio file.")

        audio_relative = _safe_relative_path(document.audio_filename.replace("\\", "/"))
        target_audio = document.path.parent / audio_relative
        if audio is not None:
            supplied_audio = audio.expanduser().resolve()
            if not supplied_audio.is_file():
                raise InputError(f"Audio file does not exist: {supplied_audio}")
            target_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(supplied_audio, target_audio)
        elif not target_audio.is_file() and source.suffix.casefold() == ".osu":
            sibling = source.parent / audio_relative
            if sibling.is_file():
                target_audio.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sibling, target_audio)
        if not target_audio.is_file():
            raise InputError(
                f"Beatmap references missing audio {document.audio_filename!r}; provide --audio."
            )
        return SourceWorkspace(temporary, root, document, target_audio)
    except Exception:
        temporary.cleanup()
        raise

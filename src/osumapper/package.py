from __future__ import annotations

import os
import tempfile
import unicodedata
import zipfile
from pathlib import Path

from osumapper.beatmap import BeatmapDocument
from osumapper.errors import InputError

_INVALID_FILENAME = '<>:"/\\|?*'


def safe_filename(value: str, suffix: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip().rstrip(".")
    cleaned = "".join(
        "_" if char in _INVALID_FILENAME or ord(char) < 32 else char for char in normalized
    )
    cleaned = " ".join(cleaned.split())[:180].strip(" .")
    return f"{cleaned or 'osumapper-generated'}{suffix}"


def generated_map_name(document: BeatmapDocument, preset: str) -> str:
    metadata = document.values("Metadata")
    artist = metadata.get("ArtistUnicode") or metadata.get("Artist") or "Unknown Artist"
    title = metadata.get("TitleUnicode") or metadata.get("Title") or "Untitled"
    return safe_filename(f"{artist} - {title} (osumapper) [{preset}]", ".osu")


def _archive_paths(root: Path, excluded: set[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise InputError(f"Refusing to package symbolic link: {path}")
        if path.is_file() and path.resolve() not in excluded:
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix().casefold())


def write_osz(root: Path, output: Path, *, exclude: set[Path] | None = None) -> Path:
    root = root.resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {item.resolve() for item in (exclude or set())}
    files = _archive_paths(root, excluded)
    if not any(path.suffix.casefold() == ".osu" for path in files):
        raise InputError("Cannot create an .osz without a .osu difficulty.")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    validate_osz(output)
    return output


def validate_osz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.casefold().endswith(".osu") for name in names):
                raise InputError(f"Generated package has no .osu difficulty: {path}")
            bad = archive.testzip()
            if bad:
                raise InputError(f"Generated package contains a corrupt entry: {bad}")
    except zipfile.BadZipFile as exc:
        raise InputError(f"Generated package is not a valid .osz: {path}") from exc

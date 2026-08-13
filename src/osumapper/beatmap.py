from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osumapper.config import GameMode
from osumapper.errors import InputError

_HEADER = re.compile(r"^osu file format v(?P<version>\d+)\s*$", re.IGNORECASE)
_SECTION = re.compile(r"^\[(?P<name>[^]]+)]\s*$")


@dataclass(frozen=True, slots=True)
class BeatmapDocument:
    path: Path
    text: str

    @classmethod
    def read(cls, path: Path) -> BeatmapDocument:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise InputError(f"Beatmap does not exist: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputError(f"Beatmap is not valid UTF-8: {path}") from exc
        document = cls(path=path, text=text.replace("\r\n", "\n").replace("\r", "\n"))
        document.validate()
        return document

    @classmethod
    def from_text(cls, text: str, path: Path | None = None) -> BeatmapDocument:
        document = cls(
            path=(path or Path("generated.osu")),
            text=text.replace("\r\n", "\n").replace("\r", "\n"),
        )
        document.validate()
        return document

    def validate(self) -> None:
        lines = self.text.splitlines()
        if not lines or not _HEADER.match(lines[0].lstrip("\ufeff")):
            raise InputError(f"Not a supported .osu file: {self.path}")
        required = {"General", "Metadata", "Difficulty", "TimingPoints"}
        missing = sorted(required - self.sections().keys())
        if missing:
            raise InputError(f"Beatmap is missing sections: {', '.join(missing)}")

    @property
    def format_version(self) -> int:
        match = _HEADER.match(self.text.splitlines()[0].lstrip("\ufeff"))
        assert match is not None
        return int(match.group("version"))

    def sections(self) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        current: str | None = None
        for line in self.text.splitlines()[1:]:
            match = _SECTION.match(line.strip())
            if match:
                current = match.group("name")
                output.setdefault(current, [])
            elif current is not None:
                output[current].append(line)
        return output

    def values(self, section: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.sections().get(section, []):
            if ":" not in line or line.lstrip().startswith("//"):
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    def value(self, section: str, key: str, default: str | None = None) -> str | None:
        return self.values(section).get(key, default)

    @property
    def audio_filename(self) -> str:
        value = self.value("General", "AudioFilename")
        if not value:
            raise InputError(f"Beatmap has no AudioFilename: {self.path}")
        return value

    @property
    def mode(self) -> GameMode:
        raw = self.value("General", "Mode", "0")
        try:
            return GameMode(int(raw or 0))
        except (ValueError, TypeError) as exc:
            raise InputError(f"Invalid Mode value in {self.path}: {raw}") from exc

    @property
    def version_name(self) -> str:
        return self.value("Metadata", "Version", "Unknown") or "Unknown"

    def timing_points(self) -> dict[str, list[dict[str, Any]]]:
        difficulty = self.values("Difficulty")
        slider_multiplier = float(difficulty.get("SliderMultiplier", "1.4"))
        inherited: list[dict[str, Any]] = []
        uninherited: list[dict[str, Any]] = []
        current_tick_length = 500.0
        for line in self.sections().get("TimingPoints", []):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            fields = [part.strip() for part in stripped.split(",")]
            if len(fields) < 2:
                continue
            begin = int(round(float(fields[0])))
            raw_length = float(fields[1])
            meter = int(fields[2]) if len(fields) > 2 and fields[2] else 4
            sample_index = int(fields[3]) if len(fields) > 3 and fields[3] else 2
            custom_set = int(fields[4]) if len(fields) > 4 and fields[4] else 1
            volume = int(fields[5]) if len(fields) > 5 and fields[5] else 100
            is_uninherited = len(fields) < 7 or fields[6] != "0"
            effects = int(fields[7]) if len(fields) > 7 and fields[7] else 0
            if is_uninherited and raw_length > 0:
                current_tick_length = raw_length
                slider_length = 100.0 * slider_multiplier
            else:
                slider_length = 100.0 * slider_multiplier * (100.0 / max(0.01, -raw_length))
            item = {
                "beginTime": begin,
                "whiteLines": meter,
                "sampleSet": "normal"
                if sample_index == 1
                else "drum"
                if sample_index == 3
                else "soft",
                "customSet": custom_set,
                "volume": volume,
                "isKiai": bool(effects & 1),
                "isInherited": not is_uninherited,
                "tickLength": current_tick_length,
                "sliderLength": slider_length,
                "bpm": max(12.0, 60000.0 / current_tick_length),
            }
            inherited.append(item)
            if is_uninherited:
                uninherited.append(item)
        if not uninherited:
            raise InputError(f"Beatmap has no uninherited timing point: {self.path}")
        return {"ts": inherited, "uts": uninherited}

    def legacy_dict(self) -> dict[str, Any]:
        general: dict[str, Any] = self.values("General")
        for key in ("AudioLeadIn", "PreviewTime", "Countdown", "Mode"):
            if key in general:
                with contextlib.suppress(ValueError):
                    general[key] = int(general[key])
        diff_values = self.values("Difficulty")
        circle_size = float(diff_values.get("CircleSize", "4"))
        diff = {
            "HD": float(diff_values.get("HPDrainRate", "5")),
            "CS": int(circle_size) if circle_size.is_integer() else circle_size,
            "OD": float(diff_values.get("OverallDifficulty", "5")),
            "AR": float(diff_values.get("ApproachRate", diff_values.get("OverallDifficulty", "5"))),
            "SV": float(diff_values.get("SliderMultiplier", "1.4")),
            "STR": int(float(diff_values.get("SliderTickRate", "1"))),
        }
        meta_values = self.values("Metadata")
        meta = {
            "title": meta_values.get("Title", "Unknown"),
            "titleUnicode": meta_values.get("TitleUnicode", ""),
            "artist": meta_values.get("Artist", "Unknown"),
            "artistUnicode": meta_values.get("ArtistUnicode", ""),
            "creator": meta_values.get("Creator", "osumapper"),
            "diffname": meta_values.get("Version", "Generated"),
            "source": meta_values.get("Source", ""),
            "tags": meta_values.get("Tags", ""),
        }
        return {
            "fileVersion": str(self.format_version),
            "general": general,
            "editor": "\n".join(self.sections().get("Editor", [])),
            "meta": meta,
            "diff": diff,
            "evt": "\n".join(self.sections().get("Events", [])),
            "timing": self.timing_points(),
            "color": "\n".join(self.sections().get("Colours", [])),
            "obj": [],
        }

    def replace_section(self, name: str, new_lines: list[str]) -> BeatmapDocument:
        lines = self.text.splitlines()
        start: int | None = None
        end = len(lines)
        for index, line in enumerate(lines):
            match = _SECTION.match(line.strip())
            if match and match.group("name") == name:
                start = index
                continue
            if start is not None and match:
                end = index
                break
        replacement = [f"[{name}]", *new_lines, ""]
        if start is None:
            updated = [*lines, "", *replacement]
        else:
            updated = [*lines[:start], *replacement, *lines[end:]]
        return BeatmapDocument.from_text("\n".join(updated).rstrip() + "\n", self.path)

    def set_value(self, section: str, key: str, value: str) -> BeatmapDocument:
        lines = list(self.sections().get(section, []))
        replacement = f"{key}:{value}"
        for index, line in enumerate(lines):
            if line.split(":", 1)[0].strip() == key:
                lines[index] = replacement
                break
        else:
            lines.append(replacement)
        return self.replace_section(section, lines)

    def with_hit_objects(
        self,
        objects: list[dict[str, Any]],
        *,
        preset: str,
        seed: int,
    ) -> BeatmapDocument:
        output = self.set_value("Metadata", "Creator", "osumapper")
        version = f"{self.version_name} - osumapper {preset} seed {seed}"
        output = output.set_value("Metadata", "Version", version)
        margin = 0
        normalized = objects
        if output.mode is GameMode.STANDARD:
            combo_colours = [
                line
                for line in output.sections().get("Colours", [])
                if line.strip().casefold().startswith("combo") and ":" in line
            ]
            if len(combo_colours) < 2:
                output = output.replace_section(
                    "Colours",
                    [
                        "Combo1 : 91,140,255",
                        "Combo2 : 66,200,154",
                        "Combo3 : 255,107,122",
                        "Combo4 : 238,242,255",
                    ],
                )
            try:
                circle_size = float(output.value("Difficulty", "CircleSize", "4") or 4)
            except ValueError:
                circle_size = 4.0
            margin = max(0, math.ceil(54.4 - 4.48 * circle_size))
            normalized = []
            for obj in objects:
                updated = dict(obj)
                if int(updated.get("type", 1)) & 8:
                    updated["x"] = 256
                    updated["y"] = 192
                normalized.append(updated)
        return output.replace_section(
            "HitObjects",
            [serialize_hit_object(obj, playfield_margin=margin) for obj in normalized],
        )


def _clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def serialize_hit_object(obj: dict[str, Any], *, playfield_margin: int = 0) -> str:
    margin = max(0, min(192, int(playfield_margin)))
    x = _clamp(float(obj.get("x", 256)), margin, 512 - margin)
    y = _clamp(float(obj.get("y", 192)), margin, 384 - margin)
    timestamp = max(0, int(round(float(obj.get("time", 0)))))
    object_type = int(obj.get("type", 1))
    combo_flags = object_type & (4 | 16 | 32 | 64)
    hitsounds = int(obj.get("hitsounds", 0))
    extended = str(obj.get("extHitsounds") or "0:0:0:0:")

    if object_type & 2:
        generator = obj.get("sliderGenerator") or {}
        endpoint = generator.get("endpoint")
        length = max(1.0, float(generator.get("len", 100.0)))
        # `slides` counts traversals, so 1 is a plain slider and 2 is one repeat.
        # osu! requires one edge sound and one edge sample set per traversal edge.
        slides = max(1, min(16, int(generator.get("repeats", 1))))
        if not endpoint:
            direction = generator.get("dOut", [1.0, 0.0])
            endpoint = [x + float(direction[0]) * length, y + float(direction[1]) * length]
        end_x = _clamp(float(endpoint[0]), margin, 512 - margin)
        end_y = _clamp(float(endpoint[1]), margin, 384 - margin)
        edge_sounds = "|".join(["0"] * (slides + 1))
        edge_sets = "|".join(["0:0"] * (slides + 1))
        return (
            f"{x},{y},{timestamp},{2 | combo_flags},{hitsounds},L|{end_x}:{end_y},{slides},"
            f"{length:.3f},{edge_sounds},{edge_sets},{extended}"
        )
    if object_type & 8:
        end_time = max(timestamp + 1, int(obj.get("spinnerEndTime", timestamp + 1000)))
        return f"{x},{y},{timestamp},{8 | combo_flags},{hitsounds},{end_time},{extended}"
    if object_type & 128:
        end_time = max(timestamp + 1, int(obj.get("holdEndTime", timestamp + 1)))
        return f"{x},{y},{timestamp},128,{hitsounds},{end_time}:{extended}"
    return f"{x},{y},{timestamp},{1 | combo_flags},{hitsounds},{extended}"

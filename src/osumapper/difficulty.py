from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osumapper.beatmap import BeatmapDocument
from osumapper.errors import DependencyError, InputError


@dataclass(frozen=True, slots=True)
class StandardDifficulty:
    key: str
    label: str
    minimum_stars: float
    maximum_stars: float | None
    default_stars: float
    target_density: float
    hp: float
    cs: float
    od: float
    ar: float

    @property
    def range_label(self) -> str:
        if self.maximum_stars is None:
            return f"{self.minimum_stars:.1f}★ and above"
        return f"{self.minimum_stars:.1f}★–{self.maximum_stars - 0.01:.2f}★"

    def contains(self, stars: float) -> bool:
        return stars >= self.minimum_stars and (
            self.maximum_stars is None or stars < self.maximum_stars
        )

    def validate_target(self, stars: float) -> float:
        value = float(stars)
        if not self.contains(value):
            raise InputError(f"{self.label} requires {self.range_label}; received {value:.2f}★.")
        return value


STANDARD_DIFFICULTIES: tuple[StandardDifficulty, ...] = (
    StandardDifficulty("easy", "Easy", 0.0, 2.0, 1.50, 0.70, 3.0, 3.5, 3.0, 4.0),
    StandardDifficulty("normal", "Normal", 2.0, 2.7, 2.35, 1.10, 4.0, 3.8, 5.0, 6.0),
    StandardDifficulty("hard", "Hard", 2.7, 4.0, 3.35, 1.60, 5.0, 4.0, 7.0, 8.0),
    StandardDifficulty("insane", "Insane", 4.0, 5.3, 4.65, 2.20, 6.0, 4.0, 8.0, 9.0),
    StandardDifficulty("expert", "Expert", 5.3, 6.5, 5.90, 2.80, 6.5, 4.0, 9.0, 9.5),
    StandardDifficulty("expert-plus", "Expert+", 6.5, None, 7.00, 3.50, 7.0, 4.0, 9.5, 10.0),
)

STANDARD_DIFFICULTY_KEYS = tuple(profile.key for profile in STANDARD_DIFFICULTIES)
STAR_TARGET_TOLERANCE = 0.25
STAR_DIFFICULTY_FEATURES = ("target_star_rating", "difficulty_tier")
LEGACY_DIFFICULTY_FEATURES = ("OD", "AR", "CS", "objects_per_second")


def standard_difficulty(value: str) -> StandardDifficulty:
    normalized = value.strip().casefold().replace("_", "-").replace("+", "-plus")
    aliases = {profile.key: profile for profile in STANDARD_DIFFICULTIES}
    aliases.update(
        {
            profile.label.casefold().replace("+", "-plus"): profile
            for profile in STANDARD_DIFFICULTIES
        }
    )
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(STANDARD_DIFFICULTY_KEYS)
        raise InputError(f"Unknown osu!standard difficulty {value!r}; choose {choices}.") from exc


def difficulty_for_stars(stars: float) -> StandardDifficulty:
    value = float(stars)
    if value < 0:
        raise InputError("Star rating cannot be negative.")
    return next(profile for profile in STANDARD_DIFFICULTIES if profile.contains(value))


def resolve_standard_difficulty(
    difficulty: str | None,
    target_stars: float | None,
) -> tuple[StandardDifficulty, float] | None:
    if difficulty is None and target_stars is None:
        return None
    profile = (
        standard_difficulty(difficulty)
        if difficulty is not None
        else difficulty_for_stars(float(target_stars))
    )
    stars = profile.default_stars if target_stars is None else profile.validate_target(target_stars)
    return profile, stars


def _require_rosu() -> Any:
    try:
        import rosu_pp_py as rosu
    except ImportError as exc:
        raise DependencyError(
            "Star calculation requires rosu-pp-py. Run `uv sync --locked`."
        ) from exc
    return rosu


def calculate_standard_stars(source: Path | BeatmapDocument) -> float:
    rosu = _require_rosu()
    try:
        beatmap = (
            rosu.Beatmap(content=source.text)
            if isinstance(source, BeatmapDocument)
            else rosu.Beatmap(path=str(source.expanduser().resolve()))
        )
        if beatmap.is_suspicious():
            raise InputError(f"Refusing star calculation for suspicious beatmap: {source}")
        attributes = rosu.Difficulty().calculate(beatmap)
        return float(attributes.stars)
    except InputError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise InputError(f"Could not calculate osu!standard stars for {source}: {exc}") from exc


def apply_standard_difficulty(
    document: BeatmapDocument,
    profile: StandardDifficulty,
) -> BeatmapDocument:
    output = document.set_value("General", "Mode", "0")
    for key, value in (
        ("HPDrainRate", profile.hp),
        ("CircleSize", profile.cs),
        ("OverallDifficulty", profile.od),
        ("ApproachRate", profile.ar),
    ):
        output = output.set_value("Difficulty", key, f"{value:g}")
    return output


def label_standard_difficulty(
    document: BeatmapDocument,
    profile: StandardDifficulty,
    actual_stars: float,
) -> BeatmapDocument:
    return document.set_value(
        "Metadata",
        "Version",
        f"{profile.label} {actual_stars:.2f}★ - {document.version_name}",
    )

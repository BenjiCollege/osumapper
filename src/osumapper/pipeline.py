from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from osumapper.config import GameMode, GenerationConfig
from osumapper.difficulty import resolve_standard_difficulty
from osumapper.engine import generate_document
from osumapper.errors import InputError
from osumapper.lazer import open_in_lazer
from osumapper.package import generated_map_name, write_osz
from osumapper.presets import get_preset
from osumapper.workspace import SUPPORTED_AUDIO, prepare_source

Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    source: Path
    output: Path
    preset: str = "default"
    mode: GameMode | None = None
    difficulty: str | None = None
    seed: int = 2026
    flow_engine: str = "auto"
    rhythm_engine: str = "legacy"
    modern_model: Path | None = None
    placement_model: Path | None = None
    rhythm_threshold: float | None = None
    target_density: float | None = None
    difficulty_tier: str | None = None
    target_stars: float | None = None
    audio: Path | None = None
    bpm: float | None = None
    offset_ms: int | None = None
    key_count: int = 4
    open_in_lazer: bool = False


def generate_package(request: GenerationRequest, *, progress: Progress = print) -> Path:
    source = request.source.expanduser().resolve()
    output = request.output.expanduser().resolve()
    if output == source:
        raise InputError("Output must not overwrite the input package.")
    if request.rhythm_threshold is not None and not 0 < request.rhythm_threshold < 1:
        raise InputError("Rhythm threshold must be between 0 and 1.")
    if request.target_density is not None and not 0 < request.target_density <= 20:
        raise InputError("Target density must be greater than 0 and at most 20.")
    if request.flow_engine == "placement" and request.rhythm_engine != "modern":
        raise InputError("Placement-v1 requires --rhythm-engine modern.")
    requested_difficulty = resolve_standard_difficulty(
        request.difficulty_tier, request.target_stars
    )
    if requested_difficulty is not None and request.rhythm_engine != "modern":
        raise InputError("Fixed star difficulty generation requires --rhythm-engine modern.")
    if requested_difficulty is not None and request.mode not in {None, GameMode.STANDARD}:
        raise InputError("Fixed star difficulty generation supports osu!standard only.")
    config = GenerationConfig(
        preset=request.preset,
        mode=request.mode,
        seed=request.seed,
        difficulty=request.difficulty,
        output=output,
        open_in_lazer=request.open_in_lazer,
        flow_engine=request.flow_engine,
        rhythm_engine=request.rhythm_engine,
        modern_model=request.modern_model,
        placement_model=request.placement_model,
        rhythm_threshold=request.rhythm_threshold,
        target_density=request.target_density,
        difficulty_tier=(requested_difficulty[0].key if requested_difficulty else None),
        target_stars=(requested_difficulty[1] if requested_difficulty else None),
    )
    progress("Opening input in an isolated workspace")
    with prepare_source(
        source,
        audio=request.audio,
        difficulty=request.difficulty,
        mode=request.mode,
        bpm=request.bpm,
        offset_ms=request.offset_ms,
        key_count=request.key_count,
    ) as workspace:
        if requested_difficulty is not None and workspace.document.mode is not GameMode.STANDARD:
            raise InputError("The selected source difficulty is not osu!standard mode.")
        source_mode = request.mode or workspace.document.mode
        preset = get_preset(request.preset, source_mode)
        generated = generate_document(
            workspace.document,
            workspace.audio,
            workspace.root,
            preset,
            config,
            progress,
        )
        generated_path = workspace.root / generated_map_name(generated, preset.name)
        generated_path.write_text(generated.text, encoding="utf-8", newline="\r\n")
        excluded = {
            workspace.root / "mapthis.json",
            workspace.root / "mapthis.npz",
            workspace.root / "rhythm_data.npz",
            workspace.root / "modern-rhythm-features.npz",
        }
        if source.suffix.casefold() in SUPPORTED_AUDIO:
            excluded.add(workspace.document.path)
        progress("Building deterministic .osz package")
        write_osz(workspace.root, output, exclude=excluded)
    if request.open_in_lazer:
        progress("Opening package in osu!lazer")
        open_in_lazer(output)
    return output

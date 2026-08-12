from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from osumapper.beatmap import BeatmapDocument
from osumapper.config import GameMode, GenerationConfig, PresetSpec
from osumapper.criteria import audit_standard_criteria
from osumapper.difficulty import (
    STANDARD_DIFFICULTIES,
    STAR_TARGET_TOLERANCE,
    StandardDifficulty,
    calculate_standard_stars,
    resolve_standard_difficulty,
)
from osumapper.engine import ensure_preview_time, generate_document
from osumapper.errors import GenerationError, InputError
from osumapper.lazer import open_in_lazer
from osumapper.package import generated_map_name, validate_osz, write_osz
from osumapper.presets import get_preset
from osumapper.training.storage import write_json
from osumapper.workspace import SUPPORTED_AUDIO, prepare_source

Progress = Callable[[str], None]
FULL_SET_DEFAULT_MAX_ATTEMPTS = 16
FULL_SET_DEFAULT_STAR_PRECISION = 0.03
FULL_SET_MAX_MEAN_STAR_ERROR = 0.15
_GAMEPLAY_BLOCKING_CRITERIA = {
    "objects_on_same_tick",
    "circle_gap_below_10ms",
    "long_object_gap_below_20ms",
    "objects_partially_offscreen_4_3",
    "duplicate_timing_points",
    "first_uninherited_point_enables_kiai",
    "drain_time_below_30_seconds",
}


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
    full_set: bool = False
    enforce_star_target: bool = True
    star_precision: float = FULL_SET_DEFAULT_STAR_PRECISION
    calibration_attempts: int = FULL_SET_DEFAULT_MAX_ATTEMPTS


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    attempt: int
    phase: str
    density: float
    flow_scale: float
    actual_stars: float
    star_error: float


@dataclass(frozen=True, slots=True)
class FullSetDifficultyResult:
    profile: StandardDifficulty
    document: BeatmapDocument
    actual_stars: float
    density: float
    flow_scale: float
    attempts: int
    requested_precision: float
    calibration_history: tuple[CalibrationAttempt, ...]


def _base_config(
    request: GenerationRequest,
    output: Path,
    requested_difficulty: tuple[StandardDifficulty, float] | None,
) -> GenerationConfig:
    return GenerationConfig(
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
        enforce_star_target=request.enforce_star_target,
    )


def _next_density(current: float, target_stars: float, actual_stars: float) -> float:
    ratio = target_stars / max(0.1, actual_stars)
    adjustment = max(0.7, min(1.45, ratio))
    candidate = max(0.1, min(20.0, current * adjustment))
    if abs(candidate - current) < 0.01:
        candidate = max(0.1, min(20.0, current + (0.1 if ratio >= 1 else -0.1)))
    return candidate


def _next_full_set_controls(
    profile: StandardDifficulty,
    density: float,
    flow_scale: float,
    actual_stars: float,
) -> tuple[float, float]:
    ratio = profile.default_stars / max(0.1, actual_stars)
    if profile.key in {"easy", "normal"}:
        next_density = density * max(0.7, min(1.3, ratio**0.6))
        next_flow = flow_scale * max(0.75, min(1.25, ratio**0.5))
        return max(0.1, min(20.0, next_density)), max(0.5, min(2.0, next_flow))
    if profile.key not in {"expert", "expert-plus"}:
        return _next_density(density, profile.default_stars, actual_stars), flow_scale
    next_density = density * max(0.9, min(1.1, ratio**0.25))
    next_flow = flow_scale * max(0.75, min(1.35, ratio**1.2))
    return max(0.1, min(20.0, next_density)), max(0.5, min(2.0, next_flow))


def _next_precision_flow(
    history: list[CalibrationAttempt],
    *,
    density: float,
    target_stars: float,
) -> float:
    """Choose the next spacing scale using local measured star results.

    Density remains fixed during this phase, which preserves the model-selected
    rhythm timestamps. Once two measurements bracket the target, linear
    interpolation converges much faster than another density adjustment.
    """

    same_rhythm = [item for item in history if abs(item.density - density) < 1e-9]
    below = sorted(
        (item for item in same_rhythm if item.actual_stars <= target_stars),
        key=lambda item: target_stars - item.actual_stars,
    )
    above = sorted(
        (item for item in same_rhythm if item.actual_stars >= target_stars),
        key=lambda item: item.actual_stars - target_stars,
    )
    candidate: float | None = None
    if below and above and below[0].actual_stars != above[0].actual_stars:
        low, high = below[0], above[0]
        fraction = (target_stars - low.actual_stars) / (high.actual_stars - low.actual_stars)
        candidate = low.flow_scale + fraction * (high.flow_scale - low.flow_scale)
    elif len(same_rhythm) >= 2:
        first, second = sorted(same_rhythm, key=lambda item: item.attempt)[-2:]
        star_delta = second.actual_stars - first.actual_stars
        flow_delta = second.flow_scale - first.flow_scale
        if abs(star_delta) >= 1e-4 and abs(flow_delta) >= 1e-5:
            candidate = second.flow_scale + (
                (target_stars - second.actual_stars) * flow_delta / star_delta
            )
    if candidate is None:
        latest = same_rhythm[-1]
        ratio = target_stars / max(0.1, latest.actual_stars)
        candidate = latest.flow_scale * max(0.82, min(1.22, ratio**1.5))

    candidate = max(0.45, min(2.0, candidate))
    # Prevent rounding back onto an already measured point. Step toward the target
    # in small increments while staying inside the PatternPlanner scale bounds.
    measured = {round(item.flow_scale, 6) for item in same_rhythm}
    if round(candidate, 6) in measured:
        direction = 1.0 if same_rhythm[-1].actual_stars < target_stars else -1.0
        for step in (0.01, 0.02, 0.04, 0.08):
            alternative = max(0.45, min(2.0, candidate + direction * step))
            if round(alternative, 6) not in measured:
                candidate = alternative
                break
    return candidate


def _generate_full_set_difficulty(
    document: BeatmapDocument,
    audio: Path,
    workspace: Path,
    preset: PresetSpec,
    base_config: GenerationConfig,
    profile: StandardDifficulty,
    star_precision: float,
    max_attempts: int,
    progress: Progress,
) -> FullSetDifficultyResult:
    density = profile.target_density
    flow_scale = 1.0
    best: tuple[BeatmapDocument, float, float, float, int] | None = None
    history: list[CalibrationAttempt] = []
    fine_density: float | None = None
    for attempt in range(1, max_attempts + 1):
        phase = "fine-spacing" if fine_density is not None else "coarse-density"
        progress(
            f"Generating {profile.label} {profile.default_stars:.2f}★ "
            f"(precision {star_precision:.3f}★, {phase}, attempt {attempt}/{max_attempts}, "
            f"density {density:.3f}, spacing {flow_scale:.3f}×)"
        )
        config = replace(
            base_config,
            mode=GameMode.STANDARD,
            difficulty_tier=profile.key,
            target_stars=profile.default_stars,
            target_density=density,
            enforce_star_target=False,
            flow_scale=flow_scale,
        )
        generated = generate_document(
            document,
            audio,
            workspace,
            preset,
            config,
            progress,
        )
        actual_stars = calculate_standard_stars(generated)
        error = abs(actual_stars - profile.default_stars)
        history.append(
            CalibrationAttempt(
                attempt=attempt,
                phase=phase,
                density=density,
                flow_scale=flow_scale,
                actual_stars=actual_stars,
                star_error=error,
            )
        )
        if best is None or error < abs(best[1] - profile.default_stars):
            best = (generated, actual_stars, density, flow_scale, attempt)
        if profile.contains(actual_stars) and error <= star_precision:
            progress(
                f"Precision target reached for {profile.label}: error {error:.4f}★ "
                f"within {star_precision:.3f}★"
            )
            return FullSetDifficultyResult(
                profile=profile,
                document=generated,
                actual_stars=actual_stars,
                density=density,
                flow_scale=flow_scale,
                attempts=attempt,
                requested_precision=star_precision,
                calibration_history=tuple(history),
            )
        if fine_density is not None or error <= STAR_TARGET_TOLERANCE:
            if fine_density is None:
                fine_density = density
                progress(
                    f"Locking {profile.label} rhythm at density {fine_density:.3f}; "
                    "fine-tuning only PatternPlanner spacing"
                )
            density = fine_density
            flow_scale = _next_precision_flow(
                history,
                density=density,
                target_stars=profile.default_stars,
            )
        else:
            fine_density = None
            density, flow_scale = _next_full_set_controls(
                profile,
                density,
                flow_scale,
                actual_stars,
            )
    assert best is not None
    _best_document, actual_stars, used_density, used_flow_scale, _attempts = best
    raise GenerationError(
        f"FullSet-v1 could not reach {profile.label} {profile.default_stars:.2f}★ "
        f"within the requested ±{star_precision:.3f}★ after {max_attempts} attempts. "
        f"Best result was {actual_stars:.2f}★ at density {used_density:.3f} and "
        f"spacing {used_flow_scale:.3f}×. "
        "Increase --calibration-attempts or relax --star-precision. The package was not "
        "exported because its measured margin exceeded the requested precision."
    )


def _assert_gameplay_safe(report: dict[str, Any], profile: StandardDifficulty) -> None:
    blocking = [
        issue
        for issue in report["issues"]
        if issue["code"] in _GAMEPLAY_BLOCKING_CRITERIA and issue["severity"] == "error"
    ]
    if not blocking:
        return
    detail = ", ".join(f"{item['code']} ({item['count']})" for item in blocking)
    raise GenerationError(
        f"Generated {profile.label} failed objective gameplay safety checks: {detail}."
    )


def _cache_exclusions(root: Path) -> set[Path]:
    return {
        root / "mapthis.json",
        root / "mapthis.npz",
        root / "rhythm_data.npz",
        root / "modern-rhythm-features.npz",
    }


def _unique_generated_path(
    parent: Path,
    document: BeatmapDocument,
    preset: str,
    difficulty_label: str,
    reserved: set[Path],
) -> Path:
    candidate = parent / generated_map_name(
        document,
        preset,
        difficulty_label=difficulty_label,
    )
    number = 2
    while candidate.resolve() in reserved or candidate.exists():
        candidate = candidate.with_name(f"{candidate.stem}-{number}{candidate.suffix}")
        number += 1
    reserved.add(candidate.resolve())
    return candidate


def _unused_workspace_audio(root: Path, selected_audio: Path) -> set[Path]:
    selected = selected_audio.resolve()
    return {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_AUDIO
        and path.resolve() != selected
    }


def _write_full_set_reports(
    output: Path,
    source: Path,
    seed: int,
    results: list[FullSetDifficultyResult],
    reports: dict[str, dict[str, Any]],
    progress: Progress,
) -> None:
    timing_sections = {
        tuple(result.document.sections().get("TimingPoints", [])) for result in results
    }
    difficulty_rows: list[dict[str, Any]] = []
    for result in results:
        report_path = output.with_name(f"{output.stem}.{result.profile.key}.criteria.json")
        write_json(report_path, reports[result.profile.key])
        progress(f"Wrote {result.profile.label} criteria audit: {report_path}")
        difficulty_rows.append(
            {
                "tier": result.profile.key,
                "label": result.profile.label,
                "target_stars": result.profile.default_stars,
                "actual_stars": result.actual_stars,
                "star_error": abs(result.actual_stars - result.profile.default_stars),
                "density": result.density,
                "flow_scale": result.flow_scale,
                "attempts": result.attempts,
                "requested_star_precision": result.requested_precision,
                "precision_target_met": (
                    abs(result.actual_stars - result.profile.default_stars)
                    <= result.requested_precision
                ),
                "calibration_history": [
                    {
                        "attempt": item.attempt,
                        "phase": item.phase,
                        "density": item.density,
                        "flow_scale": item.flow_scale,
                        "actual_stars": item.actual_stars,
                        "star_error": item.star_error,
                    }
                    for item in result.calibration_history
                ],
                "criteria_summary": reports[result.profile.key]["summary"],
                "pattern_summary": reports[result.profile.key].get("pattern_summary"),
                "map_name": generated_map_name(
                    result.document,
                    "full-set-v1",
                    difficulty_label=result.profile.label,
                ),
            }
        )
    errors = [row["star_error"] for row in difficulty_rows]
    set_report = {
        "version": 1,
        "generator": "full-set-v1",
        "source": str(source),
        "package": str(output),
        "seed": seed,
        "generated_difficulties": len(results),
        "expected_difficulties": len(STANDARD_DIFFICULTIES),
        "complete": len(results) == len(STANDARD_DIFFICULTIES),
        "shared_audio_analysis": True,
        "identical_timing_sections": len(timing_sections) == 1,
        "difficulty_nesting_enforced": False,
        "pattern_planner": "pattern-planner-v1"
        if all("pattern_summary" in report for report in reports.values())
        else None,
        "mean_star_error": sum(errors) / len(errors),
        "maximum_star_error": max(errors),
        "requested_star_precision": max(result.requested_precision for result in results),
        "all_precision_targets_met": all(
            abs(result.actual_stars - result.profile.default_stars) <= result.requested_precision
            for result in results
        ),
        "precision_calibration": {
            "algorithm": "two-phase-density-then-spacing-v1",
            "measurement": "rosu-pp-no-mod-stars",
            "rhythm_locked_during_fine_phase": True,
            "most_attempts_used": max(len(result.calibration_history) for result in results),
        },
        "difficulty_results": difficulty_rows,
        "rankability": "not-rankable-generated-draft",
        "next_model_milestones": ["conformer-v5-full-set", "placement-v2"],
        "limitations": [
            "V4 runs each difficulty independently; learned cross-tier nesting requires V5.",
            "PatternPlanner-v1 is deterministic and timing-aware, but learned Placement-v2, "
            "hitsound planning, and learned section-aware pattern planning are not yet trained.",
        ],
    }
    report_path = output.with_name(f"{output.stem}.full-set.json")
    write_json(report_path, set_report)
    progress(f"Wrote FullSet-v1 report: {report_path}")


def generate_package(request: GenerationRequest, *, progress: Progress = print) -> Path:
    source = request.source.expanduser().resolve()
    output = request.output.expanduser().resolve()
    if output == source:
        raise InputError("Output must not overwrite the input package.")
    if request.rhythm_threshold is not None and not 0 < request.rhythm_threshold < 1:
        raise InputError("Rhythm threshold must be between 0 and 1.")
    if request.target_density is not None and not 0 < request.target_density <= 20:
        raise InputError("Target density must be greater than 0 and at most 20.")
    if not 0.001 <= request.star_precision <= STAR_TARGET_TOLERANCE:
        raise InputError(f"Star precision must be between 0.001 and {STAR_TARGET_TOLERANCE:.2f}.")
    if not 1 <= request.calibration_attempts <= 30:
        raise InputError("Calibration attempts must be between 1 and 30.")
    if request.flow_engine == "placement" and request.rhythm_engine != "modern":
        raise InputError("Placement-v1 requires --rhythm-engine modern.")
    if request.full_set:
        if request.rhythm_engine != "modern":
            raise InputError("FullSet-v1 requires --rhythm-engine modern.")
        if request.mode not in {None, GameMode.STANDARD}:
            raise InputError("FullSet-v1 supports osu!standard only.")
        if request.target_density is not None:
            raise InputError("FullSet-v1 manages density per tier; omit --target-density.")
        if request.difficulty_tier is not None or request.target_stars is not None:
            raise InputError(
                "FullSet-v1 uses all six fixed targets; omit --difficulty-tier and --target-stars."
            )
    requested_difficulty = resolve_standard_difficulty(
        request.difficulty_tier, request.target_stars
    )
    if requested_difficulty is not None and request.rhythm_engine != "modern":
        raise InputError("Fixed star difficulty generation requires --rhythm-engine modern.")
    if requested_difficulty is not None and request.mode not in {None, GameMode.STANDARD}:
        raise InputError("Fixed star difficulty generation supports osu!standard only.")
    config = _base_config(request, output, requested_difficulty)
    criteria_report: dict[str, Any] | None = None
    full_set_results: list[FullSetDifficultyResult] = []
    full_set_reports: dict[str, dict[str, Any]] = {}
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
        workspace.document = ensure_preview_time(workspace.document, workspace.audio)
        if (requested_difficulty is not None or request.full_set) and (
            workspace.document.mode is not GameMode.STANDARD
        ):
            raise InputError("The selected source difficulty is not osu!standard mode.")
        source_mode = (
            GameMode.STANDARD if request.full_set else request.mode or workspace.document.mode
        )
        preset = get_preset(request.preset, source_mode)
        source_maps = {
            path.resolve()
            for path in workspace.root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".osu"
        }
        reserved_maps = set(source_maps)
        excluded = _cache_exclusions(workspace.root)

        if request.full_set:
            progress("Starting FullSet-v1 shared six-difficulty generation")
            for profile in STANDARD_DIFFICULTIES:
                result = _generate_full_set_difficulty(
                    workspace.document,
                    workspace.audio,
                    workspace.root,
                    preset,
                    config,
                    profile,
                    request.star_precision,
                    request.calibration_attempts,
                    progress,
                )
                generated_path = _unique_generated_path(
                    workspace.document.path.parent,
                    result.document,
                    preset.name,
                    profile.label,
                    reserved_maps,
                )
                generated_path.write_text(
                    result.document.text,
                    encoding="utf-8",
                    newline="\r\n",
                )
                report = audit_standard_criteria(
                    generated_path,
                    map_label=generated_path.name,
                )
                _assert_gameplay_safe(report, profile)
                full_set_results.append(result)
                full_set_reports[profile.key] = report
                summary = report["summary"]
                progress(
                    f"Accepted {profile.label} {result.actual_stars:.2f}★; "
                    f"criteria {summary['structural_error_occurrences']} structural errors, "
                    f"{summary['warnings']} warnings"
                )
            mean_star_error = sum(
                abs(result.actual_stars - result.profile.default_stars)
                for result in full_set_results
            ) / len(full_set_results)
            if mean_star_error >= FULL_SET_MAX_MEAN_STAR_ERROR:
                raise GenerationError(
                    f"FullSet-v1 mean star error was {mean_star_error:.3f}★; required below "
                    f"{FULL_SET_MAX_MEAN_STAR_ERROR:.2f}★. The package was not exported."
                )
            timing_sections = {
                tuple(result.document.sections().get("TimingPoints", []))
                for result in full_set_results
            }
            if len(timing_sections) != 1:
                raise GenerationError(
                    "FullSet-v1 produced inconsistent timing sections; "
                    "the package was not exported."
                )
            excluded.update(source_maps)
            excluded.update(_unused_workspace_audio(workspace.root, workspace.audio))
            progress("Building deterministic six-difficulty .osz package")
            write_osz(workspace.root, output, exclude=excluded)
            validate_osz(
                output,
                expected_osu_count=len(STANDARD_DIFFICULTIES),
                expected_audio_count=1,
            )
        else:
            generated = generate_document(
                workspace.document,
                workspace.audio,
                workspace.root,
                preset,
                config,
                progress,
            )
            generated_path = workspace.document.path.parent / generated_map_name(
                generated, preset.name
            )
            generated_path.write_text(generated.text, encoding="utf-8", newline="\r\n")
            if generated.mode is GameMode.STANDARD:
                criteria_report = audit_standard_criteria(
                    generated_path,
                    map_label=generated_path.name,
                )
                summary = criteria_report["summary"]
                progress(
                    "Audited objective osu!standard criteria: "
                    f"{summary['errors']} errors, {summary['warnings']} warnings; "
                    "generated output is a local, non-rankable draft"
                )
            if source.suffix.casefold() in SUPPORTED_AUDIO:
                excluded.add(workspace.document.path)
            progress("Building deterministic .osz package")
            write_osz(workspace.root, output, exclude=excluded)
    if request.full_set:
        _write_full_set_reports(
            output,
            source,
            request.seed,
            full_set_results,
            full_set_reports,
            progress,
        )
    elif criteria_report is not None:
        report_path = output.with_name(f"{output.stem}.criteria.json")
        write_json(report_path, criteria_report)
        progress(f"Wrote criteria audit: {report_path}")
    if request.open_in_lazer:
        progress("Opening package in osu!lazer")
        open_in_lazer(output)
    return output

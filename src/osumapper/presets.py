from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from osumapper.config import GameMode, PresetSpec
from osumapper.errors import InputError
from osumapper.paths import legacy_v7_root, modern_models_root

STANDARD_PRESETS = (
    "default",
    "sota",
    "vtuber",
    "flower",
    "inst",
    "lowbpm",
    "tvsize",
    "hard",
    "normal",
    "taiko",
    "catch",
)
MANIA_PRESETS = ("mania-lowkey", "mania-highkey")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InputError(f"Could not load legacy preset definitions: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _legacy_setup(mania: bool) -> ModuleType:
    filename = "mania_setup_colab.py" if mania else "setup_colab.py"
    return _load_module(legacy_v7_root() / filename, f"_osumapper_legacy_{filename[:-3]}")


def _model_path(relative: str) -> Path:
    legacy = legacy_v7_root() / relative
    folder = legacy.parent.name
    modern = modern_models_root() / folder / "rhythm.keras"
    return modern if modern.is_file() else legacy


def _absolute_optional(relative: str | None) -> Path | None:
    return legacy_v7_root() / relative if relative else None


def get_preset(name: str, source_mode: GameMode | None = None) -> PresetSpec:
    normalized = name.casefold()
    if normalized == "default" and source_mode is GameMode.MANIA:
        normalized = "mania-lowkey"
    elif normalized == "default" and source_mode is GameMode.TAIKO:
        normalized = "taiko"
    elif normalized == "default" and source_mode is GameMode.CATCH:
        normalized = "catch"

    if normalized in MANIA_PRESETS:
        key = normalized.removeprefix("mania-")
        data: dict[str, Any] = _legacy_setup(True).load_pretrained_model(key)
        return PresetSpec(
            name=normalized,
            mode=GameMode.MANIA,
            rhythm_model=_model_path(data["rhythm_model"]),
            flow_dataset=None,
            pattern_dataset=_absolute_optional(data.get("pattern_dataset")),
            hitsound_dataset=None,
            rhythm_params=tuple(data["rhythm_param"]),
            gan_params={},
            modding_params=dict(data["modding"]),
        )

    if normalized not in STANDARD_PRESETS:
        available = ", ".join((*STANDARD_PRESETS, *MANIA_PRESETS))
        raise InputError(f"Unknown preset {name!r}. Available: {available}")
    data = _legacy_setup(False).load_pretrained_model(normalized)
    mode = (
        GameMode.TAIKO
        if normalized == "taiko"
        else GameMode.CATCH
        if normalized == "catch"
        else GameMode.STANDARD
    )
    return PresetSpec(
        name=normalized,
        mode=mode,
        rhythm_model=_model_path(data["rhythm_model"]),
        flow_dataset=_absolute_optional(data.get("flow_dataset")),
        pattern_dataset=None,
        hitsound_dataset=_absolute_optional(data.get("hs_dataset")),
        rhythm_params=tuple(data["rhythm_param"]),
        gan_params=dict(data["gan"]),
        modding_params=dict(data["modding"]),
    )


def preset_names() -> tuple[str, ...]:
    return (*STANDARD_PRESETS, *MANIA_PRESETS)

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class GameMode(IntEnum):
    STANDARD = 0
    TAIKO = 1
    CATCH = 2
    MANIA = 3

    @classmethod
    def parse(cls, value: str | int | None) -> GameMode | None:
        if value is None or value == "auto":
            return None
        if isinstance(value, int) or str(value).isdigit():
            return cls(int(value))
        aliases = {
            "osu": cls.STANDARD,
            "standard": cls.STANDARD,
            "taiko": cls.TAIKO,
            "catch": cls.CATCH,
            "ctb": cls.CATCH,
            "mania": cls.MANIA,
        }
        try:
            return aliases[str(value).lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown game mode: {value}") from exc


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    preset: str = "default"
    mode: GameMode | None = None
    seed: int = 2026
    difficulty: str | None = None
    output: Path | None = None
    open_in_lazer: bool = False
    flow_engine: str = "auto"
    rhythm_engine: str = "legacy"
    modern_model: Path | None = None
    placement_model: Path | None = None
    rhythm_threshold: float | None = None
    target_density: float | None = None
    difficulty_tier: str | None = None
    target_stars: float | None = None
    enforce_star_target: bool = True
    flow_scale: float = 1.0


def configure_determinism(seed: int) -> None:
    """Seed supported runtimes without requiring ML dependencies at import time."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        with contextlib.suppress(AttributeError, RuntimeError):
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


@dataclass(frozen=True, slots=True)
class PresetSpec:
    name: str
    mode: GameMode
    rhythm_model: Path
    flow_dataset: Path | None
    pattern_dataset: Path | None
    hitsound_dataset: Path | None
    rhythm_params: tuple[Any, ...]
    gan_params: dict[str, Any]
    modding_params: dict[str, Any]

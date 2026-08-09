from __future__ import annotations


class OsumapperError(Exception):
    """Base error for user-actionable failures."""


class InputError(OsumapperError):
    """The provided beatmap, package, or audio input is invalid."""


class PackageSafetyError(InputError):
    """An archive failed safe extraction checks."""


class DependencyError(OsumapperError):
    """A required runtime dependency is unavailable or incompatible."""


class GenerationError(OsumapperError):
    """Beatmap generation failed."""

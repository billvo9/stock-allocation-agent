"""
The environment manifest of the prospective jobs, and the checks of a
running environment against the one a bundle was frozen in.

The manifest (read by forecasting.provenance.environment_manifest, written
into the bundle by freeze, so the bundle's manifest SHA-256 in the
registration record registers it):

    manifest_schema_version
    python          version, minor ("3.12"), implementation, executable_sha256
    platform        system, release, version, machine
    pins            "requirements.txt", "pyproject.toml" -> SHA-256 of the file
    timezone        key ("America/New_York"), source and the SHA-256 of the
                    zone file the recording window's conversions use
    distributions   name (PEP 503) -> version, content_sha256: the SHA-256 of
                    the installed files' actual contents, as RECORD lists
                    them, without the rows that depend on where or how the
                    distribution was installed (console scripts outside the
                    package, INSTALLER, REQUESTED, direct_url.json, RECORD),
                    so a file edited in place changes it and a reinstall into
                    another directory does not; for every distribution the
                    forecast and score jobs import, and every key one

`check(registered, running)` refuses (`environment_mismatch`) a different
Python minor version or implementation; a different version or content of a
key distribution (KEY_DISTRIBUTIONS: those whose arithmetic the forecasts and
scores use, always listed, imported lazily or not); and any difference in an
optional accelerator that changes pandas' arithmetic when present
(GUARDED_OPTIONAL), including its arrival. A bundle without a manifest is
refused (`environment_unregistered`). Every other difference (the patch
version, the executable, the platform, the pins, the timezone file, other
distributions) is reported with both values, not refused. The code identity is separate and unchanged: the registered paths
cover `src`, the job's script, `configs/prospective` and the pins.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.model_diagnostics.record import sha256_json

MANIFEST_SCHEMA_VERSION = "1.0"
KEY_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "pyarrow",
    "python-dateutil",
    "pyyaml",
    "scikit-learn",
    "scipy",
)
# Absent today; pandas uses them for reductions when installed, so their arrival refuses.
GUARDED_OPTIONAL = ("bottleneck", "numexpr", "numba")
PINS = ("requirements.txt", "pyproject.toml")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEYS = {"manifest_schema_version", "python", "platform", "pins", "timezone", "distributions"}
_PYTHON = {"version", "minor", "implementation", "executable_sha256"}
_PLATFORM = {"system", "release", "version", "machine"}


def _invalid(detail: str) -> ForecastRefused:
    return ForecastRefused("environment_invalid", detail)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{name} must be a non-empty string")
    return value


def _sha(value: object, name: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _invalid(f"{name} must be a SHA-256")


def validate_manifest(manifest: object) -> dict[str, object]:
    """A manifest's structure, refused (`environment_invalid`) unless exact."""

    if not isinstance(manifest, Mapping) or set(manifest) != _KEYS:
        raise _invalid(f"a manifest holds exactly {sorted(_KEYS)}")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise _invalid(f"manifest schema {manifest['manifest_schema_version']!r}")
    python, platform = manifest["python"], manifest["platform"]
    if not isinstance(python, Mapping) or set(python) != _PYTHON:
        raise _invalid(f"python must hold {sorted(_PYTHON)}")
    for name in ("version", "minor", "implementation"):
        _text(python[name], f"python.{name}")
    _sha(python["executable_sha256"], "python.executable_sha256")
    if not isinstance(platform, Mapping) or set(platform) != _PLATFORM:
        raise _invalid(f"platform must hold {sorted(_PLATFORM)}")
    for name in _PLATFORM:
        if not isinstance(platform[name], str):
            raise _invalid(f"platform.{name} must be a string")
    zone = manifest["timezone"]
    if not isinstance(zone, Mapping) or set(zone) != {"key", "source", "sha256"}:
        raise _invalid("timezone must hold key, source and sha256")
    _text(zone["key"], "timezone.key")
    _text(zone["source"], "timezone.source")
    _sha(zone["sha256"], "timezone.sha256")
    pins = manifest["pins"]
    if not isinstance(pins, Mapping) or set(pins) != set(PINS):
        raise _invalid(f"pins must name exactly {list(PINS)}")
    for name, digest in pins.items():
        _sha(digest, f"pins.{name}")
    distributions = manifest["distributions"]
    if not isinstance(distributions, Mapping) or not distributions:
        raise _invalid("distributions must be a non-empty mapping")
    for name, item in distributions.items():
        if not isinstance(item, Mapping) or set(item) != {"version", "content_sha256"}:
            raise _invalid(f"distributions.{name} must hold version and content_sha256")
        _text(item["version"], f"distributions.{name}.version")
        key = name in KEY_DISTRIBUTIONS  # a key distribution's content must be comparable
        _sha(item["content_sha256"], f"distributions.{name}.content_sha256", optional=not key)
    missing = sorted(set(KEY_DISTRIBUTIONS) - set(distributions))
    if missing:
        raise _invalid(f"the key distributions {missing} are missing")
    return dict(manifest)


def manifest_sha256(manifest: Mapping[str, object]) -> str:
    return sha256_json(manifest)


def _flat(manifest: Mapping[str, object]) -> dict[str, object]:
    """Every compared value, by dotted name; a distribution's absence is its own entry."""

    flat: dict[str, object] = {}
    for section in ("python", "platform", "pins", "timezone"):
        for name, value in manifest[section].items():
            flat[f"{section}.{name}"] = value
    for name, item in manifest["distributions"].items():
        flat[f"distributions.{name}"] = "installed"
        flat[f"distributions.{name}.version"] = item["version"]
        flat[f"distributions.{name}.content_sha256"] = item["content_sha256"]
    return flat


def _refusing(name: str) -> bool:
    if name in ("python.minor", "python.implementation"):
        return True
    parts = name.split(".")
    guarded = (*KEY_DISTRIBUTIONS, *GUARDED_OPTIONAL)
    return parts[0] == "distributions" and parts[1] in guarded


def check(
    registered: Mapping[str, object] | None, running: Mapping[str, object]
) -> dict[str, object]:
    """
    The running environment against the bundle's: refuses a Python minor or
    key-distribution change (and a bundle without a manifest); reports every
    other difference.
    """

    if registered is None:
        raise ForecastRefused(
            "environment_unregistered",
            "the bundle records no environment manifest: freeze again with this code",
        )
    frozen, now = validate_manifest(registered), validate_manifest(running)
    before, after = _flat(frozen), _flat(now)
    differ = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    refused = [name for name in differ if _refusing(name)]
    if refused:
        raise ForecastRefused(
            "environment_mismatch",
            f"{', '.join(refused)} differ from the bundle's environment: a new freeze and "
            "registration is needed",
        )
    # A distribution added or removed is reported once, by name; every other difference
    # keeps both values, so the record shows what changed.
    shown = {
        name: {"registered": before.get(name), "running": after.get(name)}
        for name in differ
        if not (name.startswith("distributions.") and name.rsplit(".", 1)[0] in differ)
    }
    return {
        "registered_sha256": manifest_sha256(frozen),
        "running_sha256": manifest_sha256(now),
        "differences": shown,
    }


__all__ = [
    "GUARDED_OPTIONAL",
    "KEY_DISTRIBUTIONS",
    "MANIFEST_SCHEMA_VERSION",
    "PINS",
    "check",
    "manifest_sha256",
    "validate_manifest",
]

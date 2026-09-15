"""Resolve immutable release bundles in source and installed distributions."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DEFAULT_RELEASE = "2026-08-20"
_PACKAGED_DIRECTORY = "_release_data"


def resolve_release_bundle(
    bundle: Path | str | None = None, *, release: str = DEFAULT_RELEASE
) -> Path:
    """Return an explicit, packaged, or source-checkout release directory."""
    if bundle is not None:
        resolved = Path(bundle).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"release bundle directory does not exist: {resolved}")
        return resolved

    packaged = (
        resources.files("tools.densevideo")
        .joinpath(_PACKAGED_DIRECTORY)
        .joinpath(release)
    )
    if packaged.is_dir():
        try:
            return Path(packaged).resolve()
        except TypeError as error:
            raise RuntimeError(
                "DIVE-Bench release resources require an unpacked wheel installation"
            ) from error

    source = Path(__file__).resolve().parents[2] / "release" / release
    if source.is_dir():
        return source.resolve()
    raise FileNotFoundError(
        f"release {release} is absent from both package resources and source checkout"
    )

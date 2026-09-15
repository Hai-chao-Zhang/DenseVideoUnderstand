"""Copy canonical release bundles into wheels without duplicating source files."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.errors import SetupError

_RELEASE_DIRECTORY = re.compile(r"\d{4}-\d{2}-\d{2}")
_ALLOWED_SUFFIXES = {".csv", ".js", ".json", ".md"}
_NUMERIC_REPORTS_ARCHIVE = "numeric_reports.json.gz"


class BuildPyWithReleaseData(build_py):
    """Stage every dated release directory under ``tools.densevideo``."""

    def run(self) -> None:
        super().run()
        project_root = Path(__file__).resolve().parents[1]
        source_root = project_root / "release"
        destination_root = (
            Path(self.build_lib) / "tools" / "densevideo" / "_release_data"
        )
        if not source_root.is_dir():
            raise SetupError(f"missing canonical release directory: {source_root}")

        releases = sorted(path for path in source_root.iterdir() if path.is_dir())
        if not releases:
            raise SetupError("canonical release directory contains no dated bundles")
        for release in releases:
            if not _RELEASE_DIRECTORY.fullmatch(release.name):
                raise SetupError(f"unexpected release directory name: {release.name}")

        if destination_root.exists():
            shutil.rmtree(destination_root)
        self.mkpath(str(destination_root))
        self._release_outputs: list[str] = []
        for release in releases:
            for source in sorted(release.rglob("*")):
                if source.is_symlink():
                    raise SetupError(f"release bundles cannot contain symlinks: {source}")
                if not source.is_file():
                    continue
                relative = source.relative_to(source_root)
                compressed_numeric_report = (
                    source.name == _NUMERIC_REPORTS_ARCHIVE and source.parent == release
                )
                if (source.suffix.lower() not in _ALLOWED_SUFFIXES
                        and not compressed_numeric_report) or any(
                    part.startswith(".") for part in relative.parts
                ):
                    raise SetupError(f"unexpected release artifact: {relative}")
                destination = destination_root / relative
                self.mkpath(str(destination.parent))
                copied, _ = self.copy_file(str(source), str(destination))
                self._release_outputs.append(copied)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        return super().get_outputs(include_bytecode) + getattr(
            self, "_release_outputs", []
        )

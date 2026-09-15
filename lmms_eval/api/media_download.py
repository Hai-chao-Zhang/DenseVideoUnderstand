"""Constrained archive extraction and shell-free media download arguments.

Dataset archives may contain only ordinary files and directories. Validate the
whole member list before extracting anything; never restore archive ownership,
permissions, links, or special files. This works on Python 3.10 and 3.11 without
depending on newer ``tarfile`` extraction-filter defaults.
"""

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def _member_target(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in path.parts
    ):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    target = root.joinpath(*path.parts)
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Archive member traverses an existing symlink: {name!r}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Archive member escapes the destination: {name!r}")
    return target


def _check_targets(root: Path, members: list) -> None:
    """Reject existing special files and inconsistent file/directory layouts."""
    file_targets = {target for _, target, is_dir in members if not is_dir}
    for _, target, is_dir in members:
        if not is_dir and target == root:
            raise ValueError("An archive file cannot replace its destination directory")
        if is_dir and target in file_targets:
            raise ValueError(f"Archive path is both a directory and a file: {target}")
        if any(parent in file_targets for parent in target.parents if parent != root):
            raise ValueError(f"Archive file is used as a directory: {target}")
        current = target
        while current != root:
            if current.exists():
                mode = current.stat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"Archive destination contains a special file: {current}")
                expected_dir = current != target or is_dir
                if expected_dir != stat.S_ISDIR(mode):
                    raise ValueError(
                        f"Archive destination has a file/directory conflict: {current}"
                    )
                if stat.S_ISREG(mode) and current.stat().st_nlink > 1:
                    raise ValueError(f"Archive destination is an existing hard link: {current}")
            current = current.parent


def safe_extract_tar(archive_path, destination) -> None:
    root = Path(destination).resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = []
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported TAR member type: {member.name!r}")
            members.append((member, _member_target(root, member.name), member.isdir()))
        _check_targets(root, members)
        root.mkdir(parents=True, exist_ok=True)
        for member, target, is_dir in members:
            # Recheck paths after directory creation and before opening a file.
            _member_target(root, member.name)
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def safe_extract_zip(archive_path, destination, *, skip_existing=True) -> None:
    root = Path(destination).resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = []
        for member in archive.infolist():
            file_type = stat.S_IFMT(member.external_attr >> 16)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported ZIP member type: {member.filename!r}")
            is_dir = member.is_dir()
            if file_type and (file_type == stat.S_IFDIR) != is_dir:
                raise ValueError(f"Inconsistent ZIP member type: {member.filename!r}")
            members.append((member, _member_target(root, member.filename), is_dir))
        _check_targets(root, members)
        root.mkdir(parents=True, exist_ok=True)
        for member, target, is_dir in members:
            _member_target(root, member.filename)
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            elif not (skip_existing and target.exists()):
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def youtube_download_command(video_id, target_path) -> list[str]:
    """Return literal argv; dataset text must never be interpreted by a shell."""
    video_id = str(video_id)
    # The caller also uses the ID as a filename; forbid path-form identifiers
    # before yt-dlp can write its output. Shell metacharacters remain literal.
    if (
        video_id in ("", ".", "..")
        or any(character in video_id for character in ("/", "\\", "\0"))
        or PureWindowsPath(video_id).drive
    ):
        raise ValueError(f"Unsafe YouTube video identifier: {video_id!r}")
    return [
        "yt-dlp",
        "-o",
        str(target_path),
        "-f",
        "mp4",
        "--",
        "https://www.youtube.com/watch?v=" + video_id,
    ]

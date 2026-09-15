"""Archive safety checks run without model, network, or multimedia dependencies."""

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from lmms_eval.api.media_download import (
    safe_extract_tar,
    safe_extract_zip,
    youtube_download_command,
)


@pytest.mark.parametrize(
    "video_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "/absolute",
        "nested/video",
        r"nested\video",
        r"C:\video",
        "C:video",
        "invalid\0name",
    ],
)
def test_youtube_path_form_ids_are_rejected(video_id, tmp_path):
    with pytest.raises(ValueError):
        youtube_download_command(video_id, tmp_path / "video.mp4")


def _write_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, kind, payload in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = str(payload)
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                archive.addfile(member)


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, mode, payload in members:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = mode << 16
            if stat.S_ISDIR(mode):
                member.external_attr |= 0x10
            archive.writestr(member, payload)


def _write_normal_archive(path, archive_kind, names):
    if archive_kind == "tar":
        _write_tar(path, [(name, tarfile.REGTYPE, b"new contents") for name in names])
    else:
        _write_zip(path, [(name, stat.S_IFREG | 0o644, b"new contents") for name in names])


def _extract(path, destination, archive_kind):
    if archive_kind == "tar":
        safe_extract_tar(path, destination)
    else:
        safe_extract_zip(path, destination)


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_normal_files_and_directories_extract(tmp_path, archive_kind):
    archive = tmp_path / f"normal.{archive_kind}"
    destination = tmp_path / "extracted"
    if archive_kind == "tar":
        _write_tar(
            archive,
            [
                ("nested", tarfile.DIRTYPE, b""),
                ("nested/video.txt", tarfile.REGTYPE, b"video bytes"),
                ("empty.txt", tarfile.REGTYPE, b""),
                ("unicode-视频.txt", tarfile.REGTYPE, b"unicode name"),
            ],
        )
    else:
        _write_zip(
            archive,
            [
                ("nested/", stat.S_IFDIR | 0o755, b""),
                ("nested/video.txt", stat.S_IFREG | 0o644, b"video bytes"),
                ("empty.txt", stat.S_IFREG | 0o644, b""),
                ("unicode-视频.txt", stat.S_IFREG | 0o644, b"unicode name"),
            ],
        )

    _extract(archive, destination, archive_kind)

    assert (destination / "nested").is_dir()
    assert (destination / "nested/video.txt").read_bytes() == b"video bytes"
    assert (destination / "empty.txt").read_bytes() == b""
    assert (destination / "unicode-视频.txt").read_bytes() == b"unicode name"


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../outside.txt",
        "nested/../../outside.txt",
        "/absolute-outside.txt",
        r"C:\outside.txt",
        "C:/outside.txt",
        r"..\outside.txt",
        r"nested\..\..\outside.txt",
        r"\outside.txt",
        "//server/share/outside.txt",
    ],
)
def test_unsafe_path_rejects_entire_archive_before_extraction(tmp_path, archive_kind, unsafe_name):
    archive = tmp_path / f"unsafe.{archive_kind}"
    destination = tmp_path / "extracted"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    _write_normal_archive(archive, archive_kind, ["valid-first.txt", unsafe_name])

    with pytest.raises(ValueError):
        _extract(archive, destination, archive_kind)

    assert not (destination / "valid-first.txt").exists()
    assert outside.read_bytes() == b"untouched"
    assert not destination.exists() or not list(destination.iterdir())


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_absolute_destination_outside_root_is_not_written(tmp_path, archive_kind):
    archive = tmp_path / f"absolute.{archive_kind}"
    destination = tmp_path / "extracted"
    outside = tmp_path / "outside.txt"
    _write_normal_archive(archive, archive_kind, ["valid-first.txt", str(outside)])

    with pytest.raises(ValueError):
        _extract(archive, destination, archive_kind)

    assert not outside.exists()
    assert not (destination / "valid-first.txt").exists()


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ],
)
def test_tar_special_member_rejects_archive_before_extraction(tmp_path, member_type):
    archive = tmp_path / "special.tar"
    destination = tmp_path / "extracted"
    _write_tar(
        archive,
        [
            ("valid-first.txt", tarfile.REGTYPE, b"valid data"),
            ("special", member_type, "valid-first.txt"),
        ],
    )

    with pytest.raises(ValueError):
        safe_extract_tar(archive, destination)

    assert not (destination / "valid-first.txt").exists()
    assert not (destination / "special").exists()


@pytest.mark.parametrize(
    "member_type",
    [
        stat.S_IFLNK,
        stat.S_IFIFO,
        stat.S_IFCHR,
        stat.S_IFBLK,
        stat.S_IFSOCK,
    ],
)
def test_zip_special_member_rejects_archive_before_extraction(tmp_path, member_type):
    archive = tmp_path / "special.zip"
    destination = tmp_path / "extracted"
    _write_zip(
        archive,
        [
            ("valid-first.txt", stat.S_IFREG | 0o644, b"valid data"),
            ("special", member_type | 0o644, b"valid-first.txt"),
        ],
    )

    with pytest.raises(ValueError):
        safe_extract_zip(archive, destination)

    assert not (destination / "valid-first.txt").exists()
    assert not (destination / "special").exists()


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
@pytest.mark.parametrize("link_target_inside", [False, True])
@pytest.mark.parametrize("link_is_directory", [False, True])
def test_preexisting_symlink_rejects_archive_before_extraction(
    tmp_path, archive_kind, link_target_inside, link_is_directory
):
    archive = tmp_path / f"symlink.{archive_kind}"
    destination = tmp_path / "extracted"
    destination.mkdir()
    target_parent = destination if link_target_inside else tmp_path
    if link_is_directory:
        target = target_parent / "real-directory"
        target.mkdir()
        target_file = target / "video.txt"
        target_file.write_bytes(b"untouched")
        (destination / "linked").symlink_to(target, target_is_directory=True)
        member_name = "linked/video.txt"
    else:
        target_file = target_parent / "real-file.txt"
        target_file.write_bytes(b"untouched")
        (destination / "linked").symlink_to(target_file)
        member_name = "linked"
    _write_normal_archive(archive, archive_kind, ["valid-first.txt", member_name])

    with pytest.raises(ValueError):
        _extract(archive, destination, archive_kind)

    assert not (destination / "valid-first.txt").exists()
    assert target_file.read_bytes() == b"untouched"
    assert (destination / "linked").is_symlink()


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_explicit_symlink_destination_root_is_supported(tmp_path, archive_kind):
    archive = tmp_path / f"root-link.{archive_kind}"
    destination = tmp_path / "extracted"
    actual_directory = tmp_path / "actual-directory"
    actual_directory.mkdir()
    destination.symlink_to(actual_directory, target_is_directory=True)
    _write_normal_archive(archive, archive_kind, ["video.txt"])

    _extract(archive, destination, archive_kind)

    assert (actual_directory / "video.txt").read_bytes() == b"new contents"
    assert destination.is_symlink()


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_preexisting_dangling_symlink_is_rejected(tmp_path, archive_kind):
    archive = tmp_path / f"dangling.{archive_kind}"
    destination = tmp_path / "extracted"
    destination.mkdir()
    missing_target = tmp_path / "missing-target.txt"
    (destination / "linked").symlink_to(missing_target)
    _write_normal_archive(archive, archive_kind, ["valid-first.txt", "linked"])

    with pytest.raises(ValueError):
        _extract(archive, destination, archive_kind)

    assert not (destination / "valid-first.txt").exists()
    assert not missing_target.exists()
    assert (destination / "linked").is_symlink()


@pytest.mark.parametrize("archive_kind", ["tar", "zip"])
def test_preexisting_hardlinked_file_is_rejected(tmp_path, archive_kind):
    archive = tmp_path / f"hardlink.{archive_kind}"
    destination = tmp_path / "extracted"
    destination.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    (destination / "linked").hardlink_to(outside)
    _write_normal_archive(archive, archive_kind, ["valid-first.txt", "linked"])

    with pytest.raises(ValueError):
        _extract(archive, destination, archive_kind)

    assert not (destination / "valid-first.txt").exists()
    assert outside.read_bytes() == b"untouched"
    assert (destination / "linked").read_bytes() == b"untouched"


def test_tar_overwrites_existing_regular_file(tmp_path):
    archive = tmp_path / "normal.tar"
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "video.txt").write_bytes(b"old contents")
    _write_normal_archive(archive, "tar", ["video.txt"])

    safe_extract_tar(archive, destination)

    assert (destination / "video.txt").read_bytes() == b"new contents"


@pytest.mark.parametrize("skip_existing", [None, True, False])
def test_zip_existing_file_policy(tmp_path, skip_existing):
    archive = tmp_path / "normal.zip"
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "video.txt").write_bytes(b"old contents")
    _write_normal_archive(archive, "zip", ["video.txt", "other.txt"])

    if skip_existing is None:
        safe_extract_zip(archive, destination)
    else:
        safe_extract_zip(archive, destination, skip_existing=skip_existing)

    expected = b"new contents" if skip_existing is False else b"old contents"
    assert (destination / "video.txt").read_bytes() == expected
    assert (destination / "other.txt").read_bytes() == b"new contents"


def test_zip_without_unix_file_type_extracts(tmp_path):
    archive = tmp_path / "portable.zip"
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("video.txt", b"portable contents")

    safe_extract_zip(archive, destination)

    assert (destination / "video.txt").read_bytes() == b"portable contents"


@pytest.mark.parametrize(
    "video_id",
    [
        "ordinary-video-id",
        "name; touch unexpected-file",
        "$(touch unexpected-file)",
        "`touch unexpected-file`",
        "--exec=malicious-command",
        "quote'\"&|\nnext-line",
        12345,
    ],
)
def test_youtube_command_keeps_untrusted_values_in_literal_argv(video_id):
    target = Path("videos/output; $(touch unexpected-file) ' quoted.mp4")

    command = youtube_download_command(video_id, target)

    assert isinstance(command, list)
    assert command == [
        "yt-dlp",
        "-o",
        str(target),
        "-f",
        "mp4",
        "--",
        "https://www.youtube.com/watch?v=" + str(video_id),
    ]
    assert all(isinstance(argument, str) for argument in command)

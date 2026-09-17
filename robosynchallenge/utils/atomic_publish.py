"""Linux atomic publication primitives that never replace an existing path."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def path_exists(path: str | os.PathLike[str]) -> bool:
    """Return true for every directory entry, including dangling symlinks."""
    candidate = Path(path)
    return candidate.exists() or candidate.is_symlink()


def publish_file_no_clobber(
    temporary_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> Path:
    """Atomically link a same-filesystem regular temp file into a new name."""
    temporary = Path(temporary_path)
    destination = Path(destination_path)
    os.link(temporary, destination)
    temporary.unlink()
    return destination


def rename_path_no_clobber(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> Path:
    """Atomically rename a file or directory without replacing any entry."""
    source = Path(source_path)
    destination = Path(destination_path)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for no-clobber rename")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))
    return destination

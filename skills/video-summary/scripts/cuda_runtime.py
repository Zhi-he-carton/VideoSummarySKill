"""Process-local CUDA DLL discovery for the Skill's Windows runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_DLL_DIRECTORY_HANDLES: list[Any] = []
_CONFIGURED_DIRECTORIES: set[str] = set()


def _windows_cuda_bin_directories() -> list[Path]:
    candidates: list[Path] = []
    for name, value in os.environ.items():
        if name.casefold().startswith("cuda_path") and value:
            candidates.append(Path(value) / "bin")

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    toolkit_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_root.is_dir():
        toolkit_bins = list(toolkit_root.glob("v*/bin"))

        def toolkit_version(path: Path) -> tuple[int, ...]:
            try:
                return tuple(int(part) for part in path.parent.name.removeprefix("v").split("."))
            except ValueError:
                return ()

        candidates.extend(sorted(toolkit_bins, key=toolkit_version, reverse=True))

    for item in os.environ.get("PATH", "").split(os.pathsep):
        if not item:
            continue
        path_entry = Path(item)
        try:
            has_cuda_runtime = any(path_entry.glob("cublas64_*.dll"))
        except OSError:
            has_cuda_runtime = False
        if has_cuda_runtime:
            candidates.append(path_entry)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key not in seen and resolved.is_dir():
            seen.add(key)
            unique.append(resolved)
    return unique


def configure_cuda_runtime() -> list[str]:
    """Expose installed CUDA DLL directories to this process without changing system PATH."""
    if sys.platform != "win32":
        return []

    directories = _windows_cuda_bin_directories()
    prefixes = [str(directory) for directory in directories]
    if prefixes:
        os.environ["PATH"] = os.pathsep.join([*prefixes, os.environ.get("PATH", "")])

    add_directory = getattr(os, "add_dll_directory", None)
    if add_directory is not None:
        for directory in directories:
            key = str(directory).casefold()
            if key in _CONFIGURED_DIRECTORIES:
                continue
            try:
                handle = add_directory(str(directory))
            except OSError:
                continue
            _DLL_DIRECTORY_HANDLES.append(handle)
            _CONFIGURED_DIRECTORIES.add(key)
    return prefixes

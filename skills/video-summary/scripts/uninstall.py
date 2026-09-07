#!/usr/bin/env python3
"""Remove dependency state owned exclusively by the video-summary Skill.

Run this script outside the Skill environment so Windows can remove the active
virtual environment without file-lock failures. The documented ``uv run
--no-project --no-cache`` invocation provides that isolation on both Windows
and macOS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FILESYSTEM = 5

SKILL_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIRECTORIES = (".venv", ".uv-cache", ".uv-python")
MODEL_REPOSITORIES = (
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-large-v3",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove video-summary's isolated Python dependencies and local runtime state."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Perform the removal. Without this flag, only show the removal plan.",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also remove this Skill's three faster-whisper model directories.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON result object.")
    return parser


def _huggingface_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _target(kind: str, path: Path) -> dict[str, Any]:
    return {"kind": kind, "path": str(path), "present": os.path.lexists(path)}


def _targets(
    skill_dir: Path,
    *,
    include_models: bool,
    model_cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_skill_dir = skill_dir.expanduser().resolve()
    targets = [_target("runtime", resolved_skill_dir / name) for name in RUNTIME_DIRECTORIES]
    if include_models:
        cache_root = (model_cache_root or _huggingface_cache_root()).expanduser().resolve()
        targets.extend(
            _target("model", cache_root / ("models--" + repository.replace("/", "--")))
            for repository in MODEL_REPOSITORIES
        )
    return targets


def _make_writable(function: Any, path: str, error: BaseException) -> None:
    """Retry removal of read-only files commonly found in Windows environments."""
    try:
        os.chmod(path, stat.S_IRWXU)
        function(path)
    except OSError:
        raise error


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, onexc=_make_writable)


def _clean_uv_cache(path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise OSError("uv was not found on PATH; the Skill cache was not changed.")
    completed = subprocess.run(
        [uv, "cache", "clean", "--cache-dir", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise OSError(detail[-1] if detail else "uv could not clean the Skill cache.")
    if os.path.lexists(path):
        _remove_path(path)


def uninstall(
    skill_dir: Path = SKILL_DIR,
    *,
    include_models: bool = False,
    dry_run: bool = True,
    model_cache_root: Path | None = None,
    executable: Path | None = None,
    cache_cleaner: Callable[[Path], None] = _clean_uv_cache,
) -> dict[str, Any]:
    """Preview or remove only runtime directories owned by this Skill."""
    resolved_skill_dir = skill_dir.expanduser().resolve()
    if (
        not (resolved_skill_dir / "SKILL.md").is_file()
        or not (resolved_skill_dir / "pyproject.toml").is_file()
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "error",
            "skill_dir": str(resolved_skill_dir),
            "dry_run": dry_run,
            "targets": [],
            "error": {
                "code": "invalid_skill_directory",
                "message": "The expected SKILL.md and pyproject.toml files were not found.",
            },
        }

    targets = _targets(
        resolved_skill_dir,
        include_models=include_models,
        model_cache_root=model_cache_root,
    )
    active_executable = (executable or Path(sys.executable)).resolve()
    active_targets = [
        item["path"]
        for item in targets
        if item["present"] and _is_relative_to(active_executable, Path(item["path"]).resolve())
    ]
    if active_targets and not dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "error",
            "skill_dir": str(resolved_skill_dir),
            "dry_run": False,
            "targets": targets,
            "error": {
                "code": "active_runtime",
                "message": (
                    "The uninstaller is running from a directory it must remove. "
                    "Use the documented uv run --no-project --no-cache command."
                ),
                "paths": active_targets,
            },
        }

    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "status": "preview",
            "skill_dir": str(resolved_skill_dir),
            "dry_run": True,
            "targets": targets,
            "preserved": _preserved(include_models),
            "error": None,
        }

    failures: list[dict[str, str]] = []
    for item in targets:
        path = Path(item["path"])
        item["removed"] = False
        if not item["present"]:
            continue
        try:
            if path.name == ".uv-cache":
                cache_cleaner(path)
            else:
                _remove_path(path)
        except OSError as exc:
            failures.append({"path": str(path), "message": str(exc)})
        else:
            item["removed"] = not os.path.lexists(path)

    removed_any = any(item.get("removed") for item in targets)
    status = "partial" if failures else ("uninstalled" if removed_any else "already_clean")
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": not failures,
        "status": status,
        "skill_dir": str(resolved_skill_dir),
        "dry_run": False,
        "targets": targets,
        "preserved": _preserved(include_models),
        "error": (
            {
                "code": "removal_failed",
                "message": "Some paths could not be removed.",
                "failures": failures,
            }
            if failures
            else None
        ),
    }


def _preserved(include_models: bool) -> list[str]:
    preserved = [
        "uv",
        "FFmpeg/FFprobe",
        "NVIDIA CUDA and GPU drivers",
        "source files and generated transcripts",
    ]
    if not include_models:
        preserved.append("faster-whisper model cache (use --include-models to remove it)")
    return preserved


def _print_human(result: dict[str, Any]) -> None:
    print(f"video-summary uninstall: {result['status'].upper()}")
    if result.get("error"):
        print(f"ERROR: {result['error']['message']}")
    for item in result.get("targets", []):
        if result["dry_run"]:
            state = "remove" if item["present"] else "absent"
        else:
            state = "removed" if item.get("removed") else "absent or unchanged"
        print(f"[{state}] {item['kind']}: {item['path']}")
    if result.get("preserved"):
        print("Preserved: " + ", ".join(result["preserved"]))
    if result["status"] == "preview":
        print("Rerun with --yes to perform the removal.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = uninstall(include_models=args.include_models, dry_run=not args.yes)
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        _print_human(result)
    if result["ok"]:
        return EXIT_OK
    return EXIT_USAGE if result["error"]["code"] == "active_runtime" else EXIT_FILESYSTEM


if __name__ == "__main__":
    raise SystemExit(main())

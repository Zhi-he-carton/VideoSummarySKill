#!/usr/bin/env python3
"""Read-only dependency and capability checks for the video-summary Skill."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .cuda_runtime import configure_cuda_runtime
except ImportError:  # Direct ``python scripts/doctor.py`` compatibility.
    from cuda_runtime import configure_cuda_runtime


SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_REQUIREMENTS = 3

PROFILE_MODELS = {
    "fast": "Systran/faster-whisper-small",
    "balanced": "Systran/faster-whisper-medium",
    "accurate": "Systran/faster-whisper-large-v3",
}
SKILL_DIR = Path(__file__).resolve().parents[1]
UV_CACHE_DIR = SKILL_DIR / ".uv-cache"
UV_SYNC_COMMAND = [
    "uv",
    "sync",
    "--project",
    str(SKILL_DIR),
    "--cache-dir",
    str(UV_CACHE_DIR),
    "--locked",
    "--no-dev",
    "--python",
    "3.12",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check video-summary dependencies without installing or changing anything."
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON result object.")
    parser.add_argument(
        "--require",
        action="append",
        choices=("fetch", "cpu", "cuda"),
        dest="requirements",
        help="Capability that must be ready; may be repeated (default: fetch and cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optionally check an intended output directory without creating it.",
    )
    return parser


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_import(module: str) -> tuple[bool, str | None]:
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    command = [
        sys.executable,
        "-c",
        "import importlib,sys; importlib.import_module(sys.argv[1])",
        module,
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "The isolated import probe could not complete."
    if completed.returncode == 0:
        return True, None
    detail = completed.stderr.strip().splitlines()
    return False, detail[-1] if detail else "The module import failed."


def _probe_cuda_devices() -> int:
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    command = [
        sys.executable,
        "-c",
        "import ctranslate2; print(int(ctranslate2.get_cuda_device_count()))",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if completed.returncode != 0:
        return 0
    try:
        return max(0, int(completed.stdout.strip()))
    except ValueError:
        return 0


def _check(
    name: str,
    ok: bool,
    message: str,
    *,
    required: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "required": required,
        "message": message,
        "details": details or {},
    }


def _executable_check(name: str, *, required: bool) -> tuple[dict[str, Any], str | None]:
    path = shutil.which(name)
    return (
        _check(
            name,
            path is not None,
            f"{name} is available." if path else f"{name} was not found on PATH.",
            required=required,
            details={"path": path},
        ),
        path,
    )


def _package_check(
    distribution: str,
    module: str,
    *,
    required: bool,
) -> dict[str, Any]:
    version = _version(distribution)
    imported, error = _probe_import(module) if version is not None else (False, None)
    if version is None:
        message = f"{distribution} is not installed in the Skill environment."
    elif not imported:
        message = f"{distribution} {version} is installed but cannot be imported."
    else:
        message = f"{distribution} {version} is available."
    details: dict[str, Any] = {"version": version, "module": module}
    if error:
        details["import_error"] = error
    return _check(distribution, imported, message, required=required, details=details)


def _cuda_library_names(ctranslate2_version: str | None) -> list[str]:
    if sys.platform == "darwin":
        return []
    try:
        parts = tuple(int(part) for part in (ctranslate2_version or "").split("."))
    except (TypeError, ValueError):
        parts = (4, 8, 1)

    major = parts[0] if parts else 4
    minor = parts[1] if len(parts) > 1 else 0
    patch = parts[2] if len(parts) > 2 else 0

    cuda_major = 11 if major < 4 else 12
    cudnn_major = 8 if major < 4 or (major == 4 and minor < 5) else 9
    if sys.platform == "win32":
        cublas = f"cublas64_{cuda_major}.dll"
        if (major, minor, patch) >= (4, 6, 3):
            return [cublas]
        cudnn = "cudnn_ops_infer64_8.dll" if cudnn_major == 8 else "cudnn_ops64_9.dll"
        return [cublas, cudnn]
    cublas = f"libcublas.so.{cuda_major}"
    if (major, minor, patch) >= (4, 6, 3):
        return [cublas]
    return [cublas, f"libcudnn_ops.so.{cudnn_major}"]


def _installed_cuda_toolkit_version(directories: Sequence[str]) -> tuple[int, int] | None:
    versions: list[tuple[int, int]] = []
    for directory in directories:
        name = Path(directory).parent.name.removeprefix("v")
        try:
            major, minor, *_rest = (int(part) for part in name.split("."))
        except (TypeError, ValueError):
            continue
        versions.append((major, minor))
    return max(versions, default=None)


def _loadable_library(name: str) -> tuple[bool, str | None]:
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(name)
        else:
            ctypes.CDLL(name)
    except OSError as exc:
        return False, str(exc)
    return True, None


def _gpu_check(ctranslate2_version: str | None) -> tuple[dict[str, Any], bool]:
    if sys.platform == "darwin":
        check = _check(
            "cuda-runtime",
            False,
            "CUDA transcription is not supported on macOS; CPU transcription remains available.",
            required=False,
            details={"platform": sys.platform, "cuda_devices": 0, "libraries": []},
        )
        return check, False

    configured_directories = configure_cuda_runtime()
    nvidia_smi = shutil.which("nvidia-smi")
    cuda_devices = _probe_cuda_devices()
    libraries: list[dict[str, Any]] = []
    for name in _cuda_library_names(ctranslate2_version):
        loaded, error = _loadable_library(name)
        item: dict[str, Any] = {"name": name, "loadable": loaded}
        if error:
            item["error"] = error
        libraries.append(item)
    installed_toolkit = _installed_cuda_toolkit_version(configured_directories)
    runtime_ready = bool(nvidia_smi and cuda_devices > 0 and all(x["loadable"] for x in libraries))
    missing = [item["name"] for item in libraries if not item["loadable"]]
    if runtime_ready:
        toolkit_note = (
            f" using detected Toolkit {installed_toolkit[0]}.{installed_toolkit[1]}"
            if installed_toolkit is not None
            else ""
        )
        message = f"CUDA is ready with {cuda_devices} visible device(s){toolkit_note}."
    elif missing:
        message = (
            "CUDA hardware is visible, but required runtime libraries are missing: "
            + ", ".join(missing)
        )
    elif not nvidia_smi or cuda_devices == 0:
        message = "No CUDA device is available to CTranslate2."
    else:
        message = "The CUDA runtime is incomplete."
    check = _check(
        "cuda-runtime",
        runtime_ready,
        message,
        required=False,
        details={
            "nvidia_smi": nvidia_smi,
            "cuda_devices": cuda_devices,
            "ctranslate2_version": ctranslate2_version,
            "installed_cuda_toolkit": (
                ".".join(str(part) for part in installed_toolkit)
                if installed_toolkit is not None
                else None
            ),
            "configured_dll_directories": configured_directories,
            "libraries": libraries,
        },
    )
    return check, runtime_ready


def _huggingface_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cache() -> dict[str, Any]:
    root = _huggingface_cache_root().resolve()
    profiles: dict[str, Any] = {}
    for profile, repository in PROFILE_MODELS.items():
        folder = root / ("models--" + repository.replace("/", "--"))
        try:
            cached = folder.is_dir() and any(folder.iterdir())
        except OSError:
            cached = False
        profiles[profile] = {
            "repository": repository,
            "cached": cached,
            "path": str(folder),
        }
    return {"root": str(root), "profiles": profiles}


def _output_check(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    candidate = resolved
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    ok = candidate.is_dir() and os.access(candidate, os.W_OK)
    return _check(
        "output-directory",
        ok,
        "The output directory is writable." if ok else "The output directory is not writable.",
        required=True,
        details={"requested": str(resolved), "checked_parent": str(candidate)},
    )


def _display_command(command: Sequence[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _action(
    action_id: str,
    *,
    kind: str,
    program: str | None,
    manager: str,
    automatic: bool,
    prompt: str,
    command: Sequence[str] | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    command_list = list(command) if command is not None else None
    return {
        "id": action_id,
        "kind": kind,
        "program": program,
        "manager": manager,
        "automatic": automatic,
        "requires_user_action": not automatic,
        "prompt": prompt,
        "command": command_list,
        "display_command": _display_command(command_list) if command_list else None,
        "after": after,
    }


def _remediation_actions(checks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = {check["name"]: check for check in checks if not check["ok"]}
    actions: list[dict[str, Any]] = []

    uv_managed = {"python", "yt-dlp", "av", "faster-whisper", "ctranslate2"}
    uv_failures = sorted(uv_managed.intersection(failed))
    if uv_failures and "uv" not in failed:
        actions.append(
            _action(
                "sync-locked-environment",
                kind="repair",
                program="video-summary Python environment",
                manager="uv",
                automatic=True,
                prompt=(
                    "The Skill-local Python environment is incomplete or incompatible "
                    f"({', '.join(uv_failures)}). Synchronize the committed lock file once."
                ),
                command=UV_SYNC_COMMAND,
                after="Rerun doctor once with the same --require and --output-dir arguments.",
            )
        )

    if "uv" in failed:
        if sys.platform == "win32":
            command = ["winget", "install", "--id", "astral-sh.uv", "--exact"]
        elif sys.platform == "darwin":
            command = ["brew", "install", "uv"]
        else:
            command = None
        actions.append(
            _action(
                "install-uv",
                kind="install",
                program="uv",
                manager="system",
                automatic=False,
                prompt=(
                    "uv is required to create and run the Skill-local locked Python environment. "
                    "Ask the user to install it, then reopen the shell."
                ),
                command=command,
                after="Initialize the Skill environment from uv.lock, then rerun doctor.",
            )
        )

    if "ffmpeg" in failed or "ffprobe" in failed:
        if sys.platform == "win32":
            command = [
                "winget",
                "install",
                "--id",
                "Gyan.FFmpeg",
                "--exact",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
        elif sys.platform == "darwin":
            command = ["brew", "install", "ffmpeg"]
        else:
            command = None
        actions.append(
            _action(
                "install-ffmpeg",
                kind="install",
                program="FFmpeg (including FFprobe)",
                manager="system",
                automatic=False,
                prompt=(
                    "FFmpeg and FFprobe are required to download, convert, and validate audio. "
                    "Ask the user to install them or add an existing installation to PATH."
                ),
                command=command,
                after="Open a new shell and rerun doctor.",
            )
        )

    if "cuda-runtime" in failed and sys.platform == "darwin":
        actions.append(
            _action(
                "use-cpu-on-macos",
                kind="choose",
                program="CPU transcription on macOS",
                manager="runtime",
                automatic=False,
                prompt=(
                    "CUDA transcription is unavailable on macOS. Use --device cpu or "
                    "--device auto; do not install NVIDIA runtime libraries."
                ),
                after="Rerun doctor with --require cpu.",
            )
        )
    elif "cuda-runtime" in failed:
        missing = [
            item["name"]
            for item in failed["cuda-runtime"]["details"].get("libraries", [])
            if not item.get("loadable")
        ]
        suffix = f" Missing libraries: {', '.join(missing)}." if missing else ""
        diagnosis = failed["cuda-runtime"].get("message", "")
        actions.append(
            _action(
                "install-cuda-runtime",
                kind="install",
                program="NVIDIA CUDA runtime libraries required by CTranslate2",
                manager="system",
                automatic=False,
                prompt=(
                    "Install official NVIDIA runtime libraries for the host architecture required "
                    "by the locked CTranslate2 version; do not use third-party runtime bundles "
                    f"without user approval. {diagnosis}{suffix}"
                ),
                after="Rerun doctor with --require cuda before starting GPU transcription.",
            )
        )

    if "output-directory" in failed:
        actions.append(
            _action(
                "choose-writable-output",
                kind="choose",
                program=None,
                manager="filesystem",
                automatic=False,
                prompt="Choose a writable output directory; do not change permissions automatically.",
                after="Rerun doctor with the replacement --output-dir.",
            )
        )
    return actions


def inspect_environment(
    *,
    requirements: Sequence[str] = ("fetch", "cpu"),
    output_dir: Path | None = None,
) -> dict[str, Any]:
    required = set(requirements)
    checks: list[dict[str, Any]] = []

    python_ok = (3, 12) <= sys.version_info[:2] < (3, 13)
    checks.append(
        _check(
            "python",
            python_ok,
            f"Python {sys.version.split()[0]} is active.",
            required=True,
            details={"executable": sys.executable, "required_range": ">=3.12,<3.13"},
        )
    )
    uv_check, _uv_path = _executable_check("uv", required=True)
    checks.append(uv_check)

    yt_dlp = _package_check("yt-dlp", "yt_dlp", required="fetch" in required)
    av = _package_check("av", "av", required="cpu" in required or "cuda" in required)
    faster_whisper = _package_check(
        "faster-whisper",
        "faster_whisper",
        required="cpu" in required or "cuda" in required,
    )
    ctranslate2 = _package_check(
        "ctranslate2",
        "ctranslate2",
        required="cpu" in required or "cuda" in required,
    )
    checks.extend([yt_dlp, av, faster_whisper, ctranslate2])

    ffmpeg, _ffmpeg_path = _executable_check("ffmpeg", required="fetch" in required)
    ffprobe, _ffprobe_path = _executable_check("ffprobe", required="fetch" in required)
    checks.extend([ffmpeg, ffprobe])

    ctranslate2_version = ctranslate2["details"].get("version")
    gpu, gpu_ready = _gpu_check(ctranslate2_version)
    gpu["required"] = "cuda" in required
    checks.append(gpu)

    output = _output_check(output_dir)
    if output is not None:
        checks.append(output)

    fetch_ready = bool(yt_dlp["ok"] and ffmpeg["ok"] and ffprobe["ok"])
    cpu_ready = bool(av["ok"] and faster_whisper["ok"] and ctranslate2["ok"])
    capabilities = {
        "fetch": fetch_ready,
        "transcribe_cpu": cpu_ready,
        "transcribe_cuda": cpu_ready and gpu_ready,
    }
    requested_ready = all(
        {
            "fetch": capabilities["fetch"],
            "cpu": capabilities["transcribe_cpu"],
            "cuda": capabilities["transcribe_cuda"],
        }[name]
        for name in required
    )
    required_checks_ready = all(item["ok"] for item in checks if item["required"])
    ok = bool(python_ok and uv_check["ok"] and requested_ready and required_checks_ready)
    status = "error" if not ok else ("ready" if capabilities["transcribe_cuda"] else "degraded")
    actions = _remediation_actions(checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "status": status,
        "requirements": sorted(required),
        "capabilities": capabilities,
        "checks": checks,
        "actions": actions,
        "model_cache": _model_cache(),
        "error": None if ok else {"code": "requirements_not_met", "retryable": False},
    }


def _print_human(result: dict[str, Any]) -> None:
    print(f"video-summary doctor: {result['status'].upper()}")
    for check in result["checks"]:
        label = "OK" if check["ok"] else ("FAIL" if check["required"] else "WARN")
        print(f"[{label}] {check['name']}: {check['message']}")
    print("Capabilities:")
    for name, ready in result["capabilities"].items():
        print(f"  {'yes' if ready else 'no ':3}  {name}")
    if result["actions"]:
        print("Suggested actions:")
        for action in result["actions"]:
            mode = "agent may run" if action["automatic"] else "USER ACTION REQUIRED"
            target = f" - {action['program']}" if action["program"] else ""
            print(f"  - {action['id']} [{action['kind']}; {mode}]{target}")
            print(f"    {action['prompt']}")
            if action["display_command"]:
                print(f"    Suggested command: {action['display_command']}")
            if action["after"]:
                print(f"    Afterward: {action['after']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = inspect_environment(
        requirements=args.requirements or ("fetch", "cpu"),
        output_dir=args.output_dir,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        _print_human(result)
    return EXIT_OK if result["ok"] else EXIT_REQUIREMENTS


if __name__ == "__main__":
    raise SystemExit(main())

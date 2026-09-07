#!/usr/bin/env python3
"""Transcribe one local media file with faster-whisper.

URL acquisition belongs to ``fetch.py``. This Skill-internal script loads one
resident batched Whisper model and writes deterministic timestamped artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    from .cuda_runtime import configure_cuda_runtime
except ImportError:  # Direct ``python scripts/transcribe.py`` compatibility.
    from cuda_runtime import configure_cuda_runtime


SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_TRANSCRIPTION = 4
EXIT_FILESYSTEM = 5
EXIT_INTERNAL = 6


@dataclass(frozen=True)
class Profile:
    model: str
    cuda_compute_type: str
    beam_size: int
    batch_size: int


PROFILES = {
    "fast": Profile("small", "float16", beam_size=3, batch_size=16),
    "balanced": Profile("medium", "float16", beam_size=5, batch_size=8),
    "accurate": Profile("large-v3", "int8_float16", beam_size=5, batch_size=8),
}
WORKER_ENV = "VIDEO_SUMMARY_TRANSCRIBE_WORKER"


class TranscribeError(RuntimeError):
    """A safe, machine-readable transcription failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.retryable = retryable


class _ArgumentParser(argparse.ArgumentParser):
    """Keep malformed invocations inside the JSON result contract."""

    def error(self, _message: str) -> None:
        raise TranscribeError(
            "invalid_arguments",
            "The transcribe command arguments are invalid.",
            exit_code=EXIT_USAGE,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Transcribe one local audio/video file with faster-whisper."
    )
    parser.add_argument("input", help="Local media path; use fetch.py for URLs.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for VTT and timestamped transcript outputs.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(PROFILES),
        help="Model/speed profile selected explicitly by the Skill or user.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--language",
        help="Optional ISO-639-1 code; omit to let Whisper detect the language.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional persistent model cache directory.",
    )
    return parser


def _load_backend() -> tuple[Any, Any]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "missing_faster_whisper",
            "faster-whisper is not installed in the Skill environment; initialize it explicitly.",
            exit_code=EXIT_ENVIRONMENT,
        ) from exc
    return WhisperModel, BatchedInferencePipeline


def _cuda_available() -> bool:
    configure_cuda_runtime()
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # noqa: BLE001 - native backend probes can raise implementation errors.
        return False


def _resolve_device(
    requested: str,
    cuda_probe: Callable[[], bool],
) -> tuple[str, list[dict[str, str]]]:
    if requested == "cuda":
        if not cuda_probe():
            raise TranscribeError(
                "cuda_unavailable",
                "CUDA was requested but faster-whisper cannot see a CUDA device.",
                exit_code=EXIT_ENVIRONMENT,
            )
        return "cuda", []
    if requested == "cpu":
        return "cpu", []
    if cuda_probe():
        return "cuda", []
    return "cpu", [
        {
            "code": "cuda_unavailable",
            "message": "CUDA was not available; transcription used the CPU.",
        }
    ]


def _backend_error(
    exc: Exception,
    *,
    device: str,
    fallback_code: str,
    fallback_message: str,
    fallback_exit_code: int,
) -> TranscribeError:
    detail = str(exc).casefold()
    cuda_library = any(name in detail for name in ("cublas", "cudnn", "cudart"))
    missing_library = "not found" in detail or "cannot be loaded" in detail
    incompatible_driver = "cuda driver version is insufficient" in detail
    if device == "cuda" and ((cuda_library and missing_library) or incompatible_driver):
        return TranscribeError(
            "cuda_runtime_unavailable",
            "CTranslate2 could not load the required CUDA runtime libraries.",
            exit_code=EXIT_ENVIRONMENT,
        )
    return TranscribeError(
        fallback_code,
        fallback_message,
        exit_code=fallback_exit_code,
        retryable=True,
    )


def _validate_input(raw_input: str) -> Path:
    try:
        parsed = urlsplit(raw_input)
    except ValueError as exc:
        raise TranscribeError(
            "invalid_input",
            "The local input path is malformed.",
            exit_code=EXIT_USAGE,
        ) from exc
    if parsed.scheme in {"http", "https"}:
        raise TranscribeError(
            "url_not_supported",
            "transcribe accepts local paths only; use fetch.py for URLs.",
            exit_code=EXIT_USAGE,
        )

    try:
        path = Path(raw_input).expanduser().resolve()
    except OSError as exc:
        raise TranscribeError(
            "invalid_input",
            "The local input path could not be resolved.",
            exit_code=EXIT_USAGE,
        ) from exc
    if not path.is_file():
        raise TranscribeError(
            "input_not_found",
            "The local input file does not exist or is not a regular file.",
            exit_code=EXIT_USAGE,
        )
    return path


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _elapsed_seconds(started: float, finished: float) -> float:
    """Return a stable non-negative duration for JSON output."""
    return round(max(0.0, finished - started), 3)


def _normalise_segments(raw_segments: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        start = max(0.0, _finite_float(getattr(raw, "start", 0.0)))
        end = max(start, _finite_float(getattr(raw, "end", start), start))
        text = str(getattr(raw, "text", "")).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _timestamp(seconds: float) -> str:
    total_milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _render_vtt(segments: Sequence[Mapping[str, Any]], language: str | None) -> str:
    lines = ["WEBVTT", ""]
    if language:
        lines.extend([f"Language: {language}", ""])
    for segment in segments:
        lines.append(f"{_timestamp(segment['start'])} --> {_timestamp(segment['end'])}")
        lines.extend(str(segment["text"]).splitlines() or [""])
        lines.append("")
    return "\n".join(lines)


def _render_detailed(segments: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"[{_timestamp(segment['start'])} - {_timestamp(segment['end'])}] {segment['text']}"
        for segment in segments
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _commit_artifacts(output_dir: Path, artifacts: Mapping[str, str]) -> dict[str, Path]:
    targets = {name: output_dir / name for name in artifacts}
    conflicts = [path for path in targets.values() if path.exists()]
    if conflicts:
        raise TranscribeError(
            "output_conflict",
            "Refusing to overwrite existing transcription output: "
            + ", ".join(str(path) for path in conflicts),
            exit_code=EXIT_FILESYSTEM,
        )

    try:
        staging_context = tempfile.TemporaryDirectory(prefix=".video-summary-", dir=output_dir)
    except OSError as exc:
        raise TranscribeError(
            "staging_failed",
            "A temporary transcription directory could not be created.",
            exit_code=EXIT_FILESYSTEM,
        ) from exc

    committed: dict[str, Path] = {}
    with staging_context as staging_name:
        staging = Path(staging_name)
        try:
            for name, content in artifacts.items():
                (staging / name).write_text(content, encoding="utf-8", newline="\n")
            for name in artifacts:
                target = targets[name]
                (staging / name).replace(target)
                committed[name] = target.resolve()
        except OSError as exc:
            for name, target in reversed(tuple(committed.items())):
                try:
                    target.replace(staging / name)
                except OSError:
                    pass
            raise TranscribeError(
                "commit_failed",
                "Transcription outputs could not be committed.",
                exit_code=EXIT_FILESYSTEM,
            ) from exc
    return committed


def _ensure_outputs_available(output_dir: Path) -> None:
    conflicts = [
        output_dir / name
        for name in ("subtitle.vtt", "transcript_detailed.txt")
        if (output_dir / name).exists()
    ]
    if conflicts:
        raise TranscribeError(
            "output_conflict",
            "Refusing to overwrite existing transcription output: "
            + ", ".join(str(path) for path in conflicts),
            exit_code=EXIT_FILESYSTEM,
        )


def transcribe_path(
    input_path: Path,
    output_dir: Path,
    *,
    profile: str,
    requested_device: str = "auto",
    language: str | None = None,
    model_dir: Path | None = None,
    backend: Any | None = None,
    cuda_probe: Callable[[], bool] = _cuda_available,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    total_started = clock()
    if profile not in PROFILES:
        raise TranscribeError(
            "invalid_profile",
            "profile must be fast, balanced, or accurate.",
            exit_code=EXIT_USAGE,
        )
    input_path = _validate_input(str(input_path))
    try:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TranscribeError(
            "output_unavailable",
            "The transcription output directory cannot be created or accessed.",
            exit_code=EXIT_FILESYSTEM,
        ) from exc
    _ensure_outputs_available(output_dir)

    device, warnings = _resolve_device(requested_device, cuda_probe)
    settings = PROFILES[profile]
    compute_type = settings.cuda_compute_type if device == "cuda" else "int8"
    model_name = settings.model
    batch_size = settings.batch_size
    beam_size = settings.beam_size

    if backend is None:
        WhisperModel, BatchedInferencePipeline = _load_backend()
    else:
        WhisperModel = backend.WhisperModel
        BatchedInferencePipeline = backend.BatchedInferencePipeline

    model_kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
    if model_dir is not None:
        try:
            model_dir = model_dir.expanduser().resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TranscribeError(
                "model_dir_unavailable",
                "The model cache directory cannot be created or accessed.",
                exit_code=EXIT_FILESYSTEM,
            ) from exc
        model_kwargs["download_root"] = str(model_dir)

    print(
        f"Loading {model_name} on {device}/{compute_type} (batch={batch_size})...",
        file=sys.stderr,
    )
    model_load_started = clock()
    try:
        model = WhisperModel(model_name, **model_kwargs)
        pipeline = BatchedInferencePipeline(model)
    except Exception as exc:
        raise _backend_error(
            exc,
            device=device,
            fallback_code="model_load_failed",
            fallback_message="The faster-whisper model could not be loaded or downloaded.",
            fallback_exit_code=EXIT_ENVIRONMENT,
        ) from exc
    model_load_seconds = _elapsed_seconds(model_load_started, clock())

    transcription_started = clock()
    try:
        raw_segments, info = pipeline.transcribe(
            str(input_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            batch_size=batch_size,
            vad_filter=True,
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        segments = _normalise_segments(raw_segments)
    except Exception as exc:
        raise _backend_error(
            exc,
            device=device,
            fallback_code="transcription_failed",
            fallback_message="faster-whisper could not transcribe the local input.",
            fallback_exit_code=EXIT_TRANSCRIPTION,
        ) from exc
    transcription_seconds = _elapsed_seconds(transcription_started, clock())
    if not segments:
        raise TranscribeError(
            "no_speech_segments",
            "faster-whisper produced no speech segments.",
            exit_code=EXIT_TRANSCRIPTION,
        )

    detected_language = getattr(info, "language", None)
    language_probability = _finite_float(getattr(info, "language_probability", None), 0.0)
    duration = _finite_float(getattr(info, "duration", None), segments[-1]["end"])
    if duration <= 0:
        duration = segments[-1]["end"]
    artifact_write_started = clock()
    committed = _commit_artifacts(
        output_dir,
        {
            "subtitle.vtt": _render_vtt(segments, detected_language),
            "transcript_detailed.txt": _render_detailed(segments),
        },
    )
    artifact_write_seconds = _elapsed_seconds(artifact_write_started, clock())
    total_seconds = _elapsed_seconds(total_started, clock())
    realtime_factor = (
        round(transcription_seconds / duration, 3)
        if duration > 0 and transcription_seconds > 0
        else None
    )
    processing_speed = (
        round(duration / transcription_seconds, 3)
        if duration > 0 and transcription_seconds > 0
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "status": "transcribed",
        "source": str(input_path),
        "profile": profile,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": detected_language,
        "language_probability": language_probability,
        "duration_seconds": duration,
        "segment_count": len(segments),
        "timing": {
            "total_seconds": total_seconds,
            "model_load_seconds": model_load_seconds,
            "transcription_seconds": transcription_seconds,
            "artifact_write_seconds": artifact_write_seconds,
            "realtime_factor": realtime_factor,
            "processing_speed_x": processing_speed,
        },
        "artifacts": {name: str(path) for name, path in committed.items()},
        "warnings": warnings,
        "error": None,
    }


def _error_result(raw_input: str, error: TranscribeError) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "status": "error",
        "source": raw_input,
        "artifacts": None,
        "warnings": [],
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }


def _serialize_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=True)


def _suppress_windows_crash_dialog() -> None:
    """Keep native CTranslate2 failures from opening a blocking Windows dialog."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfaulterrorbox = 0x0002
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)
    except (AttributeError, OSError):
        pass


def _source_argument(arguments: Sequence[str]) -> str:
    """Best-effort extraction of the local input for parent-side crash reports."""
    value_options = {"--output-dir", "--profile", "--device", "--language", "--model-dir"}
    skip_value = False
    positional_only = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if positional_only:
            return argument
        if argument == "--":
            positional_only = True
            continue
        if argument in value_options:
            skip_value = True
            continue
        if any(argument.startswith(f"{option}=") for option in value_options):
            continue
        if not argument.startswith("-"):
            return argument
    return ""


def _is_json_object(value: str) -> bool:
    try:
        return isinstance(json.loads(value), dict)
    except (TypeError, ValueError):
        return False


def _run_isolated(arguments: Sequence[str]) -> int:
    """Run native inference in a child so a C++ crash cannot take down the JSON wrapper."""
    environment = os.environ.copy()
    environment[WORKER_ENV] = "1"
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            creationflags=creation_flags,
        )
    except OSError:
        error = TranscribeError(
            "worker_start_failed",
            "The isolated faster-whisper worker could not be started.",
            exit_code=EXIT_ENVIRONMENT,
        )
        print(_serialize_result(_error_result("", error)))
        return error.exit_code

    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.stdout and (completed.returncode == EXIT_OK or _is_json_object(completed.stdout)):
        sys.stdout.write(completed.stdout)
        return completed.returncode

    error = TranscribeError(
        "native_runtime_crash",
        "The faster-whisper native runtime terminated unexpectedly.",
        exit_code=EXIT_TRANSCRIPTION,
        retryable=True,
    )
    print(_serialize_result(_error_result(_source_argument(arguments), error)))
    return error.exit_code


def run(args: argparse.Namespace) -> int:
    try:
        result = transcribe_path(
            Path(args.input),
            args.output_dir,
            profile=args.profile,
            requested_device=args.device,
            language=args.language,
            model_dir=args.model_dir,
        )
    except TranscribeError as exc:
        print(_serialize_result(_error_result(args.input, exc)))
        return exc.exit_code
    except Exception:  # noqa: BLE001 - preserve the public JSON error contract.
        error = TranscribeError(
            "internal_error",
            "transcribe failed because of an unexpected internal error.",
            exit_code=EXIT_INTERNAL,
        )
        print(_serialize_result(_error_result(args.input, error)))
        return error.exit_code
    print(_serialize_result(result))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and os.environ.get(WORKER_ENV) != "1":
        return _run_isolated(sys.argv[1:])
    _suppress_windows_crash_dialog()
    try:
        args = build_parser().parse_args(argv)
    except TranscribeError as exc:
        print(_serialize_result(_error_result("", exc)))
        return exc.exit_code
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

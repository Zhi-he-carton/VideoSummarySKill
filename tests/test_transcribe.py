from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import transcribe


def fake_backend(
    segments: list[SimpleNamespace] | None = None,
    *,
    language: str = "zh",
) -> tuple[SimpleNamespace, dict[str, Any]]:
    state: dict[str, Any] = {"models": [], "transcribe_calls": []}
    output_segments = (
        segments
        if segments is not None
        else [
            SimpleNamespace(start=0.0, end=4.25, text=" 第一段。 "),
            SimpleNamespace(start=305.0, end=309.5, text="第二段。"),
        ]
    )

    class WhisperModel:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs
            state["models"].append((model_name, kwargs))

    class BatchedInferencePipeline:
        def __init__(self, model: WhisperModel) -> None:
            self.model = model

        def transcribe(
            self, input_path: str, **kwargs: Any
        ) -> tuple[Any, SimpleNamespace]:
            state["transcribe_calls"].append((input_path, kwargs))
            info = SimpleNamespace(
                language=language, language_probability=0.97, duration=310.0
            )
            return iter(output_segments), info

    return SimpleNamespace(
        WhisperModel=WhisperModel,
        BatchedInferencePipeline=BatchedInferencePipeline,
    ), state


def make_input(tmp_path: Path) -> Path:
    source = tmp_path / "课程 audio.mp3"
    source.write_bytes(b"test-media")
    return source


def test_balanced_cpu_writes_timestamped_artifacts(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    output_dir = tmp_path / "output"
    backend, state = fake_backend()

    result = transcribe.transcribe_path(
        source,
        output_dir,
        profile="balanced",
        requested_device="cpu",
        backend=backend,
        cuda_probe=lambda: False,
    )

    assert result["status"] == "transcribed"
    assert result["model"] == "medium"
    assert result["device"] == "cpu"
    assert result["compute_type"] == "int8"
    assert result["segment_count"] == 2
    assert state["models"] == [("medium", {"device": "cpu", "compute_type": "int8"})]
    call = state["transcribe_calls"][0][1]
    assert call["batch_size"] == 8
    assert call["beam_size"] == 5
    assert call["vad_filter"] is True
    assert call["word_timestamps"] is False

    vtt = (output_dir / "subtitle.vtt").read_text(encoding="utf-8")
    detailed = (output_dir / "transcript_detailed.txt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT\n\nLanguage: zh")
    assert "00:00:00.000 --> 00:00:04.250" in vtt
    assert "[00:00:00.000 - 00:00:04.250] 第一段。" in detailed
    assert "[00:05:05.000 - 00:05:09.500] 第二段。" in detailed
    assert not (output_dir / "transcript.txt").exists()


def test_transcription_reports_deterministic_timing(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    backend, _state = fake_backend()
    timestamps = iter((10.0, 11.0, 13.5, 14.0, 18.0, 18.5, 20.0, 21.0))

    result = transcribe.transcribe_path(
        source,
        tmp_path / "timed-output",
        profile="fast",
        requested_device="cpu",
        backend=backend,
        cuda_probe=lambda: False,
        clock=lambda: next(timestamps),
    )

    assert result["timing"] == {
        "total_seconds": 11.0,
        "model_load_seconds": 2.5,
        "transcription_seconds": 4.0,
        "artifact_write_seconds": 1.5,
        "realtime_factor": 0.013,
        "processing_speed_x": 77.5,
    }


def test_accurate_cuda_uses_large_v3_int8_and_batch_eight(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    backend, state = fake_backend()

    result = transcribe.transcribe_path(
        source,
        tmp_path / "cuda-output",
        profile="accurate",
        requested_device="cuda",
        backend=backend,
        cuda_probe=lambda: True,
    )

    assert result["model"] == "large-v3"
    assert result["device"] == "cuda"
    assert result["compute_type"] == "int8_float16"
    assert state["transcribe_calls"][0][1]["batch_size"] == 8
    assert state["transcribe_calls"][0][1]["beam_size"] == 5
    assert state["transcribe_calls"][0][1]["condition_on_previous_text"] is True


def test_auto_falls_back_to_cpu_with_warning(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    backend, _state = fake_backend()

    result = transcribe.transcribe_path(
        source,
        tmp_path / "auto-output",
        profile="fast",
        requested_device="auto",
        backend=backend,
        cuda_probe=lambda: False,
    )

    assert result["device"] == "cpu"
    assert result["warnings"][0]["code"] == "cuda_unavailable"


def test_explicit_cuda_fails_instead_of_silently_using_cpu(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    backend, _state = fake_backend()

    with pytest.raises(transcribe.TranscribeError) as raised:
        transcribe.transcribe_path(
            source,
            tmp_path / "output",
            profile="balanced",
            requested_device="cuda",
            backend=backend,
            cuda_probe=lambda: False,
        )

    assert raised.value.code == "cuda_unavailable"
    assert raised.value.exit_code == transcribe.EXIT_ENVIRONMENT


def test_missing_cuda_runtime_is_classified_as_environment_error() -> None:
    error = transcribe._backend_error(
        RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"),
        device="cuda",
        fallback_code="transcription_failed",
        fallback_message="fallback",
        fallback_exit_code=transcribe.EXIT_TRANSCRIPTION,
    )

    assert error.code == "cuda_runtime_unavailable"
    assert error.exit_code == transcribe.EXIT_ENVIRONMENT
    assert error.retryable is False


def test_url_input_is_rejected_without_loading_backend(tmp_path: Path) -> None:
    backend, state = fake_backend()

    with pytest.raises(transcribe.TranscribeError) as raised:
        transcribe.transcribe_path(
            Path("https://example.com/audio.mp3"),
            tmp_path / "output",
            profile="fast",
            backend=backend,
        )

    assert raised.value.code == "url_not_supported"
    assert state["models"] == []


def test_existing_output_is_rejected_before_model_load(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "transcript_detailed.txt"
    existing.write_text("keep me", encoding="utf-8")
    backend, state = fake_backend()

    with pytest.raises(transcribe.TranscribeError) as raised:
        transcribe.transcribe_path(
            source,
            output_dir,
            profile="fast",
            backend=backend,
            cuda_probe=lambda: False,
        )

    assert raised.value.code == "output_conflict"
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert state["models"] == []


def test_empty_transcription_does_not_commit_outputs(tmp_path: Path) -> None:
    source = make_input(tmp_path)
    backend, _state = fake_backend(segments=[])
    output_dir = tmp_path / "output"

    with pytest.raises(transcribe.TranscribeError) as raised:
        transcribe.transcribe_path(
            source,
            output_dir,
            profile="balanced",
            requested_device="cpu",
            backend=backend,
        )

    assert raised.value.code == "no_speech_segments"
    assert not (output_dir / "subtitle.vtt").exists()


def test_invalid_cli_arguments_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = transcribe.main(["missing.mp3", "--output-dir", "out"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == transcribe.EXIT_USAGE
    assert result["error"]["code"] == "invalid_arguments"


def test_json_serialization_is_gbk_safe() -> None:
    payload = {"source": "课程 🎧.mp3"}

    serialized = transcribe._serialize_result(payload)

    serialized.encode("gbk")
    assert json.loads(serialized) == payload


def test_native_worker_crash_becomes_json_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = SimpleNamespace(returncode=-1073741819, stdout="", stderr="")
    monkeypatch.setattr(
        transcribe.subprocess, "run", lambda *_args, **_kwargs: completed
    )

    exit_code = transcribe._run_isolated(
        ["--profile", "fast", "audio.mp3", "--output-dir", "out"]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == transcribe.EXIT_TRANSCRIPTION
    assert result["error"]["code"] == "native_runtime_crash"
    assert result["source"] == "audio.mp3"


def test_partial_worker_output_is_not_forwarded_as_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = SimpleNamespace(
        returncode=-1073741819, stdout='{"ok":', stderr="native log"
    )
    monkeypatch.setattr(
        transcribe.subprocess, "run", lambda *_args, **_kwargs: completed
    )

    exit_code = transcribe._run_isolated(["audio.mp3"])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == transcribe.EXIT_TRANSCRIPTION
    assert result["error"]["code"] == "native_runtime_crash"
    assert captured.err == "native log"

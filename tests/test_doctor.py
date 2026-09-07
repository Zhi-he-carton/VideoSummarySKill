from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import doctor


def test_cuda_44_uses_cuda12_and_cudnn8_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")

    assert doctor._cuda_library_names("4.4.0") == [
        "cublas64_12.dll",
        "cudnn_ops_infer64_8.dll",
    ]


def test_cuda_3_uses_cuda11_and_cudnn8_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")

    assert doctor._cuda_library_names("3.24.0") == [
        "cublas64_11.dll",
        "cudnn_ops_infer64_8.dll",
    ]


def test_cuda_45_uses_cuda12_and_cudnn9_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")

    assert doctor._cuda_library_names("4.5.0") == [
        "cublas64_12.dll",
        "cudnn_ops64_9.dll",
    ]


def test_cuda_481_only_requires_cublas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")

    assert doctor._cuda_library_names("4.8.1") == ["cublas64_12.dll"]


def test_installed_cuda_toolkit_version_is_reported() -> None:
    assert doctor._installed_cuda_toolkit_version(
        [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"]
    ) == (12, 0)


def test_cuda_481_accepts_working_cuda_120_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(
        doctor,
        "configure_cuda_runtime",
        lambda: [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"],
    )
    monkeypatch.setattr(
        doctor.shutil,
        "which",
        lambda name: "nvidia-smi.exe" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(doctor, "_probe_cuda_devices", lambda: 1)
    monkeypatch.setattr(doctor, "_loadable_library", lambda _name: (True, None))

    check, ready = doctor._gpu_check("4.8.1")

    assert ready is True
    assert check["ok"] is True
    assert check["details"]["installed_cuda_toolkit"] == "12.0"
    assert "required_cuda_toolkit" not in check["details"]


def test_model_cache_uses_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    cached = tmp_path / "hub" / "models--Systran--faster-whisper-small"
    cached.mkdir(parents=True)
    (cached / "marker").write_text("cached", encoding="utf-8")

    result = doctor._model_cache()

    assert result["root"] == str((tmp_path / "hub").resolve())
    assert result["profiles"]["fast"]["cached"] is True
    assert result["profiles"]["balanced"]["cached"] is False


def test_default_requirements_are_fetch_and_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_inspect_environment(
        *, requirements: object, output_dir: object
    ) -> dict[str, object]:
        captured["requirements"] = requirements
        captured["output_dir"] = output_dir
        return {
            "ok": True,
            "status": "degraded",
            "checks": [],
            "capabilities": {},
            "actions": [],
        }

    monkeypatch.setattr(doctor, "inspect_environment", fake_inspect_environment)

    assert doctor.main([]) == doctor.EXIT_OK
    assert captured["requirements"] == ("fetch", "cpu")


def test_json_mode_emits_one_machine_readable_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "schema_version": 1,
        "ok": False,
        "status": "error",
        "checks": [],
        "capabilities": {},
    }
    monkeypatch.setattr(doctor, "inspect_environment", lambda **_kwargs: expected)

    exit_code = doctor.main(["--json", "--require", "cuda"])

    assert exit_code == doctor.EXIT_REQUIREMENTS
    assert json.loads(capsys.readouterr().out) == expected


def test_output_check_does_not_create_missing_directory(tmp_path: Path) -> None:
    requested = tmp_path / "missing" / "nested"

    result = doctor._output_check(requested)

    assert result is not None
    assert result["ok"] is True
    assert not requested.exists()


def test_uv_managed_failure_proposes_one_locked_sync() -> None:
    checks = [
        doctor._check("uv", True, "ok", required=True),
        doctor._check("faster-whisper", False, "broken", required=True),
        doctor._check("ctranslate2", False, "broken", required=True),
    ]

    actions = doctor._remediation_actions(checks)

    assert len(actions) == 1
    assert actions[0]["id"] == "sync-locked-environment"
    assert actions[0]["kind"] == "repair"
    assert actions[0]["program"] == "video-summary Python environment"
    assert actions[0]["manager"] == "uv"
    assert actions[0]["automatic"] is True
    assert actions[0]["requires_user_action"] is False
    assert actions[0]["command"] == doctor.UV_SYNC_COMMAND
    assert "--locked" in actions[0]["command"]


def test_cuda_failure_requires_manual_system_action() -> None:
    checks = [
        doctor._check(
            "cuda-runtime",
            False,
            "missing",
            required=True,
            details={
                "libraries": [
                    {"name": "cublas64_12.dll", "loadable": False},
                    {"name": "cudnn_ops_infer64_8.dll", "loadable": False},
                ]
            },
        )
    ]

    actions = doctor._remediation_actions(checks)

    assert actions[0]["id"] == "install-cuda-runtime"
    assert actions[0]["kind"] == "install"
    assert (
        actions[0]["program"] == "NVIDIA CUDA runtime libraries required by CTranslate2"
    )
    assert actions[0]["manager"] == "system"
    assert actions[0]["automatic"] is False
    assert actions[0]["requires_user_action"] is True
    assert actions[0]["command"] is None
    assert "cublas64_12.dll" in actions[0]["prompt"]


def test_ffmpeg_action_has_platform_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    checks = [doctor._check("ffmpeg", False, "missing", required=True)]

    actions = doctor._remediation_actions(checks)

    assert actions[0]["id"] == "install-ffmpeg"
    assert actions[0]["kind"] == "install"
    assert actions[0]["program"] == "FFmpeg (including FFprobe)"
    assert actions[0]["automatic"] is False
    assert actions[0]["requires_user_action"] is True
    assert actions[0]["command"][:4] == ["winget", "install", "--id", "Gyan.FFmpeg"]


def test_macos_cuda_failure_recommends_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    checks = [doctor._check("cuda-runtime", False, "unsupported", required=True)]

    actions = doctor._remediation_actions(checks)

    assert [action["id"] for action in actions] == ["use-cpu-on-macos"]
    assert actions[0]["command"] is None
    assert "--device cpu" in actions[0]["prompt"]
    assert "NVIDIA" not in actions[0]["program"]


def test_human_output_names_manual_program_and_follow_up(
    capsys: pytest.CaptureFixture[str],
) -> None:
    action = doctor._action(
        "install-example",
        kind="install",
        program="Example Program",
        manager="system",
        automatic=False,
        prompt="Example Program is required for the requested capability.",
        command=["installer", "install", "example"],
        after="Rerun doctor.",
    )
    result = {
        "status": "error",
        "checks": [],
        "capabilities": {},
        "actions": [action],
    }

    doctor._print_human(result)

    output = capsys.readouterr().out
    assert "USER ACTION REQUIRED" in output
    assert "Example Program" in output
    assert "Suggested command:" in output
    assert "Afterward: Rerun doctor." in output

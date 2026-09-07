from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import cli, fetch


def test_fetch_contract() -> None:
    args = fetch.build_parser().parse_args(
        ["https://example.com/lesson", "--output-dir", "output"]
    )

    assert args.url == "https://example.com/lesson"
    assert args.output_dir == Path("output")
    assert args.keep_audio is False


def test_fetch_keep_audio_contract() -> None:
    args = fetch.build_parser().parse_args(
        ["https://example.com/lesson", "--output-dir", "output", "--keep-audio"]
    )

    assert args.keep_audio is True


def test_fetch_explicit_chrome_cookie_contract() -> None:
    args = fetch.build_parser().parse_args(
        [
            "https://example.com/lesson",
            "--output-dir",
            "output",
            "--cookies-from-browser",
            "chrome",
        ]
    )

    assert args.cookies_from_browser == "chrome"


def test_json_output_is_console_encoding_safe_and_round_trips_unicode() -> None:
    payload = {"title": "中文课程 🎬"}

    serialized = fetch._serialize_result(payload)

    serialized.encode("gbk")
    assert json.loads(serialized) == payload


def test_invalid_arguments_return_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = fetch.main(["--output-dir", "output"])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == fetch.EXIT_USAGE
    assert result["error"]["code"] == "invalid_arguments"


def test_dispatcher_forwards_subcommand_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    module = SimpleNamespace(main=lambda arguments: captured.extend(arguments) or 7)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: module)

    exit_code = cli.main(["fetch", "https://example.com", "--output-dir", "output"])

    assert exit_code == 7
    assert captured == ["https://example.com", "--output-dir", "output"]

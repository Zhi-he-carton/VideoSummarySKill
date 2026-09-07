from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
from scripts import fetch

URL = "https://example.com/lesson"


def track() -> list[dict[str, str]]:
    return [{"ext": "vtt", "url": "https://captions.invalid/track.vtt"}]


class FakeYtDlp:
    def __init__(
        self,
        info: dict[str, Any],
        *,
        failed_caption_languages: set[str] | None = None,
        metadata_error: Exception | None = None,
    ) -> None:
        self.info = info
        self.failed_caption_languages = failed_caption_languages or set()
        self.metadata_error = metadata_error
        self.calls: list[dict[str, Any]] = []

        owner = self

        class YoutubeDL:
            def __init__(self, options: dict[str, Any]) -> None:
                self.options = options
                owner.calls.append(options)

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def extract_info(self, _url: str, *, download: bool) -> dict[str, Any]:
                assert download is False
                if owner.metadata_error is not None:
                    raise owner.metadata_error
                return owner.info

            def download(self, urls: list[str]) -> int:
                assert urls == [URL]
                output_dir = Path(self.options["outtmpl"]).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                if self.options.get("writesubtitles") or self.options.get(
                    "writeautomaticsub"
                ):
                    language = self.options["subtitleslangs"][0]
                    if language in owner.failed_caption_languages:
                        raise RuntimeError("caption unavailable")
                    (output_dir / f"caption.{language}.vtt").write_text(
                        "WEBVTT\n\n00:00.000 --> 00:01.000\nlesson\n",
                        encoding="utf-8",
                    )
                    return 0

                assert self.options["format"] == "bestaudio/best"
                (output_dir / "audio.mp3").write_bytes(b"fake-mp3")
                return 0

        self.module = SimpleNamespace(YoutubeDL=YoutubeDL)


def base_info() -> dict[str, Any]:
    return {
        "id": "lesson-1",
        "title": "Lesson",
        "duration": 60,
        "webpage_url": URL,
        "extractor_key": "Generic",
        "subtitles": {},
        "automatic_captions": {},
    }


def candidate_pairs(info: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (item.language, item.kind) for item in fetch.select_caption_candidates(info)
    ]


def test_caption_priority_is_fixed() -> None:
    info = base_info()
    info["subtitles"] = {"zh-Hans": track(), "en": track()}
    info["automatic_captions"] = {"zh-Hans": track(), "en": track()}

    assert candidate_pairs(info) == [
        ("zh-Hans", "manual"),
        ("zh-Hans", "automatic"),
        ("en", "manual"),
        ("en", "automatic"),
    ]


def test_chinese_automatic_precedes_english_manual() -> None:
    info = base_info()
    info["subtitles"] = {"en": track()}
    info["automatic_captions"] = {"zh-Hant": track()}

    assert candidate_pairs(info)[:2] == [
        ("zh-Hant", "automatic"),
        ("en", "manual"),
    ]


def test_caption_success_does_not_download_media(tmp_path: Path) -> None:
    info = base_info()
    info["subtitles"] = {"zh-Hans": track(), "en": track()}
    backend = FakeYtDlp(info)

    result = fetch.acquire(URL, tmp_path, ytdlp=backend.module)

    assert result["status"] == "subtitle"
    assert result["subtitle"]["language"] == "zh-Hans"
    assert result["subtitle"]["kind"] == "manual"
    assert result["audio"] is None
    assert (tmp_path / "subtitle.vtt").is_file()
    assert not (tmp_path / "audio.mp3").exists()
    assert len(backend.calls) == 2
    assert all("format" not in options for options in backend.calls)


def test_no_caption_downloads_audio_only(tmp_path: Path) -> None:
    backend = FakeYtDlp(base_info())

    result = fetch.acquire(
        URL,
        tmp_path,
        ytdlp=backend.module,
        which=lambda name: "ffmpeg" if name == "ffmpeg" else None,
        validate_audio=lambda _path, _ffmpeg: True,
    )

    assert result["status"] == "audio"
    assert result["next_action"] == "transcribe_audio"
    assert result["subtitle"] is None
    assert (tmp_path / "audio.mp3").read_bytes() == b"fake-mp3"
    audio_options = backend.calls[-1]
    assert audio_options["format"] == "bestaudio/best"
    assert "writesubtitles" not in audio_options


def test_keep_audio_preserves_both_artifacts(tmp_path: Path) -> None:
    info = base_info()
    info["automatic_captions"] = {"zh-Hant": track()}
    backend = FakeYtDlp(info)

    result = fetch.acquire(
        URL,
        tmp_path,
        keep_audio=True,
        ytdlp=backend.module,
        which=lambda _name: "ffmpeg",
        validate_audio=lambda _path, _ffmpeg: True,
    )

    assert result["status"] == "subtitle_and_audio"
    assert result["subtitle"]["kind"] == "automatic"
    assert Path(result["subtitle"]["path"]).is_absolute()
    assert Path(result["audio"]["path"]).is_absolute()


def test_failed_preferred_caption_tries_next_permitted_track(tmp_path: Path) -> None:
    info = base_info()
    info["subtitles"] = {"zh-Hans": track(), "en": track()}
    info["automatic_captions"] = {"zh-Hant": track()}
    backend = FakeYtDlp(info, failed_caption_languages={"zh-Hans"})

    result = fetch.acquire(URL, tmp_path, ytdlp=backend.module)

    assert result["subtitle"]["language"] == "zh-Hant"
    assert result["subtitle"]["kind"] == "automatic"
    assert result["warnings"][0]["code"] == "caption_download_failed"


def test_options_never_request_all_or_cookies(tmp_path: Path) -> None:
    info = base_info()
    info["automatic_captions"] = {"en": track()}
    backend = FakeYtDlp(info)

    fetch.acquire(URL, tmp_path, ytdlp=backend.module)

    for options in backend.calls:
        assert options.get("subtitleslangs") != ["all"]
        assert "cookiefile" not in options
        assert "cookiesfrombrowser" not in options
        assert options["extractor_args"] == {"youtube": {"skip": ["translated_subs"]}}


def test_explicit_chrome_cookie_authorization_reaches_every_request(
    tmp_path: Path,
) -> None:
    info = base_info()
    info["subtitles"] = {"zh": track()}
    backend = FakeYtDlp(info)

    fetch.acquire(
        URL,
        tmp_path,
        keep_audio=True,
        cookie_browser="chrome",
        ytdlp=backend.module,
        which=lambda _name: "ffmpeg",
        validate_audio=lambda _path, _ffmpeg: True,
    )

    assert len(backend.calls) == 3
    for options in backend.calls:
        assert options["cookiesfrombrowser"] == ("chrome", None, None, None)


def test_audio_fallback_requires_ffmpeg(tmp_path: Path) -> None:
    backend = FakeYtDlp(base_info())

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(URL, tmp_path, ytdlp=backend.module, which=lambda _name: None)

    assert raised.value.code == "missing_ffmpeg"
    assert raised.value.exit_code == fetch.EXIT_ENVIRONMENT
    assert not (tmp_path / "audio.mp3").exists()


def test_existing_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "subtitle.vtt"
    existing.write_text("existing", encoding="utf-8")
    backend = FakeYtDlp(base_info())

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(URL, tmp_path, ytdlp=backend.module)

    assert raised.value.code == "output_conflict"
    assert existing.read_text(encoding="utf-8") == "existing"
    assert backend.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/video",
        "http://127.0.0.1/video",
        "http://[::1]/video",
        "http://192.168.1.20/video",
        "http://2130706433/video",
        "https://user:password@example.com/video",
        "http://127。0。0。1/video",
        "http://localhost。/video",
        "http://１２７.０.０.１/video",
    ],
)
def test_private_or_credential_bearing_urls_are_rejected(
    url: str, tmp_path: Path
) -> None:
    backend = FakeYtDlp(base_info())

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(url, tmp_path, ytdlp=backend.module)

    assert raised.value.exit_code == fetch.EXIT_USAGE
    assert backend.calls == []


def test_malformed_ipv6_url_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(
            "http://[::1/video", tmp_path, ytdlp=FakeYtDlp(base_info()).module
        )

    assert raised.value.code == "invalid_url"
    assert raised.value.exit_code == fetch.EXIT_USAGE


def test_dpapi_cookie_failure_has_a_specific_local_error(tmp_path: Path) -> None:
    backend = FakeYtDlp(
        base_info(), metadata_error=RuntimeError("Failed to decrypt with DPAPI")
    )

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(URL, tmp_path, cookie_browser="chrome", ytdlp=backend.module)

    assert raised.value.code == "cookie_unavailable"
    assert raised.value.exit_code == fetch.EXIT_ENVIRONMENT


def test_wrapped_ytdlp_cookie_error_has_a_specific_local_error(tmp_path: Path) -> None:
    class CookieLoadError(Exception):
        pass

    wrapped = CookieLoadError("failed to load cookies")
    wrapped.__cause__ = RuntimeError("Failed to decrypt with DPAPI")
    backend = FakeYtDlp(base_info(), metadata_error=wrapped)

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(URL, tmp_path, cookie_browser="chrome", ytdlp=backend.module)

    assert raised.value.code == "cookie_unavailable"
    assert raised.value.exit_code == fetch.EXIT_ENVIRONMENT


def test_invalid_audio_is_not_committed(tmp_path: Path) -> None:
    backend = FakeYtDlp(base_info())

    with pytest.raises(fetch.FetchError) as raised:
        fetch.acquire(
            URL,
            tmp_path,
            ytdlp=backend.module,
            which=lambda _name: "ffmpeg",
            validate_audio=lambda _path, _ffmpeg: False,
        )

    assert raised.value.code == "invalid_audio"
    assert not (tmp_path / "audio.mp3").exists()

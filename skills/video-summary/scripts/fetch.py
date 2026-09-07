#!/usr/bin/env python3
"""Fetch preferred source captions, or audio when no captions are available.

This is a Skill-internal adapter around yt-dlp. It deliberately exposes a
small interface so the calling agent does not need to assemble yt-dlp options
or infer caption priority from filenames.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_REMOTE = 4
EXIT_FILESYSTEM = 5
EXIT_INTERNAL = 6

# The order inside each group is significant. Only these explicit language
# tags are accepted; the wrapper never requests every subtitle track.
CHINESE_LANGUAGE_GROUPS = (
    ("zh-Hans", "zh-CN", "zh-SG"),
    ("zh-Hant", "zh-TW", "zh-HK"),
    ("zh",),
)
ENGLISH_LANGUAGE_GROUPS = (
    ("en",),
    ("en-US", "en-GB"),
    ("en-orig",),
)


@dataclass(frozen=True)
class CaptionCandidate:
    """One exact caption track selected from yt-dlp metadata."""

    language: str
    kind: str


class FetchError(RuntimeError):
    """A failure that can be safely represented in the public JSON result."""

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


class _FetchArgumentParser(argparse.ArgumentParser):
    """Turn invalid invocations into the same machine-readable error contract."""

    def error(self, _message: str) -> None:
        raise FetchError(
            "invalid_arguments",
            "The fetch command arguments are invalid.",
            exit_code=EXIT_USAGE,
        )


class _QuietYtDlpLogger:
    """Prevent yt-dlp from printing request details outside our JSON contract."""

    def debug(self, _message: str) -> None:
        pass

    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass


YTDLP_LOGGER = _QuietYtDlpLogger()


def build_parser() -> argparse.ArgumentParser:
    """Build the narrow, Skill-internal command interface."""
    parser = _FetchArgumentParser(
        description="Fetch preferred captions, falling back to audio when none exist."
    )
    parser.add_argument("url", help="A single HTTP(S) URL supported by yt-dlp.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory that will receive subtitle.vtt and/or audio.mp3.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Also keep audio.mp3 when a source caption track is available.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=("chrome",),
        help="Use Chrome cookies only after the user explicitly authorizes access.",
    )
    return parser


def _load_ytdlp() -> ModuleType:
    try:
        import yt_dlp
    except ImportError as exc:
        raise FetchError(
            "missing_yt_dlp",
            "yt-dlp is not installed in the Skill environment; run the explicit setup step.",
            exit_code=EXIT_ENVIRONMENT,
        ) from exc
    return yt_dlp


def _common_options(cookie_browser: str | None = None) -> dict[str, Any]:
    """Options shared by metadata, subtitle, and audio requests."""
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "logger": YTDLP_LOGGER,
        "extractor_args": {"youtube": {"skip": ["translated_subs"]}},
    }
    if cookie_browser is not None:
        options["cookiesfrombrowser"] = (cookie_browser, None, None, None)
    return options


def _validate_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError as exc:
        raise FetchError(
            "invalid_url",
            "fetch accepts one well-formed public HTTP(S) URL.",
            exit_code=EXIT_USAGE,
        ) from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise FetchError(
            "invalid_url",
            "fetch accepts one public HTTP(S) URL without embedded credentials; "
            "send local paths to transcribe instead.",
            exit_code=EXIT_USAGE,
        )

    try:
        normalized_host = hostname.encode("idna").decode("ascii").casefold().rstrip(".")
    except UnicodeError as exc:
        raise FetchError(
            "invalid_url",
            "The URL hostname is not valid.",
            exit_code=EXIT_USAGE,
        ) from exc
    if not normalized_host:
        raise FetchError(
            "invalid_url",
            "The URL hostname is not valid.",
            exit_code=EXIT_USAGE,
        )
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise FetchError(
            "private_url",
            "fetch does not access localhost or private network resources.",
            exit_code=EXIT_USAGE,
        )

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        # Reject alternate numeric IPv4 spellings such as 2130706433 or 127.1;
        # their interpretation differs between URL clients and can hide loopback.
        numeric_parts = normalized_host.split(".")
        looks_numeric = bool(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", normalized_host))
        looks_numeric = looks_numeric or (
            len(numeric_parts) > 1
            and all(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in numeric_parts)
        )
        if looks_numeric:
            raise FetchError(
                "private_url",
                "fetch does not accept ambiguous numeric network addresses.",
                exit_code=EXIT_USAGE,
            ) from None
    else:
        if not address.is_global:
            raise FetchError(
                "private_url",
                "fetch does not access localhost or private network resources.",
                exit_code=EXIT_USAGE,
            )


def _track_map(info: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    tracks = info.get(key)
    return tracks if isinstance(tracks, Mapping) else {}


def _available_track(tracks: Mapping[str, Any], language: str) -> str | None:
    """Return the extractor's exact language key when it has at least one format."""
    keys = {str(key).casefold(): str(key) for key in tracks}
    actual = keys.get(language.casefold())
    if actual is None:
        return None
    formats = tracks.get(actual)
    usable = isinstance(formats, Sequence) and not isinstance(formats, str) and bool(formats)
    return actual if usable else None


def _candidates_for(
    tracks: Mapping[str, Any],
    groups: Sequence[Sequence[str]],
    kind: str,
) -> list[CaptionCandidate]:
    candidates: list[CaptionCandidate] = []
    seen: set[str] = set()
    for aliases in groups:
        for alias in aliases:
            actual = _available_track(tracks, alias)
            if actual is not None and actual.casefold() not in seen:
                candidates.append(CaptionCandidate(language=actual, kind=kind))
                seen.add(actual.casefold())
    return candidates


def select_caption_candidates(info: Mapping[str, Any]) -> list[CaptionCandidate]:
    """Apply the frozen manual/automatic and Chinese/English priority."""
    manual = _track_map(info, "subtitles")
    automatic = _track_map(info, "automatic_captions")
    return [
        *_candidates_for(manual, CHINESE_LANGUAGE_GROUPS, "manual"),
        *_candidates_for(automatic, CHINESE_LANGUAGE_GROUPS, "automatic"),
        *_candidates_for(manual, ENGLISH_LANGUAGE_GROUPS, "manual"),
        *_candidates_for(automatic, ENGLISH_LANGUAGE_GROUPS, "automatic"),
    ]


def _source_info(info: Mapping[str, Any], requested_url: str) -> dict[str, Any]:
    """Keep only non-sensitive metadata needed by the next workflow stage."""
    duration = info.get("duration")
    if not isinstance(duration, (int, float)):
        duration = None
    return {
        "requested_url": requested_url,
        "webpage_url": info.get("webpage_url") or requested_url,
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "id": info.get("id"),
        "title": info.get("title"),
        "duration_seconds": duration,
    }


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return a bounded exception chain without following cycles."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _cookie_load_failed(exc: BaseException, cookie_browser: str | None) -> bool:
    if cookie_browser is None:
        return False
    cookie_markers = (
        "failed to decrypt with dpapi",
        "could not copy chrome cookie database",
        "failed to load cookies",
    )
    for item in _exception_chain(exc):
        if type(item).__name__ == "CookieLoadError":
            return True
        if any(marker in str(item).casefold() for marker in cookie_markers):
            return True
    return False


def _classify_remote_error(
    exc: Exception,
    cookie_browser: str | None = None,
) -> FetchError:
    """Classify common yt-dlp failures without exposing signed URLs or headers."""
    text = " ".join(str(item).casefold() for item in _exception_chain(exc))
    if _cookie_load_failed(exc, cookie_browser):
        return FetchError(
            "cookie_unavailable",
            "Chrome cookies could not be read by this process.",
            exit_code=EXIT_ENVIRONMENT,
        )
    if any(
        marker in text
        for marker in (
            "ffmpeg not found",
            "ffprobe and ffmpeg not found",
        )
    ):
        return FetchError(
            "ffmpeg_unusable",
            "FFmpeg could not process the downloaded media.",
            exit_code=EXIT_ENVIRONMENT,
        )
    if any(marker in text for marker in ("sign in", "login", "private video", "cookie")):
        if cookie_browser is None:
            message = "The source requires authentication. Cookie access is disabled by default."
        else:
            message = "The source still requires authentication with the authorized Chrome session."
        return FetchError(
            "auth_required",
            message,
            exit_code=EXIT_REMOTE,
        )
    if any(marker in text for marker in ("geo", "not available in your country", "region")):
        return FetchError(
            "geo_restricted",
            "The source is not available in this region.",
            exit_code=EXIT_REMOTE,
        )
    if any(marker in text for marker in ("unsupported url", "no suitable extractor")):
        return FetchError(
            "unsupported_url",
            "yt-dlp does not support this URL.",
            exit_code=EXIT_REMOTE,
        )
    network_markers = (
        "timed out",
        "timeout",
        "temporary failure",
        "connection",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "too many requests",
        "service unavailable",
    )
    if any(marker in text for marker in network_markers):
        return FetchError(
            "network_error",
            "The remote request failed because of a network error.",
            exit_code=EXIT_REMOTE,
            retryable=True,
        )
    return FetchError(
        "download_failed",
        "yt-dlp could not fetch the requested source.",
        exit_code=EXIT_REMOTE,
    )


def _extract_metadata(
    ytdlp: ModuleType,
    url: str,
    cookie_browser: str | None,
) -> Mapping[str, Any]:
    options = {**_common_options(cookie_browser), "skip_download": True}
    try:
        with ytdlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise _classify_remote_error(exc, cookie_browser) from exc

    if not isinstance(info, Mapping):
        raise FetchError(
            "invalid_metadata",
            "yt-dlp returned an unsupported metadata object.",
            exit_code=EXIT_REMOTE,
        )
    if info.get("_type") in {"playlist", "multi_video"}:
        raise FetchError(
            "playlist_not_supported",
            "fetch accepts one video URL, not a playlist or channel.",
            exit_code=EXIT_REMOTE,
        )
    return info


def _is_vtt(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            prefix = stream.read(128)
    except OSError:
        return False
    return prefix.lstrip().startswith("WEBVTT")


def _download_caption(
    ytdlp: ModuleType,
    url: str,
    candidate: CaptionCandidate,
    attempt_dir: Path,
    cookie_browser: str | None,
) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    options = {
        **_common_options(cookie_browser),
        "skip_download": True,
        "writesubtitles": candidate.kind == "manual",
        "writeautomaticsub": candidate.kind == "automatic",
        "subtitleslangs": [candidate.language],
        "subtitlesformat": "vtt/best",
        "outtmpl": str(attempt_dir / "caption.%(ext)s"),
        "overwrites": True,
        "postprocessors": [
            {"key": "FFmpegSubtitlesConvertor", "format": "vtt", "when": "before_dl"}
        ],
    }
    try:
        with ytdlp.YoutubeDL(options) as downloader:
            return_code = downloader.download([url])
    except Exception as exc:
        classified = _classify_remote_error(exc, cookie_browser)
        if classified.code == "cookie_unavailable":
            raise classified from exc
        raise
    if return_code not in (None, 0):
        raise RuntimeError(f"yt-dlp returned {return_code}")

    for path in sorted(attempt_dir.glob("*.vtt")):
        if _is_vtt(path):
            return path
    raise RuntimeError("yt-dlp did not produce a valid VTT file")


def _download_audio(
    ytdlp: ModuleType,
    url: str,
    audio_dir: Path,
    which: Callable[[str], str | None],
    cookie_browser: str | None,
    validate_audio: Callable[[Path, str], bool],
) -> Path:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise FetchError(
            "missing_ffmpeg",
            "ffmpeg is required to produce audio.mp3; install it during explicit setup.",
            exit_code=EXIT_ENVIRONMENT,
        )

    audio_dir.mkdir(parents=True, exist_ok=False)
    options = {
        **_common_options(cookie_browser),
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "audio.%(ext)s"),
        "overwrites": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }
    try:
        with ytdlp.YoutubeDL(options) as downloader:
            return_code = downloader.download([url])
    except Exception as exc:
        raise _classify_remote_error(exc, cookie_browser) from exc
    if return_code not in (None, 0):
        raise FetchError(
            "download_failed",
            "yt-dlp could not produce the audio file.",
            exit_code=EXIT_REMOTE,
        )

    audio_path = audio_dir / "audio.mp3"
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FetchError(
            "audio_missing",
            "yt-dlp completed without producing audio.mp3.",
            exit_code=EXIT_REMOTE,
        )
    if not validate_audio(audio_path, ffmpeg):
        raise FetchError(
            "invalid_audio",
            "The downloaded audio could not be decoded completely.",
            exit_code=EXIT_REMOTE,
        )
    return audio_path


def _audio_decodes(audio_path: Path, ffmpeg: str) -> bool:
    """Decode the complete audio stream so truncated artifacts are never committed."""
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-i",
                str(audio_path),
                "-map",
                "0:a:0",
                "-f",
                "null",
                os.devnull,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise FetchError(
            "ffmpeg_unusable",
            "ffmpeg was found but could not be executed for audio validation.",
            exit_code=EXIT_ENVIRONMENT,
        ) from exc
    return completed.returncode == 0


def _commit(staged: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    targets = {name: output_dir / name for name in staged}
    conflicts = [str(path) for path in targets.values() if path.exists()]
    if conflicts:
        raise FetchError(
            "output_conflict",
            f"Refusing to overwrite existing output: {', '.join(conflicts)}",
            exit_code=EXIT_FILESYSTEM,
        )

    committed: dict[str, Path] = {}
    try:
        for name, source in staged.items():
            target = targets[name]
            source.replace(target)
            committed[name] = target.resolve()
    except OSError as exc:
        for name, target in reversed(tuple(committed.items())):
            try:
                target.replace(staged[name])
            except OSError:
                pass
        raise FetchError(
            "commit_failed",
            "The fetched artifacts could not be committed to the output directory.",
            exit_code=EXIT_FILESYSTEM,
        ) from exc
    return committed


def acquire(
    url: str,
    output_dir: Path,
    *,
    keep_audio: bool = False,
    cookie_browser: str | None = None,
    ytdlp: ModuleType | None = None,
    which: Callable[[str], str | None] = shutil.which,
    validate_audio: Callable[[Path, str], bool] = _audio_decodes,
) -> dict[str, Any]:
    """Fetch one URL and return the stable schema-v1 result."""
    _validate_url(url)
    if cookie_browser not in {None, "chrome"}:
        raise FetchError(
            "invalid_cookie_browser",
            "Only explicitly authorized Chrome cookie access is supported.",
            exit_code=EXIT_USAGE,
        )
    ytdlp = ytdlp or _load_ytdlp()

    try:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FetchError(
            "output_unavailable",
            "The output directory cannot be created or accessed.",
            exit_code=EXIT_FILESYSTEM,
        ) from exc

    for name in ("subtitle.vtt", "audio.mp3"):
        if (output_dir / name).exists():
            raise FetchError(
                "output_conflict",
                f"Refusing to overwrite existing output: {output_dir / name}",
                exit_code=EXIT_FILESYSTEM,
            )

    info = _extract_metadata(ytdlp, url, cookie_browser)
    source = _source_info(info, url)
    warnings: list[dict[str, Any]] = []

    try:
        staging_context = tempfile.TemporaryDirectory(prefix=".video-summary-", dir=output_dir)
    except OSError as exc:
        raise FetchError(
            "staging_failed",
            "A temporary staging directory could not be created.",
            exit_code=EXIT_FILESYSTEM,
        ) from exc

    with staging_context as staging_name:
        staging = Path(staging_name)
        selected: CaptionCandidate | None = None
        staged_subtitle: Path | None = None

        for index, candidate in enumerate(select_caption_candidates(info)):
            try:
                downloaded = _download_caption(
                    ytdlp,
                    url,
                    candidate,
                    staging / f"caption-{index}",
                    cookie_browser,
                )
            except FetchError:
                raise
            except Exception:  # noqa: BLE001 - third-party caption failures are non-fatal.
                warnings.append(
                    {
                        "code": "caption_download_failed",
                        "message": (
                            f"The {candidate.kind} {candidate.language} caption track failed; "
                            "the next permitted track was tried."
                        ),
                    }
                )
                continue
            selected = candidate
            staged_subtitle = staging / "subtitle.vtt"
            downloaded.replace(staged_subtitle)
            break

        staged_audio: Path | None = None
        if selected is None or keep_audio:
            staged_audio = _download_audio(
                ytdlp,
                url,
                staging / "audio",
                which,
                cookie_browser,
                validate_audio,
            )

        staged: dict[str, Path] = {}
        if staged_subtitle is not None:
            staged["subtitle.vtt"] = staged_subtitle
        if staged_audio is not None:
            staged["audio.mp3"] = staged_audio
        committed = _commit(staged, output_dir)

    subtitle = None
    if selected is not None:
        subtitle = {
            "path": str(committed["subtitle.vtt"]),
            "language": selected.language,
            "kind": selected.kind,
            "format": "vtt",
        }
    audio = None
    if "audio.mp3" in committed:
        audio = {"path": str(committed["audio.mp3"]), "format": "mp3"}

    if subtitle is not None and audio is not None:
        status = "subtitle_and_audio"
    elif subtitle is not None:
        status = "subtitle"
    else:
        status = "audio"

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "status": status,
        "next_action": (
            "build_transcript_from_subtitle" if subtitle is not None else "transcribe_audio"
        ),
        "source": source,
        "subtitle": subtitle,
        "audio": audio,
        "warnings": warnings,
        "error": None,
    }


def _error_result(url: str, error: FetchError) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "status": "error",
        "next_action": None,
        "source": {"requested_url": url},
        "subtitle": None,
        "audio": None,
        "warnings": [],
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }


def _serialize_result(result: Mapping[str, Any]) -> str:
    """Keep stdout ASCII-safe while preserving Unicode after JSON decoding."""
    return json.dumps(result, ensure_ascii=True)


def run(args: argparse.Namespace) -> int:
    """Execute parsed arguments and emit exactly one JSON result to stdout."""
    try:
        result = acquire(
            args.url,
            args.output_dir,
            keep_audio=args.keep_audio,
            cookie_browser=args.cookies_from_browser,
        )
    except FetchError as exc:
        print(_serialize_result(_error_result(args.url, exc)))
        return exc.exit_code
    except Exception:  # noqa: BLE001 - preserve the public JSON error contract.
        error = FetchError(
            "internal_error",
            "fetch failed because of an unexpected internal error.",
            exit_code=EXIT_INTERNAL,
        )
        print(_serialize_result(_error_result(args.url, error)))
        return error.exit_code

    print(_serialize_result(result))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except FetchError as exc:
        print(_serialize_result(_error_result("", exc)))
        return exc.exit_code
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

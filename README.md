# Video Summary

`video-summary` is a Codex/Claude Skill for caption-first video acquisition and
local speech-to-text on Windows and macOS. It prefers source captions, falls
back to audio download plus faster-whisper, and produces timestamped artifacts
that can be used for grounded summaries and content lookup.

## Features

- Fetch captions or audio from URLs supported by yt-dlp, including Bilibili.
- Prefer manual Chinese, automatic Chinese, manual English, and automatic
  English captions before local transcription.
- Transcribe local audio or video with `fast`, `balanced`, or `accurate`
  profiles.
- Produce WebVTT subtitles and detailed timestamped transcripts.
- Diagnose fetch, CPU, CUDA, FFmpeg, and output-directory readiness before a
  run.
- Keep Python dependencies isolated inside the Skill directory.

## Platform availability

| Platform | Architecture | URL fetch | CPU transcription | CUDA transcription | Verification status |
| --- | --- | --- | --- | --- | --- |
| macOS | Apple Silicon (`arm64`) | ✅ Available | ✅ Available | — Not supported on macOS | ✅ End-to-end verified on 2026-09-09 with a public Bilibili video |
| macOS | Intel (`x86_64`) | ✅ Supported | ✅ Supported | — Not supported on macOS | ⚠️ Covered by the lock file; not hardware-tested in the latest check |
| Windows | 64-bit (`AMD64`) | ✅ Supported | ✅ Supported | ⚠️ Available with a compatible NVIDIA GPU and CUDA 12 cuBLAS | ⚠️ Covered by the lock file and automated platform tests; not hardware-tested in the latest check |

`✅ Available` means the capability was exercised successfully on the listed
platform. `✅ Supported` means the repository provides the required platform
code and locked dependencies. `⚠️` identifies a hardware-dependent capability
or a platform that still needs a current physical-machine test.

The verified Apple Silicon run used Python 3.12, yt-dlp 2026.7.4,
faster-whisper 1.2.1, CTranslate2 4.8.1, FFmpeg, and the `fast` CPU profile. A
314-second video was transcribed in 22.3 seconds (approximately 15.9× real-time
speed). Performance varies by machine and model cache state.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe
- Network access for online media and first-time model downloads
- Optional on Windows: a compatible NVIDIA GPU and CUDA 12 cuBLAS for GPU
  transcription

macOS always uses CPU transcription. Do not install NVIDIA CUDA components on
macOS for this Skill.

## Setup

Set `SKILL_DIR` to the absolute path of `skills/video-summary`, then create the
locked, Skill-local environment:

```text
uv sync --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --locked --no-dev --python 3.12
```

Check the capabilities needed for a run:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary doctor --json --require fetch --require cpu --output-dir "<OUTPUT_DIR>"
```

## Usage

Fetch source captions or audio from an online video:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary fetch "<URL>" --output-dir "<OUTPUT_DIR>"
```

Transcribe a local audio or video file:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary transcribe "<LOCAL_PATH>" --output-dir "<OUTPUT_DIR>" --profile fast --device auto
```

Successful local transcription writes:

- `subtitle.vtt`
- `transcript_detailed.txt`

The `fast` profile favors speed. Domain-specific proper names and technical
terms may require the `balanced` or `accurate` profile, or manual review.

## Development checks

```text
uv sync --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --locked --group dev --python 3.12
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync ruff check "<SKILL_DIR>/scripts" tests
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync pytest
```

See [the Skill runtime documentation](skills/video-summary/README.md) and
[the Agent instructions](skills/video-summary/SKILL.md) for the complete
runtime and safety contract.

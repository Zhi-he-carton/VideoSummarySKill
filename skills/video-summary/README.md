# video-summary

Codex/Claude Skill for caption-first video acquisition and local faster-whisper
transcription on Windows and macOS.

## Runtime setup

Resolve `SKILL_DIR` to the directory containing `SKILL.md`. The same commands
work in PowerShell, Command Prompt, bash, and zsh:

```text
uv sync --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --locked --no-dev --python 3.12
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary doctor --json --require fetch
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary fetch "<URL>" --output-dir "<OUTPUT_DIR>"
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary transcribe "<AUDIO_OR_VIDEO>" --output-dir "<OUTPUT_DIR>" --profile fast --device auto
```

Python packages live in `<SKILL_DIR>/.venv`; downloads used to build that
environment live in `<SKILL_DIR>/.uv-cache`. Windows x86-64 can use CPU or a
verified NVIDIA CUDA runtime. Intel and Apple Silicon macOS use CPU because
CTranslate2 does not provide a CUDA backend there.

The `accurate` profile uses multilingual `large-v3` with GPU `int8_float16`,
batch size 8, and beam size 5. Every profile uses INT8 on CPU.

## Uninstall

The uninstaller is dependency-free and runs outside the Skill environment, so
it can remove the active virtual environment on Windows as well as macOS.
Preview first, then explicitly confirm removal:

```text
uv run --no-project --no-cache --python 3.12 "<SKILL_DIR>/scripts/uninstall.py" --json
uv run --no-project --no-cache --python 3.12 "<SKILL_DIR>/scripts/uninstall.py" --yes --json
```

Add `--include-models` to the confirmed command to remove this Skill's three
faster-whisper model directories too. Shared system tools (`uv`, FFmpeg,
FFprobe, CUDA, and GPU drivers), source files, and generated transcripts are
preserved because other applications may use them.

## Development checks

```text
uv sync --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --locked --group dev --python 3.12
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync ruff check "<SKILL_DIR>/scripts" tests
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync pytest
```

The lock file is resolved for Windows x86-64 plus Intel and Apple Silicon
macOS. Tests live at the repository root.

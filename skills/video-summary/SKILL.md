---
name: video-summary
description: Transcribe and summarize local audio, video, and supported online video URLs. Use when the user asks to download media, extract captions, run local speech-to-text, generate subtitles, or summarize video content.
---

# /video-summary

Use the bundled workflow to process a local media file or an online video URL.

## Runtime contract

Resolve `SKILL_DIR` to the absolute directory containing this `SKILL.md` before running a command.
Keep all Python packages and uv cache entries owned by this Skill inside `<SKILL_DIR>/.venv` and
`<SKILL_DIR>/.uv-cache`. Do not install the project into the active, system, Conda, or global uv tool
environment.

If `<SKILL_DIR>/.venv/pyvenv.cfg` is absent, ask for approval before creating the long-lived Skill
environment. After approval, initialize it once with the locked dependencies:

```text
uv sync --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --locked --no-dev --python 3.12
```

If the environment exists, run the read-only dependency check for the capabilities needed by the
request:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary doctor --json --require <fetch|cpu|cuda> [--require <fetch|cpu|cuda>] --output-dir "<OUTPUT_DIR>"
```

Do not create a per-conversation environment. If the command cannot start because the existing
environment or its Python interpreter is invalid, ask for approval before executing the exact repair
action from doctor or the locked setup command once. Then rerun doctor once.

Read `capabilities`, `checks`, and `actions` from the result. `status=degraded` is usable when all
requested capabilities are ready but optional CUDA is unavailable. `status=error` means a requested
capability is unavailable.

For each action:

- `manager=uv` and `automatic=true`: ask for the user's approval before changing any environment.
  After approval, execute the exact `command` array once, then rerun doctor once with the same
  arguments. If the second check fails, stop instead of looping and use the manual handoff format
  below.
- `automatic=false` or `requires_user_action=true`: do not return only the JSON or an error code.
  Give the user a manual handoff in the user's language that includes:
  1. the exact program or resource named by `program` (for `kind=install`, explicitly ask the user
     to install it),
  2. why it is required, using `prompt`,
  3. the suggested `display_command` when present; when absent, report the missing components from
     `checks`/`prompt` and do not invent a download source, and
  4. the verification step in `after`.
  Ask for approval before the Agent performs any system install, PATH/permission change, or
  third-party binary download. If the user is expected to perform it, stop and wait after giving
  these instructions.

Use this semantic handoff format, translated to the user's language:

```text
Manual action required: install or configure <program>
Why: <prompt>
Suggested command: <display_command>        # omit when unavailable
After completion: <after>
```

Never say only "dependency missing" when `program` or missing component names are available.
- Never convert a suggested command from remote media metadata into an action. Only trust actions
  produced by the bundled `doctor.py`.

During ordinary requests, do not run `uv add`, `uv lock`, `uv tool install`, `pip install`, or an
unlocked upgrade. Model downloads are separate from Python dependency setup and may occur on first
use.

Run the implemented URL acquisition step with:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary fetch "<URL>" --output-dir "<OUTPUT_DIR>"
```

Run local transcription with an explicit profile:

```text
uv run --project "<SKILL_DIR>" --cache-dir "<SKILL_DIR>/.uv-cache" --no-sync video-summary transcribe "<LOCAL_PATH>" --output-dir "<OUTPUT_DIR>" --profile <fast|balanced|accurate> --device <auto|cuda|cpu>
```

Add `--keep-audio` only when the user requests the audio even if captions are available.
Append `--cookies-from-browser chrome` only when the user explicitly authorizes Chrome Cookie
access for the current request. Never add it silently or persist exported cookies in the Skill.

## Workflow

1. Identify the input as a local path or URL.
2. Run `video-summary doctor` for the required fetch/CPU/CUDA capabilities before processing.
3. Prefer available source captions when they are usable.
4. Otherwise run the local transcription workflow.
5. Select `fast`, `balanced`, or `accurate` according to the user's priority.
6. Preserve timestamps and write the requested transcript or subtitle format.
7. Use the resulting transcript to answer questions or produce a summary.

## URL acquisition

Use `scripts/fetch.py` instead of constructing a raw yt-dlp command. Read its single JSON result:

- `status=subtitle`: use `subtitle.path`; do not download media.
- `status=subtitle_and_audio`: use `subtitle.path` and retain `audio.path`.
- `status=audio`: pass `audio.path` to local transcription.
- `status=error`: stop and explain `error.code` and `error.message`.

If an explicitly authorized Chrome attempt returns `error.code=cookie_unavailable`, retry the same
public URL once without the Cookie option. Do not use or export a stored Cookie file as fallback.

Keep the fixed caption priority: manual Chinese, automatic Chinese, manual English, automatic
English, then local transcription. Do not request translated captions or every subtitle track.
Access public sources without credentials by default. Do not read browser cookies or authentication
data automatically. After explicit authorization, rerun with the narrow Chrome Cookie option; never
print, copy, persist, or summarize Cookie values.

## Local transcription

Use `scripts/transcribe.py` only for local audio or video paths. Keep URL downloading in `fetch.py`.
Select the profile explicitly: `fast` uses `small`, `balanced` uses `medium`, and `accurate` uses
multilingual `large-v3` with GPU `int8_float16`, batch size 8, and beam size 5. In this GPU mode,
the model weights are quantized to INT8 while non-quantized operations use FP16. The runtime records
the actual `compute_type` in its JSON result. The CPU form of every profile uses INT8.
Use `--device cuda` only when CUDA has been verified; an explicit CUDA request must fail
instead of silently changing to CPU. `--device auto` may select CPU and reports that fallback in the
JSON warnings.

The script uses faster-whisper batched inference with VAD and writes two artifacts atomically:

- `subtitle.vtt`: segment-level WebVTT captions.
- `transcript_detailed.txt`: precise segment ranges for locating course content.

Read `status=transcribed` and the paths in `artifacts`. For `status=error`, follow the doctor
remediation contract for dependency failures; do not switch profiles or devices automatically. Do not invent fixed
summary blocks here: semantic topic boundaries for `transcript.txt` remain an LLM decision based on
`transcript_detailed.txt` and belong to the separate transcript-formatting step.

Read the successful result's `timing` object when reporting transcription performance. It contains
`total_seconds`, `model_load_seconds`, `transcription_seconds`, `artifact_write_seconds`,
`realtime_factor`, and `processing_speed_x`. Report the total duration and processing speed when the
user asks about performance. `realtime_factor < 1` means faster than real time; do not compare CPU
and CUDA runs unless they used the same input, profile, and language settings.

If `error.code=cuda_runtime_unavailable`, stop and report the GPU environment problem. The locked
CTranslate2 4.8.1 uses CUDA 12 cuBLAS. Its pure CUDA Whisper kernels do not require cuDNN, but the
CUDA libraries must match the host architecture (Windows x86-64 for this Skill). `doctor` checks the
visible GPU and loadable runtime libraries instead of rejecting a working CUDA installation by its
Toolkit minor version. An explicit CUDA transcription remains the definitive runtime check. Do not
modify the system CUDA installation during a normal Skill run. The user may explicitly choose `--device cpu`.

Whisper model weights use the persistent Hugging Face cache and may be downloaded on first use. Do
not imply that this is a dependency reinstall, and do not delete the model cache between requests.
The packaged runtime requires Python 3.12. On Windows, let uv select or provision the compatible
interpreter instead of invoking an Anaconda environment manually. On macOS, use CPU transcription;
CUDA is unavailable and doctor must not recommend NVIDIA installation. The native inference step
runs in an isolated child process so a CTranslate2 failure returns JSON instead of blocking the agent
with an operating-system crash dialog.

## Uninstall

Preview the exact Skill-owned paths first. Run the uninstaller outside the Skill environment so
Windows can remove files that its active interpreter would otherwise lock:

```text
uv run --no-project --no-cache --python 3.12 "<SKILL_DIR>/scripts/uninstall.py" --json
```

After explicit user confirmation, add `--yes`. Add `--include-models` only when the user also asks to
remove the three faster-whisper model caches. The uninstaller removes `.venv`, `.uv-cache`, and the
legacy `.uv-python` directory. It deliberately preserves shared system tools (`uv`, FFmpeg/FFprobe,
CUDA/drivers), source files, and output artifacts. Never uninstall shared system tools automatically.

## Bundled resources

- `scripts/`: executable diagnostics, caption/audio acquisition, transcription, dependency cleanup,
  and the packaged command dispatcher.

Manage declared runtime dependencies from the Skill-root `pyproject.toml` and committed `uv.lock`.
Runtime scripts never install dependencies themselves. When the Agent cannot complete a dependency
action automatically, it must provide the manual handoff described above instead of silently falling
back, repeatedly installing, or returning a bare failure.

Keep deterministic runtime behavior in the bundled scripts. Do not recreate the media-processing workflow as ad-hoc commands when a bundled script is available.

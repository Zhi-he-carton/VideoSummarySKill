"""Unified command-line entry point for the video-summary Skill.

The individual commands remain in their dedicated scripts so the Skill can
also call them directly when needed. This small dispatcher gives the isolated
project environment one stable entry point without a global installation.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_COMMANDS = {
    "doctor": "scripts.doctor",
    "fetch": "scripts.fetch",
    "transcribe": "scripts.transcribe",
}


def _usage() -> str:
    return (
        "Usage: video-summary <doctor|fetch|transcribe> [arguments]\n"
        "\n"
        "Commands:\n"
        "  doctor       Check Skill dependencies and runtime capabilities.\n"
        "  fetch        Prefer captions, otherwise download audio for transcription.\n"
        "  transcribe   Transcribe a local audio/video path with faster-whisper.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage(), end="")
        return 0

    command = arguments.pop(0)
    module_name = _COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr, end="")
        return 2

    module = importlib.import_module(module_name)
    return int(module.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())

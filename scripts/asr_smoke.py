"""Smoke test one local video against DashScope ASR.

Usage:
    .venv\\Scripts\\python.exe scripts\\asr_smoke.py data\\demo.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.asr import AsrError, transcribe_video  # noqa: E402
from app.srt import format_timestamp  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    video_path = Path(sys.argv[1]).resolve()
    try:
        result = transcribe_video(video_path, ROOT / "data" / "asr-smoke")
    except AsrError as exc:
        print(f"ASR failed: {exc} (code={exc.code})")
        return 1

    print(f"model={result.model} duration={result.duration_ms} words={len(result.words)}")
    print(result.text)
    for segment in result.segments:
        print(
            f"{format_timestamp(segment.start_ms)} --> {format_timestamp(segment.end_ms)}  "
            f"{segment.text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

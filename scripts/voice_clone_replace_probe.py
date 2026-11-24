"""Replace one segment of the test audio with the cloned voice.

Keeps the original audio before/after the target interval, and writes:
- segment-original.mp3
- segment-clone.mp3
- full-replaced.mp3
- preview-15-27s.mp3
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "tts-sdk"))

from dashscope import Client  # noqa: E402

FFMPEG = Path(r"D:\Users\AppData\Local\ffmpeg\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\Users\AppData\Local\ffmpeg\bin\ffprobe.exe")
OUT_DIR = Path(r"D:\code\agent_workspace\yuwen-publish-precheck\参考\voice-clone-test")
SOURCE = Path(r"D:\code\agent_workspace\yuwen-publish-precheck\参考\测试音频.mp3")
TARGET_START = 19.88
TARGET_END = 23.96
TARGET_SECONDS = TARGET_END - TARGET_START
REPLACEMENT_TEXT = "腹部包裹感更明显，具体感受因人而异。"


def load_api_key() -> str:
    for raw in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("DASHSCOPE_API_KEY") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("missing DASHSCOPE_API_KEY")


def duration_ms(path: Path) -> int:
    completed = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(float(completed.stdout.strip()) * 1000)


def download(url: str, target: Path) -> None:
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_bytes():
                if chunk:
                    output.write(chunk)


def run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(
        [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr[-800:])


def main() -> int:
    summary = json.loads((OUT_DIR / "result.json").read_text(encoding="utf-8"))
    model = summary["model"]
    voice_id = summary["voice_id"]
    client = Client(api_key=load_api_key(), timeout=90.0, max_retries=0)

    synthesis = client.audio.tts.http.synthesize(
        REPLACEMENT_TEXT,
        model=model,
        voice=voice_id,
        format="mp3",
        sample_rate=24000,
    )
    replacement = OUT_DIR / "segment-clone.mp3"
    download(str(synthesis.audio_url), replacement)
    replacement_ms = duration_ms(replacement)
    factor = min(1.8, max(0.7, replacement_ms / (TARGET_SECONDS * 1000)))

    original_segment = OUT_DIR / "segment-original.mp3"
    run_ffmpeg(
        [
            "-ss",
            f"{TARGET_START:.3f}",
            "-t",
            f"{TARGET_SECONDS:.3f}",
            "-i",
            str(SOURCE),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(original_segment),
        ]
    )

    full = OUT_DIR / "full-replaced.mp3"
    filter_complex = (
        f"[0:a]atrim=0:{TARGET_START:.3f},asetpts=PTS-STARTPTS[a0];"
        f"[1:a]atempo={factor:.4f},apad,atrim=0:{TARGET_SECONDS:.3f},asetpts=PTS-STARTPTS[a1];"
        f"[0:a]atrim={TARGET_END:.3f}:30.145,asetpts=PTS-STARTPTS[a2];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[out]"
    )
    run_ffmpeg(
        [
            "-i",
            str(SOURCE),
            "-i",
            str(replacement),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(full),
        ]
    )

    preview = OUT_DIR / "preview-15-27s.mp3"
    run_ffmpeg(
        [
            "-ss",
            "15.0",
            "-t",
            "12.0",
            "-i",
            str(full),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(preview),
        ]
    )

    result = {
        "voice_id": voice_id,
        "model": model,
        "target_start_seconds": TARGET_START,
        "target_end_seconds": TARGET_END,
        "replacement_text": REPLACEMENT_TEXT,
        "replacement_original_ms": replacement_ms,
        "atempo_factor": factor,
        "files": {
            "original_segment": str(original_segment),
            "cloned_segment": str(replacement),
            "full_replaced": str(full),
            "preview": str(preview),
        },
    }
    (OUT_DIR / "replace-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

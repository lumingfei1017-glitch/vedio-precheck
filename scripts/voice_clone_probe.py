"""One-off CosyVoice/Qwen-TTS voice-clone smoke test.

The reference audio is sent as a data URI; no OSS upload is needed for the
voice-enrollment endpoint. Outputs stay under the requested output directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "tts-sdk"))

from dashscope import Client  # noqa: E402

FFMPEG = Path(r"D:\Users\AppData\Local\ffmpeg\bin\ffmpeg.exe")


def load_api_key() -> str:
    for raw in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("DASHSCOPE_API_KEY") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("missing DASHSCOPE_API_KEY")


def trim_reference(source: Path, target: Path, start: float, duration: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not target.is_file():
        raise SystemExit(f"ffmpeg trim failed: {completed.stderr[-500:]}")


def wait_ready(client: Client, voice_id: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        voice = client.audio.voices.get(voice_id)
        status = str(getattr(voice, "status", "") or "")
        if status != last:
            print(f"voice status: {status}", flush=True)
            last = status
        if status == "OK":
            return
        if status and status != "DEPLOYING":
            raise SystemExit(f"voice unavailable: {status}")
        time.sleep(2)
    raise SystemExit("voice deployment timed out")


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_bytes():
                if chunk:
                    output.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cosyvoice-v3-plus")
    parser.add_argument(
        "--audio",
        default=r"D:\code\agent_workspace\yuwen-publish-precheck\参考\测试音频.mp3",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\code\agent_workspace\yuwen-publish-precheck\参考\voice-clone-test",
    )
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    reference = out_dir / "reference-20s.wav"
    trim_reference(Path(args.audio), reference, args.start, args.duration)
    audio_data = base64.b64encode(reference.read_bytes()).decode("ascii")
    data_uri = f"data:audio/wav;base64,{audio_data}"

    client = Client(api_key=load_api_key(), timeout=90.0, max_retries=0)
    print(f"creating clone with {args.model}", flush=True)
    voice = client.audio.voices.create(
        target_model=args.model,
        prefix="vptest",
        audio=data_uri,
    )
    voice_id = str(voice.voice_id)
    print(f"voice_id: {voice_id}", flush=True)
    wait_ready(client, voice_id)

    texts = [
        "姐妹们，这件塑身衣上身包裹感比较明显，腰腹这块看着更利落，穿着感受每个人不太一样。",
        "秋天到了，又到了穿塑身衣的季节。这件上身会更贴合，平时搭配高腰下装也比较省心。",
    ]
    results = []
    for index, text in enumerate(texts, start=1):
        synthesis = client.audio.tts.http.synthesize(
            text,
            model=args.model,
            voice=voice_id,
            format="mp3",
            sample_rate=24000,
        )
        audio_url = str(synthesis.audio_url)
        output = out_dir / f"clone-{index}.mp3"
        download(audio_url, output)
        results.append(
            {
                "index": index,
                "text": text,
                "output": str(output),
                "request_id": getattr(synthesis, "request_id", None),
            }
        )
        print(f"wrote {output}", flush=True)

    summary = {
        "model": args.model,
        "voice_id": voice_id,
        "reference": str(reference),
        "reference_start_seconds": args.start,
        "reference_duration_seconds": args.duration,
        "results": results,
    }
    (out_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

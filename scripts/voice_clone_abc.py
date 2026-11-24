"""Generate three local-replacement variants with the cloned voice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "tts-sdk"))
sys.path.insert(0, str(ROOT))

from dashscope import Client  # noqa: E402

from app.asr import transcribe_video  # noqa: E402
from app.srt import normalize  # noqa: E402
from voice_clone_replace_probe import (  # noqa: E402
    OUT_DIR,
    SOURCE,
    TARGET_END,
    TARGET_SECONDS,
    TARGET_START,
    download,
    duration_ms,
    load_api_key,
    run_ffmpeg,
)

ABC_DIR = OUT_DIR / "abc"
A_TEXT = "腹部包裹感更明显，具体感受因人而异。"
C_TEXT = "腹部这块的包裹感更明显，版型看着也更利落。"
B_TEXT = (
    "排扣一扣，拉链往上一拉。"
    "腹部这块的包裹感更明显，版型看着也更利落。"
    "来，姐妹啊，你肚子比较大的。"
)


def synthesize(client: Client, model: str, voice_id: str, text: str, rate: float, target: Path) -> str:
    """Prefer native rate control over post-hoc atempo."""
    try:
        audio = client.audio.tts.synthesize(
            text,
            model=model,
            voice=voice_id,
            format="mp3",
            sample_rate=24000,
            rate=rate,
        )
        if not audio:
            raise RuntimeError("empty websocket audio")
        target.write_bytes(audio)
        return "websocket-native-rate"
    except Exception as exc:  # noqa: BLE001
        result = client.audio.tts.http.synthesize(
            text,
            model=model,
            voice=voice_id,
            format="mp3",
            sample_rate=24000,
            parameters={"rate": rate},
        )
        if not result.audio_url:
            raise RuntimeError(f"TTS failed and HTTP fallback has no audio: {exc}")
        download(str(result.audio_url), target)
        return "http-post-fit"


def fit_segment(source: Path, target: Path) -> tuple[float, int]:
    source_ms = duration_ms(source)
    factor = 1.0
    if source_ms > TARGET_SECONDS * 1000 * 1.02:
        factor = min(1.15, source_ms / (TARGET_SECONDS * 1000))
    run_ffmpeg(
        [
            "-i",
            str(source),
            "-filter:a",
            f"atempo={factor:.4f},apad,atrim=0:{TARGET_SECONDS:.3f}",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(target),
        ]
    )
    return factor, source_ms


def build_full(replacement: Path, target: Path) -> None:
    replacement_ms = duration_ms(replacement)
    factor = 1.0
    if replacement_ms > TARGET_SECONDS * 1000 * 1.02:
        factor = min(1.15, replacement_ms / (TARGET_SECONDS * 1000))
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
            str(target),
        ]
    )


def find_replacement_span(audio: Path) -> tuple[int, int, str]:
    result = transcribe_video(audio, ABC_DIR / "b-asr")
    segments = result.segments
    start_index = None
    end_index = None
    for index, segment in enumerate(segments):
        text = normalize(segment.text)
        if start_index is None and "腹部" in text:
            start_index = index
        if start_index is not None and ("利落" in text or "版型" in text):
            end_index = index
            break
    if start_index is None:
        raise RuntimeError(f"B 版找不到替换片段的起点：{result.text}")
    if end_index is None:
        end_index = min(len(segments) - 1, start_index + 1)
    start_ms = segments[start_index].start_ms
    end_ms = segments[end_index].end_ms
    return start_ms, end_ms, result.text


def make_preview(full: Path, target: Path) -> None:
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
            str(target),
        ]
    )


def main() -> int:
    ABC_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT_DIR / "result.json").read_text(encoding="utf-8"))
    model = summary["model"]
    voice_id = summary["voice_id"]
    client = Client(api_key=load_api_key(), timeout=120.0, max_retries=0)

    results: dict[str, dict] = {}

    # A: natural-rate short copy, no context.
    a_raw = ABC_DIR / "A-raw.mp3"
    a_method = synthesize(client, model, voice_id, A_TEXT, 1.0, a_raw)
    a_segment = ABC_DIR / "A-segment.mp3"
    a_factor, a_raw_ms = fit_segment(a_raw, a_segment)
    a_full = ABC_DIR / "A-full.mp3"
    build_full(a_segment, a_full)
    a_preview = ABC_DIR / "A-preview.mp3"
    make_preview(a_full, a_preview)
    results["A"] = {
        "text": A_TEXT,
        "method": a_method,
        "raw_ms": a_raw_ms,
        "fit_factor": a_factor,
        "segment": str(a_segment),
        "full": str(a_full),
        "preview": str(a_preview),
    }

    # B: synthesize with surrounding context, then crop the target phrase.
    b_raw = ABC_DIR / "B-raw.mp3"
    b_method = synthesize(client, model, voice_id, B_TEXT, 1.0, b_raw)
    b_start, b_end, b_transcript = find_replacement_span(b_raw)
    b_segment = ABC_DIR / "B-segment.mp3"
    run_ffmpeg(
        [
            "-ss",
            f"{b_start / 1000:.3f}",
            "-t",
            f"{(b_end - b_start) / 1000:.3f}",
            "-i",
            str(b_raw),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(b_segment),
        ]
    )
    b_full = ABC_DIR / "B-full.mp3"
    build_full(b_segment, b_full)
    b_preview = ABC_DIR / "B-preview.mp3"
    make_preview(b_full, b_preview)
    results["B"] = {
        "text": B_TEXT,
        "method": b_method,
        "raw_ms": duration_ms(b_raw),
        "crop_start_ms": b_start,
        "crop_end_ms": b_end,
        "asr_text": b_transcript,
        "segment": str(b_segment),
        "full": str(b_full),
        "preview": str(b_preview),
    }

    # C: longer copy at native rate candidates, choose the closest duration.
    c_candidates = []
    for rate in (0.95, 1.0, 1.05):
        raw = ABC_DIR / f"C-raw-{rate:.2f}.mp3"
        method = synthesize(client, model, voice_id, C_TEXT, rate, raw)
        candidate = {
            "rate": rate,
            "method": method,
            "raw_ms": duration_ms(raw),
            "raw": str(raw),
        }
        c_candidates.append(candidate)
    best = min(c_candidates, key=lambda item: abs(item["raw_ms"] - TARGET_SECONDS * 1000))
    c_segment = ABC_DIR / "C-segment.mp3"
    c_factor, c_raw_ms = fit_segment(Path(best["raw"]), c_segment)
    c_full = ABC_DIR / "C-full.mp3"
    build_full(c_segment, c_full)
    c_preview = ABC_DIR / "C-preview.mp3"
    make_preview(c_full, c_preview)
    results["C"] = {
        "text": C_TEXT,
        "candidates": c_candidates,
        "chosen_rate": best["rate"],
        "fit_factor": c_factor,
        "raw_ms": c_raw_ms,
        "segment": str(c_segment),
        "full": str(c_full),
        "preview": str(c_preview),
    }

    output = {
        "voice_id": voice_id,
        "model": model,
        "target_start_seconds": TARGET_START,
        "target_end_seconds": TARGET_END,
        "target_seconds": TARGET_SECONDS,
        "variants": results,
    }
    (ABC_DIR / "abc-result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

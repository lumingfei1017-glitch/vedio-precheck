from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.asr import AsrError, transcribe_video  # noqa: E402

VIDEO_DIR = ROOT.parent / "参考" / "审核建议"
ASR_DIR = ROOT / "data" / "ref-eval" / "asr"


def transcribe_one(video_id: str) -> tuple[str, str | None]:
    cache_path = ASR_DIR / f"{video_id}.json"
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("segments"):
                return video_id, None
        except (json.JSONDecodeError, OSError):
            pass
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    try:
        result = transcribe_video(video_path, ASR_DIR / video_id)
        payload = {
            "id": video_id,
            "text": result.text,
            "duration_ms": result.duration_ms,
            "request_id": result.request_id,
            "model": result.model,
            "segments": [
                {
                    "index": segment.index,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "text": segment.text,
                }
                for segment in result.segments
            ],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return video_id, None
    except AsrError as exc:
        cache_path.write_text(
            json.dumps({"id": video_id, "error": str(exc), "code": exc.code}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return video_id, str(exc)


def main() -> int:
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    video_ids = sorted(path.stem for path in VIDEO_DIR.glob("*.mp4"))
    missing = [
        video_id
        for video_id in video_ids
        if not (ASR_DIR / f"{video_id}.json").is_file()
    ]
    print(json.dumps({"videos": len(video_ids), "missing": len(missing)}, ensure_ascii=True))
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(transcribe_one, video_id): video_id for video_id in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            video_id, error = future.result()
            print(f"[{index}/{len(missing)}] {video_id} {'ERROR ' + error if error else 'ok'}", flush=True)
            if error:
                errors.append({"video_id": video_id, "error": error})
    print(json.dumps({"completed": len(missing), "errors": errors}, ensure_ascii=True, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

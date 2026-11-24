"""镜头切分与类型判定。

Wan 只能整镜重生成，替换粒度必须对齐原片切点，所以这里先把素材拆成镜头清单，
再按“镜头时长 + 嘴是否可见 + 违规覆盖率”决定走 Wan、TTS 还是人工。

2026-11-21 实测（4 条素材 / 79 个镜头 / 171 条平台审核引文）：
- 违规点中位 2.13s，但含违规的镜头跨度中位 10.35s；
- 镜头中位 2.70s，35% 短于 2s，最长 13.07s。
所以 Wan 的单次生成不会超时长，但短镜与无脸镜走 Wan 没有意义：
Wan 最小输出 2s，短于 2s 的镜头提交即报错；无脸镜要求口型同步也会带偏画面。
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import FFMPEG_PATH, MERGE_SCENE_SCORE_MAX, WAN_MERGE_MAX_SECONDS
from .srt import Segment

FaceProbe = Callable[[Path, int, int], tuple[float | None, float | None]]

KIND_LABELS = {
    "talking_closeup": "口播特写",
    "talking_full": "口播全身",
    "talking_side": "侧身口播",
    "voiceover": "画外音/演示",
    "broll": "无口播/特写",
    "unknown": "待判定",
}

WAN_MIN_SECONDS = 2.0
WAN_FULL_COVERAGE = 0.6
WAN_PARTIAL_COVERAGE = 0.4
_SCENE_SCORE_CACHE: dict[tuple[str, int], dict[int, float]] = {}


def scene_scores(video: Path) -> dict[int, float]:
    """整片取一次 ffmpeg scene 分数，键为毫秒时间戳。"""
    if FFMPEG_PATH is None:
        return {}
    try:
        stat = Path(video).stat()
    except OSError:
        return {}
    key = (str(video), int(stat.st_mtime_ns))
    if key in _SCENE_SCORE_CACHE:
        return _SCENE_SCORE_CACHE[key]
    completed = subprocess.run(
        [
            str(FFMPEG_PATH), "-hide_banner", "-i", str(video),
            "-vf", "select='gte(scene,0)',metadata=print:file=-",
            "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        errors="ignore",
        timeout=300,
        check=False,
    )
    scores: dict[int, float] = {}
    current: float | None = None
    for line in (completed.stdout + completed.stderr).splitlines():
        found = re.search(r"pts_time:([0-9.]+)", line)
        if found:
            current = float(found.group(1))
            continue
        found = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if found and current is not None:
            scores[int(round(current * 1000))] = float(found.group(1))
    _SCENE_SCORE_CACHE[key] = scores
    return scores


def scene_score_at(video: Path, t_ms: int, *, window_ms: int = 80) -> float | None:
    scores = scene_scores(video)
    near = [value for stamp, value in scores.items() if abs(stamp - t_ms) <= window_ms]
    return max(near) if near else None


def can_merge_across(video: Path, cut_ms: int, *, limit: float = MERGE_SCENE_SCORE_MAX) -> bool:
    score = scene_score_at(video, cut_ms)
    return score is not None and score <= limit


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    coverage: float | None = None


def decide_route(
    kind: str,
    seconds: float,
    chars: int,
    coverage: float | None = None,
) -> RouteDecision:
    """Wan 主路径，TTS 只接短镜、无脸镜和被局部改动的镜头。

    coverage 是违规区间占该镜头时长的比例，未知时按整镜处理。
    """
    if chars <= 0:
        return RouteDecision("keep", "本镜头没有口播")
    if kind == "unknown":
        return RouteDecision("manual", "人脸检测不可用，需要人工确认口播方式", coverage)
    if seconds < WAN_MIN_SECONDS:
        return RouteDecision(
            "tts",
            f"镜头 {seconds:.1f}s 短于 Wan 最短 {WAN_MIN_SECONDS:.0f}s，只能局部换音轨",
            coverage,
        )
    if kind == "voiceover":
        return RouteDecision("tts", "画外音/演示镜头没有口型，保持画面不动", coverage)
    # 侧身镜头的口型仍可见，Wan 重生成的收益大于风险，统一定为可重生成。
    if coverage is not None and coverage < WAN_PARTIAL_COVERAGE:
        return RouteDecision(
            "tts",
            f"违规只覆盖本镜头 {coverage:.0%}，局部补音不动画面",
            coverage,
        )
    return RouteDecision("wan", "口播特写且违规覆盖整镜，整镜重生成", coverage)


class ShotError(RuntimeError):
    """镜头切分失败。"""


@dataclass(frozen=True)
class Shot:
    start_ms: int
    end_ms: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text.strip())

    @property
    def seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000

    @property
    def kind(self) -> str:
        return classify_shot(self.chars, self.face_hit, self.face_ratio)

    @property
    def decision(self) -> RouteDecision:
        return decide_route(self.kind, self.seconds, self.chars, self.coverage)

    @property
    def route(self) -> str:
        return self.decision.route

    @property
    def route_reason(self) -> str:
        return self.decision.reason

    face_hit: float | None = None
    face_ratio: float | None = None
    coverage: float | None = None

    def as_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "seconds": round(self.seconds, 2),
            "chars": self.chars,
            "face_hit": self.face_hit,
            "face_ratio": self.face_ratio,
            "kind": self.kind,
            "route": self.route,
            "route_reason": self.route_reason,
            "coverage": self.coverage,
            "text": self.text,
        }


def route_for_span(shot: Shot, start_ms: int, end_ms: int) -> RouteDecision:
    """按违规区间在该镜头内的占比给出路由。"""
    span = max(1, shot.end_ms - shot.start_ms)
    overlap = max(0, min(end_ms, shot.end_ms) - max(start_ms, shot.start_ms))
    coverage = min(1.0, overlap / span)
    return decide_route(shot.kind, shot.seconds, shot.chars, coverage)


def classify_shot(chars: int, face_hit: float | None, face_ratio: float | None) -> str:
    if chars <= 0:
        return "broll"
    if face_hit is None:
        return "unknown"
    ratio = face_ratio or 0.0
    if face_hit >= 0.5 and ratio >= 0.06:
        return "talking_closeup"
    if face_hit >= 0.5:
        return "talking_full"
    if face_hit >= 0.1:
        return "talking_side"
    return "voiceover"


def detect_cuts_ms(video: Path, threshold: float = 0.3) -> list[int]:
    if FFMPEG_PATH is None:
        raise ShotError("未找到 ffmpeg，无法切分镜头")
    completed = subprocess.run(
        [
            str(FFMPEG_PATH), "-hide_banner", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        errors="ignore",
        timeout=300,
        check=False,
    )
    cuts: list[int] = []
    for line in completed.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            cuts.append(int(round(float(line.split("pts_time:")[1].split()[0]) * 1000)))
        except (IndexError, ValueError):
            continue
    return cuts


def default_face_probe(video: Path, start_ms: int, end_ms: int, fps: float = 2.0) -> tuple[float | None, float | None]:
    """用 OpenCV Haar 粗测人脸命中率和占比；没装 cv2 时返回 None 交给人工/VLM。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, None
    # OpenCV 5 移除了 Haar 级联，退化为“待判定”，由人工或 VLM 确认。
    if not hasattr(cv2, "CascadeClassifier"):
        return None, None

    span = max(0.1, (end_ms - start_ms) / 1000)
    completed = subprocess.run(
        [
            str(FFMPEG_PATH), "-v", "error", "-ss", f"{start_ms / 1000:.3f}", "-t", f"{span:.3f}",
            "-i", str(video), "-vf", f"fps={fps},scale=360:-2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
        timeout=180,
        check=False,
    )
    width, height = 360, int(round(360 * 16 / 9))
    size = width * height * 3
    count = len(completed.stdout) // size
    if count == 0:
        return 0.0, 0.0
    frames = np.frombuffer(completed.stdout[: count * size], dtype=np.uint8).reshape(count, height, width, 3)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
        area = max((w * h for (_, _, w, h) in faces), default=0)
        ratios.append(area / float(height * width))
    hit = sum(1 for value in ratios if value > 0) / len(ratios)
    ordered = sorted(ratios)
    median = ordered[len(ordered) // 2]
    return hit, median


def build_manifest(
    video: Path,
    segments: list[Segment],
    *,
    duration_ms: int | None = None,
    threshold: float = 0.3,
    cuts_ms: list[int] | None = None,
    face_probe: FaceProbe | None = None,
) -> list[Shot]:
    cuts = cuts_ms if cuts_ms is not None else detect_cuts_ms(video, threshold)
    total = duration_ms or max((segment.end_ms for segment in segments), default=0) + 1000
    boundaries = sorted({0, *[cut for cut in cuts if 0 < cut < total], total})
    probe = face_probe or default_face_probe
    shots: list[Shot] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 400:
            continue
        text = "".join(
            segment.text for segment in segments if segment.end_ms > start and segment.start_ms < end
        )
        hit, ratio = probe(video, start, end)
        shots.append(Shot(start_ms=start, end_ms=end, text=text, face_hit=hit, face_ratio=ratio))
    return shots


def manifest_signature(video: Path, segments: list[Segment]) -> str:
    stat = video.stat()
    return f"{stat.st_size}-{int(stat.st_mtime)}-{len(segments)}-{sum(len(s.text) for s in segments)}"


def load_manifest_cache(cache_file: Path, signature: str) -> list[Shot] | None:
    if not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("signature") != signature:
        return None
    return [
        Shot(
            start_ms=item["start_ms"],
            end_ms=item["end_ms"],
            text=item.get("text", ""),
            face_hit=item.get("face_hit"),
            face_ratio=item.get("face_ratio"),
            coverage=item.get("coverage"),
        )
        for item in payload.get("shots", [])
    ]


def cached_manifest(
    video: Path,
    segments: list[Segment],
    duration_ms: int,
    cache_file: Path,
    *,
    refresh: bool = False,
    face_probe: FaceProbe | None = None,
    threshold: float = 0.3,
) -> list[Shot]:
    signature = manifest_signature(video, segments)
    if not refresh:
        cached = load_manifest_cache(cache_file, signature)
        if cached is not None:
            return cached
    shots = build_manifest(
        video,
        segments,
        duration_ms=duration_ms,
        threshold=threshold,
        face_probe=face_probe,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {"signature": signature, "shots": [shot.as_dict() for shot in shots]},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return shots

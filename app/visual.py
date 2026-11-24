from __future__ import annotations

import base64
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .asr import probe_duration_ms
from .config import (
    DASHSCOPE_API_KEY,
    FFMPEG_PATH,
    VISUAL_ENABLED,
    VISUAL_ENDPOINT,
    VISUAL_FRAME_WIDTH,
    VISUAL_MAX_FRAMES,
    VISUAL_MAX_TOKENS,
    VISUAL_MODEL,
    VISUAL_OWN_BRANDS,
    VISUAL_RELOCALIZE_ENABLED,
    VISUAL_RELOCALIZE_MAX_EVENTS,
    VISUAL_RELOCALIZE_MAX_FRAMES,
    VISUAL_RELOCALIZE_RADIUS_MS,
    VISUAL_RELOCALIZE_STEP_MS,
    VISUAL_TIMEOUT_SECONDS,
)


class VisualError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VisualEvidence:
    description: str
    start_ms: int
    end_ms: int
    frame_ms: int
    frame_index: int


@dataclass(frozen=True)
class VisualEvent:
    rule_id: str
    title: str
    category: str
    severity: str
    evidence: tuple[VisualEvidence, ...]
    reason: str
    suggestion: str
    confidence: float


VISUAL_PROMPT = """你是抖音千川服装配饰（塑身衣）广告视觉审核专家。
只根据画面判断，不依赖口播。重点检查：
1. 低俗擦边：敏感部位特写、低角度、透视/暴露、内裤痕迹、刻意突出臀部或胸部。
2. 品牌混淆：未授权品牌 logo、商标、包装、与知名品牌相似的视觉元素。
3. 夸张性能演示：极端拉伸、承重、暴力按压等展示。
4. 身材对比演示：穿前穿后、腹部/臀部局部对比、刻意展示身材变化。
5. 其他明显违反服装配饰广告规范的画面。

中性产品展示、正常上身展示、普通面料细节不算违规。不要因为人物穿着紧身衣就判违规。

只输出 JSON：
{
  "events": [
    {
      "rule_id": "VIS-LOW-SUGGESTIVE|VIS-BRAND|VIS-EXTREME-STRETCH|VIS-BODY-COMPARISON|VIS-RISK",
      "title": "简短中文标题",
      "category": "不良导向/低俗擦边|未授权品牌/商标混淆|夸大功效/虚假塑身效果|其他视觉风险",
      "severity": "blocking|suggest|info",
      "frame_index": 0,
      "description": "画面里具体看到了什么",
      "reason": "为什么构成或可能构成违规",
      "suggestion": "删除/替换/遮挡/补授权等修改建议",
      "confidence": 0.0
    }
  ]
}
没有违规时输出 {"events": []}。"""


def review_visual(video_path: Path, work_dir: Path | None = None) -> list[VisualEvent]:
    if not VISUAL_ENABLED:
        return []
    if not video_path.is_file():
        raise VisualError("视频文件不存在")
    if FFMPEG_PATH is None:
        raise VisualError("未找到 FFmpeg")
    api_key = DASHSCOPE_API_KEY
    if not api_key:
        raise VisualError("未配置 DASHSCOPE_API_KEY")

    duration_ms = probe_duration_ms(video_path)
    if duration_ms <= 0:
        raise VisualError("无法读取视频时长")
    frame_dir = (work_dir or video_path.parent) / "frames"
    frames = _extract_frames(video_path, duration_ms, frame_dir)
    if not frames:
        raise VisualError("关键帧抽取失败")

    own_brands = "、".join(VISUAL_OWN_BRANDS) or "无"
    prompt = (
        f"{VISUAL_PROMPT}\n\n广告主自身品牌：{own_brands}。"
        "画面或字幕只出现自身品牌名不算品牌混淆；只有第三方品牌 logo、商标、"
        "包装或明显模仿才属于 VIS-BRAND。"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, (timestamp_ms, frame_path) in enumerate(frames):
        content.append(
            {
                "type": "text",
                "text": f"帧 {index} · {timestamp_ms / 1000:.1f} 秒",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(frame_path.read_bytes()).decode()}"
                },
            }
        )

    payload = {
        "model": VISUAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": VISUAL_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    content_text = _call_visual(api_key, payload)
    data = _parse_json(content_text)
    raw_events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(raw_events, list):
        raise VisualError("视觉审核返回缺少 events 数组")

    step_ms = max(1, duration_ms // max(1, len(frames)))
    events: list[VisualEvent] = []
    seen: set[tuple[str, int]] = set()
    for item in raw_events[:12]:
        if not isinstance(item, dict):
            continue
        try:
            frame_index = int(item.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or frame_index >= len(frames):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        category = str(item.get("category") or "其他视觉风险").strip()
        if "品牌" in category and _is_own_brand_description(description):
            continue
        key = (category, frame_index)
        if key in seen:
            continue
        seen.add(key)

        frame_ms = frames[frame_index][0]
        start_ms = max(0, frame_ms - step_ms // 2)
        end_ms = min(duration_ms, frame_ms + step_ms // 2)
        severity = str(item.get("severity") or "blocking").lower()
        if severity not in {"blocking", "suggest", "info"}:
            severity = "blocking"
        events.append(
            VisualEvent(
                rule_id=str(item.get("rule_id") or "VIS-RISK"),
                title=str(item.get("title") or "视觉审核风险"),
                category=category,
                severity=severity,
                evidence=(
                    VisualEvidence(
                        description=description,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        frame_ms=frame_ms,
                        frame_index=frame_index,
                    ),
                ),
                reason=str(item.get("reason") or ""),
                suggestion=str(item.get("suggestion") or "删除或替换该画面，改用产品细节或正常穿着展示。"),
                confidence=_clamp_float(item.get("confidence"), 0.8),
            )
        )
    events = _merge_adjacent_visual_events(events)
    if VISUAL_RELOCALIZE_ENABLED and events:
        events = _relocalize_visual_events(video_path, duration_ms, events, api_key, work_dir)
    return events


RELOCALIZE_PROMPT = """下面是从同一段视频里密集抽取的帧，相邻两帧间隔 {step:.1f} 秒，第一帧时间 {start:.1f} 秒。
请只做定位，只找**清晰、无歧义**出现下面这个具体特征的帧：

特征：{desc}

要求：
- 只返回最清楚的 1–3 帧，不要返回一整段；
- 画面里出现同类物品但该特征看不清、被遮挡或角度不对的，不要选；
- 没有一帧清楚符合时返回空数组。

只输出 JSON：{{"frames": [0]}} 或 {{"frames": []}}。不要重新判断是否违规，也不要描述画面。"""

# 按规则特征定位，而不是照抄模型写的描述。实测描述会把"背景老花面料"
# 说成"塑身衣内侧蕾丝"，按描述找会回到错误的帧。
LOCATOR_HINTS = {
    "VIS-BRAND": "第三方品牌 logo、商标、monogram 老花图案",
    "VIS-BODY-COMPARISON": "穿前穿后对比、左右分屏、刻意展示身材变化",
    "VIS-LOW-SUGGESTIVE": "敏感部位特写、低角度、透视暴露",
    "VIS-EXTREME-STRETCH": "极端拉伸、承重、暴力按压演示",
}


def _locator_text(event: VisualEvent) -> str:
    hint = LOCATOR_HINTS.get(event.rule_id)
    if hint:
        return hint
    return "；".join(dict.fromkeys(item.description for item in event.evidence))[:200]


def _relocalize_visual_events(
    video_path: Path,
    duration_ms: int,
    events: list[VisualEvent],
    api_key: str,
    work_dir: Path | None,
) -> list[VisualEvent]:
    """密集复核视觉事件的时间位置。

    粗扫每 5–6 秒一帧，事件时间窗等于一整个采样桶，模型很容易把某个画面
    记到相邻帧上（实测把 33s 的图案标成 22.5–28.1s）。这里以事件为中心
    重新按 0.5 秒抽帧，让模型指认真正出现该画面的帧。
    """
    limit = max(0, VISUAL_RELOCALIZE_MAX_EVENTS)
    refined: list[VisualEvent] = []
    for event in events[:limit]:
        reliable, located = _locate_event(video_path, duration_ms, event, api_key, work_dir)
        if not reliable:
            refined.append(event)
        elif not located:
            # 复核没复现不等于一定误报：可能是抽帧角度没覆盖到。保留并标注，
            # 交给人工看，避免把真问题静默删掉。
            refined.append(_event_with_note(event, "密集复核未在该时段复现，时间以抽样帧为准"))
        else:
            refined.append(_event_with_spans(event, located))
    refined.extend(events[limit:])
    return sorted(refined, key=lambda item: min(e.start_ms for e in item.evidence))


def _locate_event(
    video_path: Path,
    duration_ms: int,
    event: VisualEvent,
    api_key: str,
    work_dir: Path | None,
) -> tuple[bool, list[tuple[int, int]] | None]:
    """返回 (是否可用, 定位结果)。

    (False, None) 表示复核失败，保留原始事件；(True, None) 表示复核成功但画面
    没有出现，事件应当丢弃；(True, [(start, end), ...]) 表示定位成功，可能有多处。
    """
    base_start = min(item.start_ms for item in event.evidence)
    base_end = max(item.end_ms for item in event.evidence)
    window_start = max(0, base_start - VISUAL_RELOCALIZE_RADIUS_MS)
    window_end = min(duration_ms, base_end + VISUAL_RELOCALIZE_RADIUS_MS)
    step = max(100, VISUAL_RELOCALIZE_STEP_MS)
    max_frames = max(8, VISUAL_RELOCALIZE_MAX_FRAMES)
    if window_end - window_start > step * max_frames:
        # 帧数过多会让请求过大、模型返回失败；以事件中心为准截到上限。
        center = (base_start + base_end) // 2
        half = step * max_frames // 2
        window_start = max(0, center - half)
        window_end = min(duration_ms, window_start + step * max_frames)
        window_start = max(0, window_end - step * max_frames)
    if window_end - window_start < step * 4:
        return False, None
    frames = _extract_dense_frames(video_path, work_dir, window_start, window_end)
    if not frames:
        return False, None

    description = _locator_text(event)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": RELOCALIZE_PROMPT.format(
                step=VISUAL_RELOCALIZE_STEP_MS / 1000,
                start=window_start / 1000,
                desc=description[:400],
            ),
        }
    ]
    for index, (timestamp_ms, frame_path) in enumerate(frames):
        content.append({"type": "text", "text": f"帧 {index} · {timestamp_ms / 1000:.1f} 秒"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(frame_path.read_bytes()).decode()}"
                },
            }
        )
    payload = {
        "model": VISUAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    try:
        data = _parse_json(_call_visual(api_key, payload))
    except (VisualError, ValueError, TypeError):
        return False, None
    raw = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return False, None
    indexes = sorted(
        {
            int(value)
            for value in raw
            if isinstance(value, (int, float, str)) and str(value).strip().lstrip("-").isdigit()
        }
    )
    indexes = [index for index in indexes if 0 <= index < len(frames)]
    if not indexes:
        return True, None
    return True, _cluster_spans(frames, indexes, duration_ms)


def _cluster_spans(
    frames: list[tuple[int, Path]],
    indexes: list[int],
    duration_ms: int,
) -> list[tuple[int, int]]:
    """把命中帧按连续段分组，返回每一段的起止时间。

    同一元素可能在片子里出现多次（实测双 G 内衬在 25s 和 33s 各出现一次），
    只保留一处会让后续删段漏掉另一半。
    """
    runs: list[list[int]] = []
    for index in indexes:
        if runs and index - runs[-1][-1] <= 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    step = frames[1][0] - frames[0][0] if len(frames) > 1 else VISUAL_RELOCALIZE_STEP_MS
    spans: list[tuple[int, int]] = []
    for run in runs:
        start = frames[run[0]][0]
        end = min(duration_ms, frames[run[-1]][0] + step)
        if end <= start:
            end = min(duration_ms, start + step)
        spans.append((start, end))
    return spans[:8]


def _event_with_spans(event: VisualEvent, spans: list[tuple[int, int]]) -> VisualEvent:
    description = "；".join(dict.fromkeys(item.description for item in event.evidence))
    return VisualEvent(
        rule_id=event.rule_id,
        title=event.title,
        category=event.category,
        severity=event.severity,
        evidence=tuple(
            VisualEvidence(
                description=description,
                start_ms=start_ms,
                end_ms=end_ms,
                frame_ms=(start_ms + end_ms) // 2,
                frame_index=event.evidence[0].frame_index,
            )
            for start_ms, end_ms in spans
        ),
        reason=event.reason,
        suggestion=event.suggestion,
        confidence=event.confidence,
    )


def _event_with_note(event: VisualEvent, note: str) -> VisualEvent:
    reason = f"{event.reason}；{note}" if event.reason else note
    return VisualEvent(
        rule_id=event.rule_id,
        title=event.title,
        category=event.category,
        severity=event.severity,
        evidence=event.evidence,
        reason=reason,
        suggestion=event.suggestion,
        confidence=event.confidence,
    )


def _extract_dense_frames(
    video_path: Path,
    work_dir: Path | None,
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, Path]]:
    if FFMPEG_PATH is None:
        return []
    frame_dir = (work_dir or video_path.parent) / "relocalize"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("r_*.jpg"):
        old.unlink(missing_ok=True)
    step = max(100, VISUAL_RELOCALIZE_STEP_MS)
    completed = subprocess.run(
        [
            str(FFMPEG_PATH), "-y", "-v", "error",
            "-ss", f"{start_ms / 1000:.3f}",
            "-t", f"{(end_ms - start_ms) / 1000:.3f}",
            "-i", str(video_path),
            "-vf", f"fps={1000 / step:g},scale={VISUAL_FRAME_WIDTH // 2}:-2",
            str(frame_dir / "r_%03d.jpg"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        return []
    files = sorted(frame_dir.glob("r_*.jpg"))
    return [(start_ms + index * step, path) for index, path in enumerate(files)]


def _is_own_brand_description(description: str) -> bool:
    if any(brand and brand in description for brand in VISUAL_OWN_BRANDS):
        if not any(
            word in description
            for word in ("第三方", "未授权", "相似", "模仿", "商标", "logo", "包装")
        ):
            return True
    return "字幕" in description and "品牌名" in description


def _merge_adjacent_visual_events(events: list[VisualEvent]) -> list[VisualEvent]:
    groups: dict[tuple[str, str], list[VisualEvent]] = {}
    for event in events:
        groups.setdefault((event.category, event.rule_id), []).append(event)

    merged: list[VisualEvent] = []
    for group in groups.values():
        group.sort(key=lambda item: item.evidence[0].start_ms)
        current = group[0]
        for event in group[1:]:
            previous_end = max(item.end_ms for item in current.evidence)
            current_start = min(item.start_ms for item in event.evidence)
            if current_start > previous_end + 500:
                merged.append(current)
                current = event
                continue
            evidence = {
                item.frame_index: item for item in [*current.evidence, *event.evidence]
            }
            current = VisualEvent(
                rule_id=current.rule_id,
                title=current.title,
                category=current.category,
                severity=current.severity,
                evidence=tuple(evidence[index] for index in sorted(evidence)),
                reason=current.reason,
                suggestion=current.suggestion,
                confidence=max(current.confidence, event.confidence),
            )
        merged.append(current)
    return sorted(merged, key=lambda item: item.evidence[0].start_ms)


def _extract_frames(
    video_path: Path,
    duration_ms: int,
    frame_dir: Path,
) -> list[tuple[int, Path]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    count = min(VISUAL_MAX_FRAMES, max(8, math.ceil(duration_ms / 5000)))
    step = duration_ms / count
    frames: list[tuple[int, Path]] = []
    for index in range(count):
        timestamp_ms = int(step * (index + 0.5))
        output_path = frame_dir / f"frame_{index:03d}.jpg"
        completed = subprocess.run(
            [
                str(FFMPEG_PATH),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp_ms / 1000:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={VISUAL_FRAME_WIDTH}:-2",
                "-q:v",
                "4",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 1000:
            frames.append((timestamp_ms, output_path))
    return frames


def _call_visual(api_key: str, payload: dict[str, Any]) -> str:
    try:
        response = httpx.post(
            VISUAL_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(VISUAL_TIMEOUT_SECONDS, connect=20.0),
        )
    except httpx.HTTPError as exc:
        raise VisualError(f"视觉审核请求失败：{type(exc).__name__}") from exc
    if response.status_code != 200:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = {}
        message = str(body.get("error", {}).get("message") or body.get("message") or response.text[:300])
        raise VisualError(f"视觉审核调用失败：{message[:240]}", code=str(response.status_code))
    try:
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return str(message.get("content") or message.get("reasoning_content") or "")
    except Exception as exc:  # noqa: BLE001
        raise VisualError(f"视觉审核响应解析失败：{exc}") from exc


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise VisualError("视觉审核返回不是合法 JSON")
    decoder = json.JSONDecoder()
    for candidate in (text[start:], _escape_control_chars(text[start:])):
        try:
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise VisualError("视觉审核返回不是合法 JSON")


def _escape_control_chars(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                output.append(char)
                in_string = False
            elif char == "\n":
                output.append("\\n")
            elif char == "\r":
                output.append("\\r")
            elif char == "\t":
                output.append("\\t")
            else:
                output.append(char)
        else:
            output.append(char)
            if char == '"':
                in_string = True
    return "".join(output)


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))

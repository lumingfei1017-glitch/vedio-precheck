from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import (
    DASHSCOPE_API_KEY,
    SEMANTIC_ENABLED,
    SEMANTIC_ENDPOINT,
    SEMANTIC_MAX_TOKENS,
    SEMANTIC_MODEL,
    SEMANTIC_TIMEOUT_SECONDS,
)
from .srt import Segment, align_quote, normalize


class SemanticError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SemanticEvidence:
    quote: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SemanticEvent:
    rule_id: str
    title: str
    severity: str
    evidence: tuple[SemanticEvidence, ...]
    reason: str
    suggestion: str
    method: str
    abc_class: str
    confidence: float
    needs_human: bool


SYSTEM_PROMPT = """你是抖音千川服装配饰（塑身衣）广告审核专家。
你的任务不是关键词匹配，而是识别“违规事件”：把同一违规意图下的多句口播和演示描述合并成一条证据链。

平台判例口径（171 条真实千川判例）：
- 169 条是“夸大功效/虚假塑身效果”，只有 2 条是“不良导向/低俗擦边”。只在这两类明确命中时输出，不要自创类别。
- 平台处置文字统一是“请修改”，不是“删除整段”。口播类违规默认给改写，不要默认删段。
- 典型判例（同一口径才算命中）：
  · “小肚腩收紧了，大肚腩还有你的游泳圈以及大尾巴全部整理平整了”
  · “腰腹线条就被柔和修饰了”
  · “穿上它不仅能给你带来身材上面的改变”
  · “显瘦十几斤”“130 斤穿成 100 斤”
  · “150 的穿北极绒，120 的穿北极绒，90 斤的还是穿北极绒”（对适用范围做明确承诺）
  · “无特效无剪辑，现场给姐妹看一下这个收腹效果”＋“肚子一下就兜进去”（真实性背书＋即时收腹演示）

审核口径：
1. 服装配饰不得夸大产品功效、性能、服务或使用效果，不得承诺确定性的塑形/身材改变结果。
2. 高风险承诺包括：显瘦斤数、蚂蚁腰、蜜桃臀、S 型、收腹提臀、肚子变平/消失、身材大变样、越穿越有型、长期定型。
3. “无特效无剪辑”“现场给姐妹看”“真实演示”“穿前穿后对比”等真实性背书，如果和即时收腹/身材改变描述组合，要视为一个虚假效果展示事件。
4. 不得宣称减肥、减脂、燃脂、治疗疾病、改善私处健康；不得使用绝对化或伪科技表述。
5. “赘肉、脂肪、肚子大、肉肉”本身是中性描述，不单独判违规；只有和收紧、消失、减掉、提上来、管理住等功效承诺组合时才算违规。
6. 证据必须逐字来自给定口播。不要编造句子；找不到原文就不要输出该证据。

动作选择（必须遵守）：
- 违规只来自口播或字幕 → method 用 WAN_LIPSYNC（画面有人脸）或 TTS_SUBTITLE（无脸/画外音），abc_class=A；
  suggestion 必须是一条可以直接念出来的新台词，长度按 evidence 总时长 × 5 字/秒估算。
  不要写“删除……改为……”这类指令。
- 只有画面本身违规才用 CUT_BROLL / B：假道具冒充真实效果、穿前穿后分屏、重影或特效展示身材变化、身体部位长时间特写。
- 一句话能改好的，不要扩成整段删除。

粒度：
- 一个 event 的 evidence 时间跨度不得超过 8 秒；超过就按时间拆成多个 event，每个 event 各自给改写台词。
- 证据之间间隔超过 1.5 秒的，视为不同事件。

不要报的中性表述：
- 结构描述：多排扣、拉链、双层塑形力、面料、版型、工艺；
- 价格与活动：原价、直播间价、大促；
- 单纯体型描述：肚子大、有赘肉、肉松（不与“收紧/消失/管理住/变平”组合时不报）；
- 主观体感：穿着舒服、包裹感明显、看着利落。

只输出一个 JSON 对象，不要 Markdown，不要解释。格式：
{
  "events": [
    {
      "rule_id": "SEM-APPAREL-EFFECT",
      "title": "简短中文标题",
      "severity": "blocking|suggest|info",
      "evidence": [
        {"quote": "口播原文", "start_ms": 1200, "end_ms": 3400}
      ],
      "reason": "为什么这些证据组合起来构成违规",
      "suggestion": "可直接用于改写或删句的中文建议",
      "method": "TTS_SUBTITLE|CUT_BROLL|WAN_LIPSYNC|MANUAL",
      "abc_class": "A|B|C|MANUAL",
      "confidence": 0.0,
      "needs_human": true
    }
  ]
}
如果没有违规，输出 {"events": []}。"""


POSITIVE_EXAMPLE = {
    "transcript": (
        "[1] [1200-4200] 无特效无剪辑，现场给姐妹看一下这个收腹效果\n"
        "[2] [4300-5600] 肚子一下就兜进去\n"
        "[3] [6100-7200] 压的特别紧实\n"
        "[4] [8200-10500] 从肩膀到大腿，整个人的身材都能大变样"
    ),
    "output": {
        "events": [
            {
                "rule_id": "SEM-APPAREL-EFFECT",
                "title": "现场演示 + 确定性塑形承诺",
                "severity": "blocking",
                "evidence": [
                    {
                        "quote": "无特效无剪辑，现场给姐妹看一下这个收腹效果",
                        "start_ms": 1200,
                        "end_ms": 4200,
                    },
                    {"quote": "肚子一下就兜进去", "start_ms": 4300, "end_ms": 5600},
                    {"quote": "压的特别紧实", "start_ms": 6100, "end_ms": 7200},
                    {
                        "quote": "从肩膀到大腿，整个人的身材都能大变样",
                        "start_ms": 8200,
                        "end_ms": 10500,
                    },
                ],
                "reason": "以“无特效无剪辑”作真实性背书，叠加现场收腹演示和全身改变描述，形成确定性塑形承诺和虚假效果展示。",
                "suggestion": "删除现场效果保证和全身改变承诺，改为单句描述版型包裹感和穿着体感。",
                "method": "CUT_BROLL",
                "abc_class": "B",
                "confidence": 0.95,
                "needs_human": True,
            }
        ]
    },
}

NEGATIVE_EXAMPLE = {
    "transcript": (
        "[1] [0-2600] 你如果肚子没有赘肉，不要拍\n"
        "[2] [2800-6200] 我们肚子上只要有赘肉的姐妹，可以看看这条"
    ),
    "output": {"events": []},
}


def review_semantics(
    transcript: str,
    segments: list[Segment],
    rule_findings: list[dict[str, Any]],
    *,
    has_face: bool | None = None,
) -> list[SemanticEvent]:
    if not SEMANTIC_ENABLED or not transcript.strip():
        return []
    api_key = DASHSCOPE_API_KEY
    if not api_key:
        raise SemanticError("未配置 DASHSCOPE_API_KEY")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "正例输入：\n"
                f"{POSITIVE_EXAMPLE['transcript']}\n"
                "正例输出：\n"
                f"{json.dumps(POSITIVE_EXAMPLE['output'], ensure_ascii=False)}\n\n"
                "反例输入：\n"
                f"{NEGATIVE_EXAMPLE['transcript']}\n"
                "反例输出：\n"
                f"{json.dumps(NEGATIVE_EXAMPLE['output'], ensure_ascii=False)}"
            ),
        },
        {"role": "user", "content": _build_review_input(transcript, segments, rule_findings)},
    ]

    payload = {
        "model": SEMANTIC_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": SEMANTIC_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    content = _call_model(api_key, payload)
    try:
        data = _parse_json(content)
    except Exception:  # noqa: BLE001
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content[:6000]},
            {
                "role": "user",
                "content": (
                    "上面的输出不是合法 JSON。请只返回修正后的合法 JSON："
                    "字符串内部的引号和换行必须转义，不要输出 Markdown 或解释。"
                ),
            },
        ]
        repaired = _call_model(api_key, {**payload, "messages": repair_messages})
        try:
            data = _parse_json(repaired)
        except Exception as exc:  # noqa: BLE001
            raise SemanticError(f"语义审核返回不是合法 JSON：{exc}") from exc

    raw_events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(raw_events, list):
        raise SemanticError("语义审核返回缺少 events 数组")

    events: list[SemanticEvent] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        event = _parse_event(item, transcript, segments, has_face)
        if event is None:
            continue
        key = (event.title, tuple(sorted(normalize(e.quote) for e in event.evidence)))
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
    return events


def _call_model(api_key: str, payload: dict[str, Any]) -> str:
    try:
        response = httpx.post(
            SEMANTIC_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(SEMANTIC_TIMEOUT_SECONDS, connect=20.0),
        )
    except httpx.HTTPError as exc:
        raise SemanticError(f"语义审核请求失败：{type(exc).__name__}") from exc

    if response.status_code != 200:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = {}
        message = str(body.get("error", {}).get("message") or body.get("message") or response.text[:300])
        raise SemanticError(f"语义审核调用失败：{message[:240]}", code=str(response.status_code))

    try:
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return str(message.get("content") or message.get("reasoning_content") or "")
    except Exception as exc:  # noqa: BLE001
        raise SemanticError(f"语义审核响应解析失败：{exc}") from exc


def _build_review_input(
    transcript: str,
    segments: list[Segment],
    rule_findings: list[dict[str, Any]],
) -> str:
    lines = []
    for segment in segments:
        lines.append(f"[{segment.index}] [{segment.start_ms}-{segment.end_ms}] {segment.text}")
    transcript_with_times = "\n".join(lines) if lines else transcript

    candidates = []
    for finding in rule_findings[:40]:
        candidates.append(
            {
                "rule_id": finding.get("rule_id"),
                "quote": finding.get("quote"),
                "start_ms": finding.get("start_ms"),
                "end_ms": finding.get("end_ms"),
            }
        )
    from .audit_feed import build_prompt_block

    platform_cases = build_prompt_block()
    case_block = (
        f"平台近期拒审判例（自动同步，只作口径参考）：\n{platform_cases}\n\n"
        if platform_cases
        else ""
    )
    return (
        "请审核下面这条塑身衣广告的完整口播。规则候选只作提示，不是必须命中；"
        "如果多条候选属于同一个违规事件，请合并成一条 event。\n\n"
        f"{case_block}"
        f"规则候选：\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"完整口播（带时间码）：\n{transcript_with_times}\n\n"
        "只输出规定格式的 JSON。"
    )


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    for candidate in (text[start:], _escape_control_chars(text[start:])):
        try:
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("invalid JSON object")


def _escape_control_chars(text: str) -> str:
    """Escape raw newlines/tabs that models sometimes put inside JSON strings."""
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


def _parse_event(
    item: dict[str, Any],
    transcript: str,
    segments: list[Segment],
    has_face: bool | None,
) -> SemanticEvent | None:
    evidence_items = item.get("evidence")
    if not isinstance(evidence_items, list):
        return None

    evidence: list[SemanticEvidence] = []
    normalized_transcript = normalize(transcript)
    for raw_evidence in evidence_items[:6]:
        if not isinstance(raw_evidence, dict):
            continue
        quote = str(raw_evidence.get("quote") or "").strip()
        if not quote or normalize(quote) not in normalized_transcript:
            continue
        start_ms = _to_int(raw_evidence.get("start_ms"))
        end_ms = _to_int(raw_evidence.get("end_ms"))
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            aligned_start, aligned_end, _ = align_quote(quote, segments)
            start_ms, end_ms = aligned_start, aligned_end
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        evidence.append(SemanticEvidence(quote=quote, start_ms=start_ms, end_ms=end_ms))

    if not evidence:
        return None

    severity = str(item.get("severity") or "blocking").lower()
    if severity not in {"blocking", "suggest", "info"}:
        severity = "blocking"
    method = str(item.get("method") or "WAN_LIPSYNC").upper()
    if method not in {"TTS_SUBTITLE", "CUT_BROLL", "WAN_LIPSYNC", "MANUAL"}:
        method = "WAN_LIPSYNC"
    abc_class = str(item.get("abc_class") or "").upper()
    if abc_class not in {"A", "B", "C", "MANUAL"}:
        abc_class = _default_class(method, has_face)
    if method == "WAN_LIPSYNC" and has_face is False:
        method, abc_class = "TTS_SUBTITLE", "A"
    elif method == "TTS_SUBTITLE":
        abc_class = "A"
    elif method == "CUT_BROLL":
        abc_class = "B"
    elif method == "MANUAL":
        abc_class = "MANUAL"
    elif method == "WAN_LIPSYNC" and abc_class not in {"C", "MANUAL"}:
        abc_class = "C"

    return SemanticEvent(
        rule_id=str(item.get("rule_id") or "SEM-APPAREL-EFFECT"),
        title=str(item.get("title") or "语义审核事件"),
        severity=severity,
        evidence=tuple(evidence),
        reason=str(item.get("reason") or ""),
        suggestion=str(item.get("suggestion") or "改为描述穿着体感和版型，删除确定性效果承诺。"),
        method=method,
        abc_class=abc_class,
        confidence=_clamp_float(item.get("confidence"), 0.85),
        needs_human=bool(item.get("needs_human", True)),
    )


def _default_class(method: str, has_face: bool | None) -> str:
    if method == "CUT_BROLL":
        return "B"
    if method == "MANUAL":
        return "MANUAL"
    if method == "TTS_SUBTITLE" or has_face is False:
        return "A"
    return "C"


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))

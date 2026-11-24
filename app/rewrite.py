from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import (
    DASHSCOPE_API_KEY,
    REWRITE_ENABLED,
    REWRITE_ENDPOINT,
    REWRITE_MAX_TOKENS,
    REWRITE_MODEL,
    REWRITE_TIMEOUT_SECONDS,
    SPEECH_CHARS_PER_SECOND,
)
from .rules import scan
from .srt import Segment, text_from_segments
from .suggest import estimate_duration_ms


class RewriteError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RewriteResult:
    action: str
    after_text: str | None
    edit_instruction: str
    kept_selling_point: str
    estimated_duration_ms: int | None
    alternatives: tuple[str, ...]
    confidence: float


REWRITE_PROMPT = """你是抖音千川塑身衣广告编导，不是审核员。
审核 Agent 已经找出了违规点。你的任务是把每个 target 改成可以直接给主播口播的中文成稿，或者给出明确的剪辑指令。

必须遵守：
1. replace_script：写 1-2 句自然口语，保留原来的卖点、主播口吻和前后衔接。
   **字数必须落在 target.min_chars 到 target.max_chars 之间**：
   超过 max_chars 会导致吞字糊音；低于 min_chars 会让时长缩短超过 25%，
   结尾被原句碎片填空。写不满就补一个具体穿着场景，但不要新增功效。
2. cut_segment：如果问题在穿前穿后对比、局部特写、极端拉伸等画面，只写剪辑指令，after_text 必须为 null。
3. 不新增功效，不出现量化显瘦、减肥、治疗、私密健康、绝对化、确定性收腹提臀等承诺。
4. 不把“删除……改为……”这种审核建议写进 after_text。after_text 必须是能直接读出来的口播。
5. 如果原句只是中性描述，不要过度改写。
6. 改写必须能接住前一句、并自然引出后一句，读起来像同一个人在同一条片子里接着说话，
   不能写成独立广告口号。先看完整口播再动笔，不要只盯着被改写的那一句。
7. 不新增原文没有的信息：不加新场景、新人群、新功效、新价格。
8. 主文案求稳，另外再给 **1 条备选**（alternatives 恰好 1 条），
   备选是同一意思的不同说法，不要换成另一套卖点，也不要和主文案重复。
9. 不要用“姐妹们可以试试看”“喜欢的姐妹可以试试”“推荐大家试一试”这类空话凑字数。
   每条改写至少保留一个具体信息：穿着场景、动作步骤、或产品结构（多排扣、拉链、面料）。
   写不出具体信息时，宁可描述当下的穿着动作，也不要写空泛的推荐语。

参考写法（违规句 → 可直接口播的成稿）：
- 这件塑身衣上身显瘦十几斤 → 这件上身包裹感挺明显，腰腹这块看着更利落。
- 小肚腩收紧了，大肚腩还有你的游泳圈全部整理平整了 → 腰腹这一圈包得比较服帖，穿起来平整一些。
- 无特效无剪辑，现场给姐妹看一下这个收腹效果 → 给大家看一下上身效果，腰腹的包裹感还是挺明显的。
- 150的穿北极绒，120的穿北极绒，90斤的还是穿北极绒 → 不同体型的姐妹，按尺码表选都能找到合适的码数。
- 连续穿它3个月，你就离不开它了 → 秋天贴身穿在里面，外面搭件薄针织也看不出来。

反例（空话凑数，不允许）：秋天穿它正合适，姐妹们可以试试看。

正例：
原文：这件塑身衣上身显瘦十几斤
改写：姐妹们，这件上身包裹感挺明显，腰腹这块看着更利落。

反例：
删除量化形体改变，改为穿着体感描述。

只输出 JSON：
{
  "rewrites": [
    {
      "target_index": 0,
      "action": "replace_script|cut_segment|manual",
      "after_text": "可直接口播的成稿，cut_segment/manual 时为 null",
      "edit_instruction": "剪辑或人工处理指令",
      "kept_selling_point": "保留的卖点",
      "estimated_duration_ms": 2600,
      "alternatives": ["备选口播 1"],
      "confidence": 0.86
    }
  ]
}"""


def rewrite_suggestions(
    transcript: str,
    segments: list[Segment],
    findings: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
    *,
    has_face: bool | None = None,
    instruction: str | None = None,
    strict: bool = False,
) -> dict[str, RewriteResult]:
    if not REWRITE_ENABLED or not findings or not suggestions:
        return {}
    api_key = DASHSCOPE_API_KEY
    if not api_key:
        raise RewriteError("未配置 DASHSCOPE_API_KEY")

    finding_by_id = {finding["id"]: finding for finding in findings}
    targets = []
    target_suggestions = []
    for suggestion in suggestions:
        finding = finding_by_id.get(suggestion["finding_id"])
        if finding is None:
            continue
        targets.append(_build_target(len(targets), finding, suggestion, segments))
        target_suggestions.append(suggestion)
    if not targets:
        return {}

    payload = {
        "model": REWRITE_MODEL,
        "messages": [
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"完整口播（带时间码）：\n{_transcript_with_times(segments, transcript)}\n\n"
                    f"待改写目标：\n{json.dumps(targets, ensure_ascii=False)}\n\n"
                    f"额外要求：{instruction or '无'}\n"
                    "只输出规定的 JSON。"
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": REWRITE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    content = _call_rewrite(api_key, payload)
    data = _parse_json(content)
    raw_rewrites = data.get("rewrites") if isinstance(data, dict) else None
    if not isinstance(raw_rewrites, list):
        raise RewriteError("改写返回缺少 rewrites 数组")

    results: dict[str, RewriteResult] = {}
    for item in raw_rewrites:
        if not isinstance(item, dict):
            continue
        try:
            target_index = int(item.get("target_index"))
        except (TypeError, ValueError):
            continue
        if target_index < 0 or target_index >= len(target_suggestions):
            continue
        suggestion = target_suggestions[target_index]
        result = _parse_rewrite(item, suggestion)
        if result is not None:
            results[suggestion["id"]] = result

    if not strict:
        pairs = [
            (finding_by_id[suggestion["finding_id"]], suggestion)
            for suggestion in target_suggestions
        ]
        over = [
            (finding, suggestion)
            for finding, suggestion in pairs
            if (result := results.get(suggestion["id"])) is not None
            and _off_budget(result, suggestion) == "over"
        ]
        under = [
            (finding, suggestion)
            for finding, suggestion in pairs
            if (result := results.get(suggestion["id"])) is not None
            and _off_budget(result, suggestion) == "under"
        ]
        if over:
            results.update(
                rewrite_suggestions(
                    transcript,
                    segments,
                    [item[0] for item in over],
                    [item[1] for item in over],
                    has_face=has_face,
                    instruction="以下目标超出镜头容量，必须压到 max_chars 以内：删修饰语、删重复，只保留核心信息。",
                    strict=True,
                )
            )
        repeat = [
            (finding, suggestion)
            for finding, suggestion in pairs
            if (result := results.get(suggestion["id"])) is not None
            and _off_budget(result, suggestion) == "repeat"
        ]
        if repeat:
            results.update(
                rewrite_suggestions(
                    transcript,
                    segments,
                    [item[0] for item in repeat],
                    [item[1] for item in repeat],
                    has_face=has_face,
                    instruction="以下目标的开头或结尾与原句相同，容易被模型顺着复读。请换掉开头和结尾的措辞，保持意思不变。",
                    strict=True,
                )
            )
        missing = [
            (finding, suggestion)
            for finding, suggestion in pairs
            if (result := results.get(suggestion["id"])) is not None
            and result.action == "replace_script"
            and result.after_text
            and not result.alternatives
        ]
        if missing:
            results.update(
                rewrite_suggestions(
                    transcript,
                    segments,
                    [item[0] for item in missing],
                    [item[1] for item in missing],
                    has_face=has_face,
                    instruction="每个目标除了主文案，必须再给 1 条措辞明显不同的备选，意思保持接近。",
                    strict=True,
                )
            )
        if under:
            results.update(
                rewrite_suggestions(
                    transcript,
                    segments,
                    [item[0] for item in under],
                    [item[1] for item in under],
                    has_face=has_face,
                    instruction="以下目标太短，时长缩短超过 20%。请扩写到 min_chars 以上：补一个具体穿着场景或体感描述，不要新增功效。",
                    strict=True,
                )
            )
    return results


def _clean_text(value: str) -> str:
    return re.sub(r"[\s，。！？；：、,.!?;:：]", "", value or "")


def _repeat_edges(text: str, original: str) -> bool:
    def clean(value: str) -> str:
        return re.sub(r"[\s，。！？；：、,.!?;:：]", "", value or "")

    new_text = clean(text)
    old_text = clean(original)
    if len(new_text) < 5 or len(old_text) < 5:
        return False
    return new_text[:5] == old_text[:5] or new_text[-5:] == old_text[-5:]


def _off_budget(result: RewriteResult, suggestion: dict[str, Any]) -> str | None:
    """返回 over / under / repeat / None：任一不满足都要重写。"""
    if result.after_text and _repeat_edges(result.after_text, suggestion.get("before_text") or ""):
        return "repeat"
    estimated = result.estimated_duration_ms or 0
    original = suggestion.get("original_duration_ms") or 0
    if not estimated or not original:
        return None
    if estimated > original * 1.05 + 150:
        return "over"
    if estimated < original * 0.75:
        return "under"
    return None


def rewrite_one(
    transcript: str,
    segments: list[Segment],
    finding: dict[str, Any],
    suggestion: dict[str, Any],
    *,
    has_face: bool | None = None,
    instruction: str | None = None,
) -> RewriteResult | None:
    results = rewrite_suggestions(
        transcript,
        segments,
        [finding],
        [suggestion],
        has_face=has_face,
        instruction=instruction,
    )
    return results.get(suggestion["id"])


def _build_target(
    index: int,
    finding: dict[str, Any],
    suggestion: dict[str, Any],
    segments: list[Segment],
) -> dict[str, Any]:
    start_ms = finding.get("start_ms")
    end_ms = finding.get("end_ms")
    before: list[str] = []
    after: list[str] = []
    if start_ms is not None and end_ms is not None:
        first_index = next(
            (i for i, segment in enumerate(segments) if segment.end_ms > start_ms),
            None,
        )
        last_index = next(
            (i for i, segment in enumerate(segments) if segment.end_ms >= end_ms),
            first_index,
        )
        if first_index is not None:
            before = [segment.text for segment in segments[max(0, first_index - 4) : first_index]]
            after = [segment.text for segment in segments[(last_index or first_index) + 1 : (last_index or first_index) + 5]]
    return {
        "target_index": index,
        "rule_id": finding.get("rule_id"),
        "title": finding.get("title"),
        "severity": finding.get("severity"),
        "quote": finding.get("quote"),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "max_chars": _max_chars(suggestion, start_ms, end_ms),
        "min_chars": _min_chars(suggestion, start_ms, end_ms),
        "method": suggestion.get("method"),
        "abc_class": suggestion.get("abc_class"),
        "original_duration_ms": suggestion.get("original_duration_ms"),
        "context_before": before,
        "context_after": after,
    }


def _max_chars(suggestion: dict[str, Any], start_ms: Any, end_ms: Any) -> int | None:
    """镜头能容纳的台词字数。

    Wan 只收整数秒，所以容量按 floor(时长) 计算；用原始时长会高估半秒左右。
    """
    original = suggestion.get("original_duration_ms")
    if not original and isinstance(start_ms, (int, float)) and isinstance(end_ms, (int, float)):
        original = int(end_ms) - int(start_ms)
    if not original or original <= 0:
        return None
    seconds = int(int(original) // 1000)
    if seconds < 2:
        seconds = max(2, int(round(int(original) / 1000)))
    return max(6, int(seconds * SPEECH_CHARS_PER_SECOND))


def _min_chars(suggestion: dict[str, Any], start_ms: Any, end_ms: Any) -> int | None:
    """时长短缩不超过 20% 对应的最少字数。"""
    maximum = _max_chars(suggestion, start_ms, end_ms)
    if maximum is None:
        return None
    original = suggestion.get("original_duration_ms")
    if not original and isinstance(start_ms, (int, float)) and isinstance(end_ms, (int, float)):
        original = int(end_ms) - int(start_ms)
    if not original or original <= 0:
        return None
    return max(6, min(maximum, int(int(original) / 1000 * SPEECH_CHARS_PER_SECOND * 0.75)))


def _transcript_with_times(segments: list[Segment], transcript: str) -> str:
    if not segments:
        return transcript
    return "\n".join(
        f"[{segment.index}] [{segment.start_ms}-{segment.end_ms}] {segment.text}"
        for segment in segments
    )


def _parse_rewrite(
    item: dict[str, Any],
    suggestion: dict[str, Any],
) -> RewriteResult | None:
    method = suggestion.get("method")
    default_action = (
        "cut_segment" if method == "CUT_BROLL" else "manual" if method == "MANUAL" else "replace_script"
    )
    action = str(item.get("action") or default_action)
    if method == "CUT_BROLL":
        action = "cut_segment"
    elif method == "MANUAL":
        action = "manual"
    elif action not in {"replace_script", "cut_segment", "manual"}:
        action = "replace_script"

    edit_instruction = str(item.get("edit_instruction") or "").strip()
    after_text = str(item.get("after_text") or "").strip()
    if action == "replace_script":
        if not after_text:
            return None
        blocking = [match for match in scan(after_text) if match.rule.severity == "blocking"]
        if blocking:
            return None
        edit_instruction = edit_instruction or f"替换 {suggestion.get('before_text', '')} 这段口播"
    else:
        after_text = None
        if not edit_instruction:
            edit_instruction = suggestion.get("after_text") or "按审核证据删改对应片段"

    alternatives = tuple(
        str(value).strip()
        for value in (item.get("alternatives") or [])
        if isinstance(value, str) and value.strip()
    )
    if alternatives:
        alternatives = tuple(
            value
            for value in alternatives
            if not [match for match in scan(value) if match.rule.severity == "blocking"]
        )

    # 主文案超容量时，先从已有的备选里挑一条装得下的，避免白白多跑一次生成。
    swapped = False
    if after_text:
        budget = _max_chars(suggestion, suggestion.get("start_ms"), suggestion.get("end_ms"))
        if budget and len(after_text) > budget:
            # 太短的备选会丢掉卖点，宁可交给后面的压缩重试，也不静默换成一句干巴巴的话。
            floor = int(budget * 0.6)
            fitting = [value for value in alternatives if floor <= len(value) <= budget]
            if fitting:
                after_text = max(fitting, key=len)
                swapped = True

    estimated = item.get("estimated_duration_ms")
    try:
        estimated = int(estimated) if estimated is not None else None
    except (TypeError, ValueError):
        estimated = None
    # 统一按系统的 5 字/秒口径算时长，不采信模型自报的数字，避免显示与字数对不上。
    if after_text:
        estimated = estimate_duration_ms(after_text)

    return RewriteResult(
        action=action,
        after_text=after_text,
        edit_instruction=edit_instruction,
        kept_selling_point=str(item.get("kept_selling_point") or "").strip(),
        estimated_duration_ms=estimated,
        alternatives=tuple(
            value
            for value in alternatives[:2]
            if _clean_text(value) != _clean_text(after_text or "")
        )[:1],
        confidence=_clamp_float(item.get("confidence"), 0.8),
    )


def _call_rewrite(api_key: str, payload: dict[str, Any]) -> str:
    try:
        response = httpx.post(
            REWRITE_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(REWRITE_TIMEOUT_SECONDS, connect=20.0),
        )
    except httpx.HTTPError as exc:
        raise RewriteError(f"改写请求失败：{type(exc).__name__}") from exc
    if response.status_code != 200:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = {}
        message = str(body.get("error", {}).get("message") or body.get("message") or response.text[:300])
        raise RewriteError(f"改写调用失败：{message[:240]}", code=str(response.status_code))
    try:
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return str(message.get("content") or message.get("reasoning_content") or "")
    except Exception as exc:  # noqa: BLE001
        raise RewriteError(f"改写响应解析失败：{exc}") from exc


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise RewriteError("改写返回不是合法 JSON")
    decoder = json.JSONDecoder()
    for candidate in (text[start:], _escape_control_chars(text[start:])):
        try:
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RewriteError("改写返回不是合法 JSON")


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

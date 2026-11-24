from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import SPEECH_CHARS_PER_SECOND
from .rules import Rule, RuleMatch

CHARS_PER_SECOND = SPEECH_CHARS_PER_SECOND
MANUAL_ACTIONS = {"request_material", "request_authorization"}

SAFE_SENTENCES = {
    "AP-01": "穿上后腰腹包裹感更明显，搭配高腰下装更利落。",
    "AP-02": "版型剪裁利落，腰线和臀线看起来更自然。",
    "AP-03": "删除穿前穿后对比镜头，改成单一口播或产品细节展示。",
    "AP-04": "日常穿着更贴合，具体感受以实际穿着为准。",
    "AP-05": "版型剪裁利落，日常搭配更省心。",
    "AP-06": "多种尺码可选，建议按尺码表选择。",
    "AP-07": "展示正常弹力和面料细节。",
    "AP-08": "保暖面料，适合日常穿着。",
    "AP-09": "删除该表述，或补充对应质检报告后再使用。",
    "AP-10": "删除或替换涉及敏感部位的镜头。",
    "AP-11": "删除明星/品牌相关表述，或补充授权。",
    "AP-12": "日常穿着舒适，版型利落。",
    "AP-14": "删除绝对化表述，改成有边界的日常描述。",
    "AP-15": "版型包裹感更明显，腰腹更贴合。",
}


@dataclass(frozen=True)
class SuggestionDraft:
    before_text: str
    after_text: str
    method: str
    abc_class: str
    estimated_duration_ms: int | None
    original_duration_ms: int | None
    duration_warning: bool
    confidence: float


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([，。！？；])\1+", r"\1", text)
    text = re.sub(r"\s+([，。！？；])", r"\1", text)
    return text


def classify(rule: Rule, has_face: bool | None) -> tuple[str, str]:
    if rule.action in MANUAL_ACTIONS or rule.default_method == "MANUAL":
        return "MANUAL", "MANUAL"
    if rule.default_method == "WAN_LIPSYNC":
        if has_face is False:
            return "TTS_SUBTITLE", "A"
        return "WAN_LIPSYNC", "C"
    if rule.default_method == "CUT_BROLL":
        return "CUT_BROLL", "B"
    return "TTS_SUBTITLE", "A"


def _apply_replacements(text: str, replacements: Iterable[tuple[re.Pattern[str], str]]) -> str:
    result = text
    for pattern, replacement in replacements:
        result = pattern.sub(replacement, result)
    return clean_text(result)


def rewrite_text(quote: str, current_rule: Rule, all_rules: Iterable[Rule]) -> str:
    if current_rule.action == "cut" or current_rule.action in MANUAL_ACTIONS:
        return SAFE_SENTENCES.get(current_rule.id, current_rule.fallback)

    result = _apply_replacements(quote, current_rule.replacements)
    for rule in all_rules:
        if rule.id == current_rule.id:
            continue
        if rule.action not in {"rewrite", "replace_selling_point"}:
            continue
        result = _apply_replacements(result, rule.replacements)

    if not result or result == quote:
        return SAFE_SENTENCES.get(current_rule.id, current_rule.fallback)
    if "自带面料成分" in result:
        return SAFE_SENTENCES["AP-08"]
    if "坚持日常穿着" in result:
        return SAFE_SENTENCES["AP-04"]
    if result.startswith("可以日常穿着舒适"):
        return "日常穿着舒适。"
    return result


def estimate_duration_ms(text: str) -> int:
    compact = re.sub(r"\s+", "", text)
    return max(800, int(len(compact) / CHARS_PER_SECOND * 1000))


def build_suggestion(
    match: RuleMatch,
    has_face: bool | None,
    original_duration_ms: int | None,
    all_rules: Iterable[Rule],
) -> SuggestionDraft:
    method, abc_class = classify(match.rule, has_face)
    after_text = rewrite_text(match.quote, match.rule, all_rules)
    estimated = None if method in {"MANUAL", "CUT_BROLL"} else estimate_duration_ms(after_text)
    warning = bool(
        estimated is not None
        and original_duration_ms is not None
        and original_duration_ms > 0
        and estimated > original_duration_ms * 1.15 + 500
    )
    confidence = 0.9 if method == "MANUAL" else (0.85 if after_text != match.quote else 0.65)
    return SuggestionDraft(
        before_text=match.quote,
        after_text=after_text,
        method=method,
        abc_class=abc_class,
        estimated_duration_ms=estimated,
        original_duration_ms=original_duration_ms,
        duration_warning=warning,
        confidence=confidence,
    )


def refresh_text(current: str, constraint: str | None, instruction: str | None = None) -> str:
    value = (constraint or "").strip()
    hint = (instruction or "").strip()
    merged = f"{value} {hint}"

    if "换卖点" in merged or value == "change_selling_point":
        return SAFE_SENTENCES["AP-05"]

    if "更保守" in merged or value == "more_conservative":
        conservative = current
        for pattern, replacement in (
            (r"显瘦|瘦", "修饰"),
            (r"收腹", "包裹"),
            (r"提臀", "贴合"),
            (r"紧致|塑形", "版型"),
            (r"惊艳|绝了|必买", "可以看看"),
        ):
            conservative = re.sub(pattern, replacement, conservative)
        conservative = clean_text(conservative)
        return conservative if conservative != current else SAFE_SENTENCES["AP-01"]

    if "更口语" in merged or value == "more_colloquial":
        colloquial = current
        for pattern, replacement in (
            (r"更为", "更"),
            (r"具有", "有"),
            (r"能够", "能"),
            (r"非常", "很"),
        ):
            colloquial = re.sub(pattern, replacement, colloquial)
        return clean_text(colloquial)

    if "短" in merged or value == "shorter":
        parts = [part for part in re.split(r"[，,。！？]", current) if part.strip()]
        if len(parts) > 1:
            return clean_text("，".join(parts[:1])) + "。"
        return clean_text(current)

    if "保留" in merged or value == "keep_hook":
        return clean_text(current)

    return clean_text(current)

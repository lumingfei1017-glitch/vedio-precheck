"""人工修改清单：把审核结果整理成可以直接带到剪映的工单。

工作台只自动执行 Wan 纯口播改写，画面类问题和局部换音一律转人工。
清单的判定直接读 build_edit_plan，保证"能自动执行的"和计划里的一致。
时间码基于原始视频（不删段、不重排），可以直接在剪映里跳转。
"""

from __future__ import annotations

from typing import Any

from .edit import build_edit_plan
from .review import get_report
from .srt import format_timestamp

METHOD_HINTS = {
    "CUT_BROLL": "删段或换镜头",
    "TTS_SUBTITLE": "口播需人工配音",
    "MANUAL": "人工处理",
}

RULE_HINTS = {
    "VIS-BRAND": "遮挡品牌元素或换镜头",
    "VIS-BODY-COMPARISON": "删除穿前穿后对比画面",
    "VIS-LOW-SUGGESTIVE": "改构图或遮挡",
    "VIS-EXTREME-STRETCH": "删除极端演示画面",
    "VIS-RISK": "人工确认画面",
}


def build_worklist(material_id: str) -> dict[str, Any] | None:
    report = get_report(material_id)
    if report is None:
        return None
    plan = build_edit_plan(material_id) or {}
    findings = {item["id"]: item for item in report["findings"]}

    auto: list[dict[str, Any]] = []
    auto_ids: set[str] = set()
    for op in plan.get("operations", []):
        if op.get("op") != "replace_script" or op.get("route") != "wan":
            continue
        ids = list(op.get("finding_ids") or [])
        auto_ids.update(ids)
        source = findings.get(ids[0]) if ids else None
        auto.append(
            {
                "start_ms": int(op["start_ms"]),
                "end_ms": int(op["end_ms"]),
                "rule_id": (source or {}).get("rule_id", ""),
                "title": (source or {}).get("title", "口播改写"),
                "text": op.get("text") or "",
            }
        )

    manual: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for finding in report["findings"]:
        if finding["id"] in auto_ids:
            continue
        suggestion = finding.get("suggestion") or {}
        if not suggestion or suggestion.get("status") == "false_positive":
            continue
        manual.append((finding, suggestion))

    return {
        "markdown": _markdown(report, auto, manual),
        "srt": _srt(manual),
        "auto_count": len(auto),
        "manual_count": len(manual),
    }


def _start(item: dict[str, Any]) -> int:
    value = item.get("start_ms")
    return int(value) if isinstance(value, (int, float)) else 0


def _end(item: dict[str, Any]) -> int:
    value = item.get("end_ms")
    return int(value) if isinstance(value, (int, float)) else 0


def _clock(ms: int) -> str:
    total = max(0, ms) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _hint(finding: dict[str, Any], suggestion: dict[str, Any]) -> str:
    rule = str(finding.get("rule_id") or "")
    if rule in RULE_HINTS:
        return RULE_HINTS[rule]
    method = str(suggestion.get("method") or "")
    if method in METHOD_HINTS:
        return METHOD_HINTS[method]
    if suggestion.get("after_text"):
        return "按改后口播人工配音或重录"
    return "人工处理"


def _cell(value: Any) -> str:
    text = str(value or "").replace("|", "／").replace("\n", " ")
    return text[:120]


def _markdown(
    report: dict[str, Any],
    auto: list[dict[str, Any]],
    manual: list[tuple[dict[str, Any], dict[str, Any]]],
) -> str:
    material = report["material"]
    duration = material.get("duration_ms") or 0
    lines = [
        f"# 人工修改清单 · {material.get('filename') or material['id']}",
        "",
        f"- 素材时长：{_clock(duration)}",
        f"- 可自动改写（Wan）：{len(auto)} 条",
        f"- 需人工处理：{len(manual)} 条",
        "",
        "## 一、可自动改写（Wan 单镜重生成）",
        "",
    ]
    if auto:
        lines += ["| 时间 | 规则 | 问题 | 改后口播 |", "| --- | --- | --- | --- |"]
        for item in auto:
            lines.append(
                f"| {_clock(item['start_ms'])}–{_clock(item['end_ms'])} "
                f"| {item['rule_id']} "
                f"| {_cell(item['title'])} "
                f"| {_cell(item['text'])} |"
            )
    else:
        lines.append("（无）")
    lines += ["", "## 二、需人工处理（剪映）", ""]
    if manual:
        lines += ["| 时间 | 规则 | 问题 | 建议动作 |", "| --- | --- | --- | --- |"]
        for finding, suggestion in manual:
            lines.append(
                f"| {_clock(_start(finding))}–{_clock(_end(finding))} "
                f"| {finding.get('rule_id')} "
                f"| {_cell(finding.get('title'))} "
                f"| {_cell(_hint(finding, suggestion))} |"
            )
    else:
        lines.append("（无）")
    lines += [
        "",
        "> 时间码基于原始视频，工作台未删段、未重排时间轴。",
        "",
    ]
    return "\n".join(lines)


def _srt(manual: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    blocks: list[str] = []
    for index, (finding, suggestion) in enumerate(manual, start=1):
        start = _start(finding)
        end = max(start + 800, _end(finding))
        text = f"{finding.get('title') or finding.get('rule_id')}｜{_hint(finding, suggestion)}"
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text[:60]}\n"
        )
    return "\n".join(blocks)

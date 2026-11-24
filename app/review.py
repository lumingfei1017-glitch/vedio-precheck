from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import REWRITE_ENABLED, SEMANTIC_ENABLED, VISUAL_ENABLED
from .db import get_db
from .rules import RuleMatch, load_rules, rule_version, scan
from .semantic import SemanticError, SemanticEvent, review_semantics
from .srt import Segment, align_quote, normalize, parse_srt, read_text
from .suggest import build_suggestion, estimate_duration_ms
from .visual import VisualError, VisualEvent, review_visual
from .rewrite import RewriteError, RewriteResult, rewrite_suggestions


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReviewResult:
    findings: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    recheck: dict[str, Any]
    summary: dict[str, Any]


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def review_transcript(
    material: dict[str, Any],
    transcript: str,
    segments: list[Segment],
) -> ReviewResult:
    all_rules = load_rules()
    matches = scan(transcript, all_rules)
    # AP-03 owns before/after comparisons; avoid a second TTS rewrite for the same clause.
    matches = [
        match
        for match in matches
        if not (
            match.rule.id == "AP-02"
            and re.search(r"穿前|穿后|穿之前|穿之后|前后对比|对比镜头", match.quote)
        )
    ]
    has_face = _bool_or_none(material.get("has_face"))

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for match in matches:
        start_ms, end_ms, align_confidence = align_quote(match.quote, segments)
        original_duration = end_ms - start_ms if start_ms is not None and end_ms is not None else None
        draft = build_suggestion(match, has_face, original_duration, all_rules)

        finding_id = uuid.uuid4().hex
        finding = {
            "id": finding_id,
            "rule_id": match.rule.id,
            "title": match.rule.title,
            "severity": match.rule.severity,
            "category": match.rule.category,
            "modality": list(match.rule.modality),
            "quote": match.quote,
            "rationale": f"{match.rule.why} 命中表述：{match.match_text}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": max(align_confidence, 0.6) if start_ms is not None else 0.55,
            "needs_manual_time": bool(start_ms is None),
        }
        suggestion = {
            "id": uuid.uuid4().hex,
            "finding_id": finding_id,
            "before_text": draft.before_text,
            "after_text": draft.after_text,
            "method": draft.method,
            "abc_class": draft.abc_class,
            "estimated_duration_ms": draft.estimated_duration_ms,
            "original_duration_ms": draft.original_duration_ms,
            "duration_warning": draft.duration_warning,
            "confidence": draft.confidence,
            "status": "pending",
        }
        pairs.append((finding, suggestion))

    findings, suggestions = _merge_same_segment(pairs)
    findings, suggestions = _drop_contained_findings(findings, suggestions)

    recheck = _recheck_suggestions(suggestions)
    summary = _summarize(findings, suggestions, transcript, has_face)
    return ReviewResult(findings=findings, suggestions=suggestions, recheck=recheck, summary=summary)


def review_material_content(
    material: dict[str, Any],
    transcript: str,
    segments: list[Segment],
    progress: Callable[[str, int], None] | None = None,
) -> ReviewResult:
    """Run deterministic rules, semantic events, then visual events."""
    result = review_transcript(material, transcript, segments)
    if progress:
        progress("rules", 55)

    has_face = _bool_or_none(material.get("has_face"))
    if SEMANTIC_ENABLED:
        if progress:
            progress("semantic", 72)
        try:
            events = review_semantics(
                transcript,
                segments,
                result.findings,
                has_face=has_face,
            )
            result.summary["semantic_status"] = "ok"
            result.summary["semantic_count"] = len(events)
            if events:
                _merge_semantic_events(result, events, transcript, has_face)
        except SemanticError as exc:
            result.summary["semantic_status"] = "failed"
            result.summary["semantic_error"] = str(exc)
            result.summary["semantic_count"] = 0
    else:
        result.summary["semantic_status"] = "disabled"
        result.summary["semantic_count"] = 0

    if VISUAL_ENABLED:
        video_path = material.get("video_path")
        if video_path and Path(video_path).is_file():
            if progress:
                progress("visual", 84)
            try:
                visual_events = review_visual(Path(video_path))
                result.summary["visual_status"] = "ok"
                result.summary["visual_count"] = len(visual_events)
                if visual_events:
                    _merge_visual_events(result, visual_events, transcript, has_face)
            except VisualError as exc:
                result.summary["visual_status"] = "failed"
                result.summary["visual_error"] = str(exc)
                result.summary["visual_count"] = 0
        else:
            result.summary["visual_status"] = "failed"
            result.summary["visual_error"] = "视频文件不存在"
            result.summary["visual_count"] = 0
    else:
        result.summary["visual_status"] = "disabled"
        result.summary["visual_count"] = 0

    if REWRITE_ENABLED and result.suggestions:
        if progress:
            progress("rewrite", 94)
        try:
            rewrites = rewrite_suggestions(
                transcript,
                segments,
                result.findings,
                result.suggestions,
                has_face=has_face,
            )
            _apply_rewrites(result, rewrites, transcript, has_face)
            result.summary["rewrite_status"] = "ok"
            result.summary["rewrite_count"] = len(rewrites)
        except RewriteError as exc:
            result.summary["rewrite_status"] = "failed"
            result.summary["rewrite_error"] = str(exc)
            result.summary["rewrite_count"] = 0
    else:
        result.summary["rewrite_status"] = "disabled" if not REWRITE_ENABLED else "empty"
        result.summary["rewrite_count"] = 0

    if progress:
        progress("merge", 92)
    return result


def _merge_semantic_events(
    result: ReviewResult,
    events: list[SemanticEvent],
    transcript: str,
    has_face: bool | None,
) -> None:
    semantic_ranges: list[tuple[int, int]] = []
    semantic_quotes: list[str] = []
    semantic_findings: list[dict[str, Any]] = []
    semantic_suggestions: list[dict[str, Any]] = []

    for event in events:
        evidence = [
            {
                "quote": item.quote,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
            }
            for item in event.evidence
        ]
        start_ms = min(item.start_ms for item in event.evidence)
        end_ms = max(item.end_ms for item in event.evidence)
        quote = "；".join(dict.fromkeys(item.quote for item in event.evidence))
        finding_id = uuid.uuid4().hex
        semantic_ranges.append((start_ms, end_ms))
        semantic_quotes.extend(normalize(item.quote) for item in event.evidence)

        semantic_findings.append(
            {
                "id": finding_id,
                "rule_id": event.rule_id,
                "title": event.title,
                "severity": event.severity,
                "category": "语义审核/夸大功效",
                "modality": ["audio", "subtitle"],
                "quote": quote,
                "rationale": event.reason,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": event.confidence,
                "needs_manual_time": False,
                "source": "semantic",
                "evidence": evidence,
            }
        )

        method = event.method
        estimated = (
            None
            if method in {"MANUAL", "CUT_BROLL"}
            else estimate_duration_ms(event.suggestion)
        )
        original_duration = max(0, end_ms - start_ms)
        warning = bool(
            estimated is not None
            and original_duration > 0
            and estimated > original_duration * 1.15 + 500
        )
        semantic_suggestions.append(
            {
                "id": uuid.uuid4().hex,
                "finding_id": finding_id,
                "before_text": quote,
                "after_text": event.suggestion,
                "method": method,
                "abc_class": event.abc_class,
                "estimated_duration_ms": estimated,
                "original_duration_ms": original_duration,
                "duration_warning": warning,
                "confidence": event.confidence,
                "status": "pending",
            }
        )

    suggestion_by_finding = {
        suggestion["finding_id"]: suggestion for suggestion in result.suggestions
    }
    kept_findings: list[dict[str, Any]] = []
    kept_suggestions: list[dict[str, Any]] = []
    for finding in result.findings:
        if _rule_finding_covered(finding, semantic_ranges, semantic_quotes):
            continue
        kept_findings.append(finding)
        suggestion = suggestion_by_finding.get(finding["id"])
        if suggestion is not None:
            kept_suggestions.append(suggestion)

    result.findings = kept_findings + semantic_findings
    result.suggestions = kept_suggestions + semantic_suggestions
    result.summary = _summarize(result.findings, result.suggestions, transcript, has_face)
    result.summary["semantic_status"] = "ok"
    result.summary["semantic_count"] = len(events)
    result.recheck = _recheck_suggestions(result.suggestions)


def _apply_rewrites(
    result: ReviewResult,
    rewrites: dict[str, RewriteResult],
    transcript: str,
    has_face: bool | None,
) -> None:
    preserved = {
        key: result.summary.get(key)
        for key in (
            "semantic_status",
            "semantic_error",
            "semantic_count",
            "visual_status",
            "visual_error",
            "visual_count",
        )
    }
    for suggestion in result.suggestions:
        rewrite = rewrites.get(suggestion["id"])
        if rewrite is None:
            suggestion["rewrite_status"] = "fallback"
            suggestion["action"] = _default_action(suggestion["method"])
            continue

        suggestion["action"] = rewrite.action
        suggestion["edit_instruction"] = rewrite.edit_instruction
        suggestion["kept_selling_point"] = rewrite.kept_selling_point
        suggestion["alternatives"] = list(rewrite.alternatives)
        suggestion["rewrite_status"] = "ok"
        if rewrite.action == "replace_script" and rewrite.after_text:
            suggestion["after_text"] = rewrite.after_text
            suggestion["estimated_duration_ms"] = rewrite.estimated_duration_ms
            original = suggestion.get("original_duration_ms") or 0
            suggestion["duration_warning"] = bool(
                rewrite.estimated_duration_ms
                and original > 0
                and rewrite.estimated_duration_ms > original * 1.15 + 500
            )
            suggestion["confidence"] = rewrite.confidence
        else:
            suggestion["estimated_duration_ms"] = None
            suggestion["duration_warning"] = False

    result.summary = _summarize(result.findings, result.suggestions, transcript, has_face)
    for key, value in preserved.items():
        if value is not None:
            result.summary[key] = value
    result.summary["rewrite_status"] = "ok"
    result.summary["rewrite_count"] = len(rewrites)
    result.recheck = _recheck_suggestions(result.suggestions)


def _default_action(method: str) -> str:
    if method == "CUT_BROLL":
        return "cut_segment"
    if method == "MANUAL":
        return "manual"
    return "replace_script"


def _merge_visual_events(
    result: ReviewResult,
    events: list[VisualEvent],
    transcript: str,
    has_face: bool | None,
) -> None:
    semantic_status = result.summary.get("semantic_status")
    semantic_count = result.summary.get("semantic_count", 0)
    semantic_error = result.summary.get("semantic_error")
    for event in events:
        evidence = [
            {
                "quote": item.description,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "frame_ms": item.frame_ms,
                "frame_index": item.frame_index,
            }
            for item in event.evidence
        ]
        start_ms = min(item.start_ms for item in event.evidence)
        end_ms = max(item.end_ms for item in event.evidence)
        quote = "；".join(dict.fromkeys(item.description for item in event.evidence))
        finding_id = uuid.uuid4().hex
        original_duration = max(0, end_ms - start_ms)
        result.findings.append(
            {
                "id": finding_id,
                "rule_id": event.rule_id,
                "title": event.title,
                "severity": event.severity,
                "category": event.category,
                "modality": ["visual"],
                "quote": quote,
                "rationale": event.reason,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": event.confidence,
                "needs_manual_time": False,
                "source": "vision",
                "evidence": evidence,
            }
        )
        result.suggestions.append(
            {
                "id": uuid.uuid4().hex,
                "finding_id": finding_id,
                "before_text": quote,
                "after_text": event.suggestion,
                "method": "CUT_BROLL",
                "abc_class": "B",
                "estimated_duration_ms": None,
                "original_duration_ms": original_duration,
                "duration_warning": False,
                "confidence": event.confidence,
                "status": "pending",
            }
        )

    result.summary = _summarize(result.findings, result.suggestions, transcript, has_face)
    result.summary["semantic_status"] = semantic_status
    result.summary["semantic_count"] = semantic_count
    if semantic_error:
        result.summary["semantic_error"] = semantic_error
    result.summary["visual_status"] = "ok"
    result.summary["visual_count"] = len(events)
    result.recheck = _recheck_suggestions(result.suggestions)


def _rule_finding_covered(
    finding: dict[str, Any],
    semantic_ranges: list[tuple[int, int]],
    semantic_quotes: list[str],
) -> bool:
    quote = normalize(finding.get("quote", ""))
    if quote and any(quote in item or item in quote for item in semantic_quotes if item):
        return True

    start_ms = finding.get("start_ms")
    end_ms = finding.get("end_ms")
    if start_ms is None or end_ms is None:
        return False
    for semantic_start, semantic_end in semantic_ranges:
        overlap = max(0, min(end_ms, semantic_end) - max(start_ms, semantic_start))
        shortest = max(1, min(end_ms - start_ms, semantic_end - semantic_start))
        if overlap / shortest >= 0.5:
            return True
    return False


def _merge_same_segment(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], list[str]]] = {}
    order: list[tuple[Any, ...]] = []
    for finding, suggestion in pairs:
        if finding["start_ms"] is None or finding["end_ms"] is None:
            key: tuple[Any, ...] = (finding["id"],)
        else:
            key = (finding["rule_id"], finding["start_ms"], finding["end_ms"])
        if key not in grouped:
            grouped[key] = (finding, suggestion, [finding["quote"]])
            order.append(key)
        else:
            grouped[key][2].append(finding["quote"])

    findings: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for key in order:
        finding, suggestion, quotes = grouped[key]
        unique_quotes = list(dict.fromkeys(quote for quote in quotes if quote))
        merged_finding = dict(finding)
        merged_finding["quote"] = "，".join(unique_quotes)
        if len(unique_quotes) > 1:
            merged_finding["rationale"] = (
                f"{merged_finding['rationale']} 同段合并 {len(unique_quotes)} 处命中。"
            )
        findings.append(merged_finding)
        suggestions.append(suggestion)
    return findings, suggestions


def _drop_contained_findings(
    findings: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop a narrower finding when a same-rule finding covers its quote and range."""
    kept_findings: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for finding in findings:
        quote = normalize(finding["quote"])
        start_ms = finding.get("start_ms")
        end_ms = finding.get("end_ms")
        contained = False
        if quote and start_ms is not None and end_ms is not None:
            for other in findings:
                if other is finding or other["rule_id"] != finding["rule_id"]:
                    continue
                other_quote = normalize(other["quote"])
                other_start = other.get("start_ms")
                other_end = other.get("end_ms")
                if (
                    quote in other_quote
                    and other_start is not None
                    and other_end is not None
                    and other_start <= start_ms
                    and other_end >= end_ms
                    and (len(other_quote) > len(quote) or other_end - other_start > end_ms - start_ms)
                ):
                    contained = True
                    break
        if not contained:
            kept_findings.append(finding)
            kept_ids.add(finding["id"])

    kept_suggestions = [
        suggestion for suggestion in suggestions if suggestion["finding_id"] in kept_ids
    ]
    return kept_findings, kept_suggestions


def _recheck_suggestions(suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    after_texts = [
        suggestion["after_text"]
        for suggestion in suggestions
        if suggestion["method"] in {"TTS_SUBTITLE", "WAN_LIPSYNC"} and suggestion.get("after_text")
    ]
    remaining_matches = scan("\n".join(after_texts))
    remaining = [
        {
            "rule_id": match.rule.id,
            "title": match.rule.title,
            "severity": match.rule.severity,
            "quote": match.quote,
            "match": match.match_text,
        }
        for match in remaining_matches
    ]
    blocking = [item for item in remaining if item["severity"] == "blocking"]
    manual_count = sum(1 for suggestion in suggestions if suggestion.get("method") == "MANUAL")
    return {
        "status": "passed" if not blocking else "needs_revision",
        "remaining": remaining,
        "blocking_count": len(blocking),
        "manual_count": manual_count,
        "note": "词面复检不等于平台最终审核；仍需人工看语境和画面。",
    }


def _summarize(
    findings: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
    transcript: str,
    has_face: bool | None,
) -> dict[str, Any]:
    return {
        "total": len(findings),
        "blocking": sum(1 for finding in findings if finding["severity"] == "blocking"),
        "suggest": sum(1 for finding in findings if finding["severity"] == "suggest"),
        "info": sum(1 for finding in findings if finding["severity"] == "info"),
        "manual": sum(1 for suggestion in suggestions if suggestion["method"] == "MANUAL"),
        "a_count": sum(1 for suggestion in suggestions if suggestion["abc_class"] == "A"),
        "b_count": sum(1 for suggestion in suggestions if suggestion["abc_class"] == "B"),
        "c_count": sum(1 for suggestion in suggestions if suggestion["abc_class"] == "C"),
        "semantic_count": sum(1 for finding in findings if finding.get("source") == "semantic"),
        "visual_count": sum(1 for finding in findings if finding.get("source") == "vision"),
        "rewrite_count": sum(1 for suggestion in suggestions if suggestion.get("rewrite_status") == "ok"),
        "transcript_chars": len(transcript),
        "has_face": has_face,
        "rule_version": rule_version(),
    }


def store_review(material_id: str, result: ReviewResult) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM suggestion WHERE finding_id IN (SELECT id FROM finding WHERE material_id = ?)",
            (material_id,),
        )
        conn.execute("DELETE FROM finding WHERE material_id = ?", (material_id,))
        conn.execute("DELETE FROM recheck WHERE material_id = ?", (material_id,))

        for finding in result.findings:
            conn.execute(
                """
                INSERT INTO finding
                (id, material_id, rule_id, title, severity, category, modality, quote,
                 rationale, start_ms, end_ms, confidence, needs_manual_time, source, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding["id"],
                    material_id,
                    finding["rule_id"],
                    finding["title"],
                    finding["severity"],
                    finding["category"],
                    json.dumps(finding["modality"], ensure_ascii=False),
                    finding["quote"],
                    finding["rationale"],
                    finding["start_ms"],
                    finding["end_ms"],
                    finding["confidence"],
                    int(finding["needs_manual_time"]),
                    finding.get("source", "rule"),
                    json.dumps(
                        finding.get("evidence")
                        or [
                            {
                                "quote": finding["quote"],
                                "start_ms": finding["start_ms"],
                                "end_ms": finding["end_ms"],
                            }
                        ],
                        ensure_ascii=False,
                    ),
                ),
            )

        timestamp = now_iso()
        for suggestion in result.suggestions:
            conn.execute(
                """
                INSERT INTO suggestion
                (id, finding_id, before_text, after_text, method, abc_class,
                 estimated_duration_ms, original_duration_ms, duration_warning,
                 confidence, status, action, edit_instruction, kept_selling_point,
                 alternatives_json, rewrite_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion["id"],
                    suggestion["finding_id"],
                    suggestion["before_text"],
                    suggestion["after_text"],
                    suggestion["method"],
                    suggestion["abc_class"],
                    suggestion["estimated_duration_ms"],
                    suggestion["original_duration_ms"],
                    int(suggestion["duration_warning"]),
                    suggestion["confidence"],
                    suggestion["status"],
                    suggestion.get("action") or _default_action(suggestion["method"]),
                    suggestion.get("edit_instruction"),
                    suggestion.get("kept_selling_point"),
                    json.dumps(suggestion.get("alternatives") or [], ensure_ascii=False),
                    suggestion.get("rewrite_status"),
                    timestamp,
                    timestamp,
                ),
            )

        conn.execute(
            "INSERT INTO recheck (id, material_id, status, remaining_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                material_id,
                result.recheck["status"],
                json.dumps(result.recheck, ensure_ascii=False),
                timestamp,
            ),
        )
        conn.execute(
            """
            UPDATE material
            SET status = 'reviewed', semantic_status = ?, semantic_error = ?,
                visual_status = ?, visual_error = ?, rewrite_status = ?, rewrite_error = ?
            WHERE id = ?
            """,
            (
                result.summary.get("semantic_status"),
                result.summary.get("semantic_error"),
                result.summary.get("visual_status"),
                result.summary.get("visual_error"),
                result.summary.get("rewrite_status"),
                result.summary.get("rewrite_error"),
                material_id,
            ),
        )


def get_report(material_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        material = conn.execute("SELECT * FROM material WHERE id = ?", (material_id,)).fetchone()
        if material is None:
            return None

        rows = conn.execute(
            """
            SELECT f.*, s.id AS suggestion_id, s.before_text, s.after_text, s.method,
                   s.abc_class, s.estimated_duration_ms, s.original_duration_ms,
                   s.duration_warning, s.confidence AS suggestion_confidence, s.status AS suggestion_status,
                   s.action, s.edit_instruction, s.kept_selling_point, s.alternatives_json, s.rewrite_status
            FROM finding f
            LEFT JOIN suggestion s ON s.finding_id = f.id
            WHERE f.material_id = ?
            ORDER BY
                CASE f.severity WHEN 'blocking' THEN 0 WHEN 'suggest' THEN 1 ELSE 2 END,
                (f.start_ms IS NULL), f.start_ms, f.id
            """,
            (material_id,),
        ).fetchall()

        recheck_row = conn.execute(
            "SELECT * FROM recheck WHERE material_id = ? ORDER BY created_at DESC LIMIT 1",
            (material_id,),
        ).fetchone()

    findings: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for row in rows:
        suggestion = None
        if row["suggestion_id"]:
            suggestion = {
                "id": row["suggestion_id"],
                "finding_id": row["id"],
                "before_text": row["before_text"],
                "after_text": row["after_text"],
                "method": row["method"],
                "abc_class": row["abc_class"],
                "estimated_duration_ms": row["estimated_duration_ms"],
                "original_duration_ms": row["original_duration_ms"],
                "duration_warning": bool(row["duration_warning"]),
                "confidence": row["suggestion_confidence"],
                "status": row["suggestion_status"],
                "action": row["action"] or _default_action(row["method"]),
                "edit_instruction": row["edit_instruction"],
                "kept_selling_point": row["kept_selling_point"],
                "alternatives": json.loads(row["alternatives_json"]) if row["alternatives_json"] else [],
                "rewrite_status": row["rewrite_status"],
            }
            suggestions.append(suggestion)
        findings.append(
            {
                "id": row["id"],
                "rule_id": row["rule_id"],
                "title": row["title"],
                "severity": row["severity"],
                "category": row["category"],
                "modality": json.loads(row["modality"]),
                "quote": row["quote"],
                "rationale": row["rationale"],
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "confidence": row["confidence"],
                "needs_manual_time": bool(row["needs_manual_time"]),
                "source": row["source"] or "rule",
                "evidence": json.loads(row["evidence_json"]) if row["evidence_json"] else [],
                "suggestion": suggestion,
            }
        )

    summary = _summarize(findings, suggestions, material["transcript"] or "", _bool_or_none(material["has_face"]))
    summary["transcript_source"] = material["transcript_source"]
    summary["word_count"] = material["word_count"]
    summary["asr_error"] = material["asr_error"]
    summary["semantic_status"] = material["semantic_status"]
    summary["semantic_error"] = material["semantic_error"]
    summary["visual_status"] = material["visual_status"]
    summary["visual_error"] = material["visual_error"]
    summary["rewrite_status"] = material["rewrite_status"]
    summary["rewrite_error"] = material["rewrite_error"]
    summary["render_status"] = material["render_status"]
    summary["render_error"] = material["render_error"]
    ignored_findings = {
        suggestion["finding_id"]
        for suggestion in suggestions
        if suggestion.get("status") == "false_positive"
    }
    summary["false_positive"] = len(ignored_findings)
    if ignored_findings:
        for severity in ("blocking", "suggest", "info"):
            summary[severity] = sum(
                1
                for finding in findings
                if finding["severity"] == severity and finding["id"] not in ignored_findings
            )
    recheck = None
    if recheck_row:
        recheck = json.loads(recheck_row["remaining_json"])
        recheck["status"] = recheck_row["status"]
        recheck["created_at"] = recheck_row["created_at"]

    segments: list[dict[str, Any]] = []
    srt_path = material["srt_path"]
    if srt_path and Path(srt_path).is_file():
        segments = [
            {
                "index": segment.index,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
            }
            for segment in parse_srt(read_text(Path(srt_path)))
        ]

    return {
        "material": {
            "id": material["id"],
            "batch_id": material["batch_id"],
            "filename": material["filename"],
            "category": material["category"],
            "has_face": _bool_or_none(material["has_face"]),
            "status": material["status"],
            "duration_ms": material["duration_ms"],
            "transcript_source": material["transcript_source"],
            "asr_model": material["asr_model"],
            "asr_error": material["asr_error"],
            "word_count": material["word_count"],
            "semantic_status": material["semantic_status"],
            "semantic_error": material["semantic_error"],
            "visual_status": material["visual_status"],
            "visual_error": material["visual_error"],
            "rewrite_status": material["rewrite_status"],
            "rewrite_error": material["rewrite_error"],
            "render_status": material["render_status"],
            "render_output_path": material["render_output_path"],
            "render_error": material["render_error"],
            "has_srt": bool(material["srt_path"]),
            "created_at": material["created_at"],
        },
        "transcript": material["transcript"] or "",
        "segments": segments,
        "summary": summary,
        "findings": findings,
        "recheck": recheck,
    }


def update_suggestion(
    suggestion_id: str,
    after_text: str | None = None,
    method: str | None = None,
    abc_class: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM suggestion WHERE id = ?", (suggestion_id,)).fetchone()
        if row is None:
            return None
        new_text = after_text if after_text is not None else row["after_text"]
        new_method = method if method is not None else row["method"]
        new_class = abc_class if abc_class is not None else row["abc_class"]
        new_status = status if status is not None else row["status"]
        new_action = "replace_script" if after_text is not None else (row["action"] or _default_action(new_method))
        new_rewrite_status = "manual" if after_text is not None else row["rewrite_status"]
        estimated = (
            None
            if new_method in {"MANUAL", "CUT_BROLL"}
            else estimate_duration_ms(new_text)
        )
        warning = bool(
            estimated is not None
            and row["original_duration_ms"]
            and estimated > row["original_duration_ms"] * 1.15 + 500
        )
        conn.execute(
            """
            UPDATE suggestion
            SET after_text = ?, method = ?, abc_class = ?, estimated_duration_ms = ?,
                duration_warning = ?, status = ?, action = ?, rewrite_status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_text,
                new_method,
                new_class,
                estimated,
                int(warning),
                new_status,
                new_action,
                new_rewrite_status,
                now_iso(),
                suggestion_id,
            ),
        )

    return get_suggestion(suggestion_id)


def update_suggestion_rewrite(
    suggestion_id: str,
    rewrite: RewriteResult,
) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM suggestion WHERE id = ?", (suggestion_id,)).fetchone()
        if row is None:
            return None
        after_text = (
            rewrite.after_text
            if rewrite.action == "replace_script" and rewrite.after_text
            else row["after_text"]
        )
        estimated = rewrite.estimated_duration_ms if rewrite.action == "replace_script" else None
        original = row["original_duration_ms"] or 0
        warning = bool(
            estimated is not None
            and original > 0
            and estimated > original * 1.15 + 500
        )
        conn.execute(
            """
            UPDATE suggestion
            SET after_text = ?, estimated_duration_ms = ?, duration_warning = ?,
                action = ?, edit_instruction = ?, kept_selling_point = ?,
                alternatives_json = ?, rewrite_status = 'refreshed', updated_at = ?
            WHERE id = ?
            """,
            (
                after_text,
                estimated,
                int(warning),
                rewrite.action,
                rewrite.edit_instruction,
                rewrite.kept_selling_point,
                json.dumps(list(rewrite.alternatives), ensure_ascii=False),
                now_iso(),
                suggestion_id,
            ),
        )
    return get_suggestion(suggestion_id)


def get_suggestion(suggestion_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM suggestion WHERE id = ?", (suggestion_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "finding_id": row["finding_id"],
        "before_text": row["before_text"],
        "after_text": row["after_text"],
        "method": row["method"],
        "abc_class": row["abc_class"],
        "estimated_duration_ms": row["estimated_duration_ms"],
        "original_duration_ms": row["original_duration_ms"],
        "duration_warning": bool(row["duration_warning"]),
        "confidence": row["confidence"],
        "status": row["status"],
        "action": row["action"] or _default_action(row["method"]),
        "edit_instruction": row["edit_instruction"],
        "kept_selling_point": row["kept_selling_point"],
        "alternatives": json.loads(row["alternatives_json"]) if row["alternatives_json"] else [],
        "rewrite_status": row["rewrite_status"],
    }


def recheck_material(material_id: str) -> dict[str, Any]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT s.after_text, s.method FROM suggestion s
            JOIN finding f ON s.finding_id = f.id
            WHERE f.material_id = ?
            """,
            (material_id,),
        ).fetchall()
        suggestions = [
            {"after_text": row["after_text"], "method": row["method"]}
            for row in rows
        ]
        result = _recheck_suggestions(suggestions)
        conn.execute("DELETE FROM recheck WHERE material_id = ?", (material_id,))
        conn.execute(
            "INSERT INTO recheck (id, material_id, status, remaining_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, material_id, result["status"], json.dumps(result, ensure_ascii=False), now_iso()),
        )
    return result


def confirm_material(material_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        material = conn.execute("SELECT id FROM material WHERE id = ?", (material_id,)).fetchone()
        if material is None:
            return None
        conn.execute(
            """
            UPDATE suggestion SET status = 'accepted', updated_at = ?
            WHERE finding_id IN (SELECT id FROM finding WHERE material_id = ?)
            """,
            (now_iso(), material_id),
        )
        conn.execute("UPDATE material SET status = 'confirmed' WHERE id = ?", (material_id,))
    return get_report(material_id)


def dry_run_plan(material_id: str) -> dict[str, Any] | None:
    from .edit import build_edit_plan

    plan = build_edit_plan(material_id)
    if plan is None:
        return None
    return {
        **plan,
        "plan": plan["operations"],
        "estimated_cost": 0.0,
        "cost_note": "A/B 类本地渲染，成本以 TTS 实际用量为准",
    }


def fmt_ms(value: int | None) -> str:
    if value is None:
        return "未定位片段"
    seconds = value / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"

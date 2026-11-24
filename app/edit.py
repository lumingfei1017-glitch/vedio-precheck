from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from .asr import probe_duration_ms
from .config import (
    DATA_DIR,
    FFMPEG_PATH,
    FFPROBE_PATH,
    RENDER_ENABLED,
    AUTO_CUT_ENABLED,
    SEGMENT_PRESERVE_BGM,
    SEGMENT_REPLACE_ENABLED,
    TTS_ROUTE_ENABLED,
    TTS_RATE,
    TTS_VOICE,
    VOICE_CACHE_DIR,
    VOICE_REFERENCE_MAX_MS,
    VOICE_REFERENCE_MIN_MS,
    VOICE_REFERENCE_TARGET_MS,
)
from .review import get_report
from .srt import Segment, format_timestamp, parse_srt, read_text
from .tts import TTSError, fit_audio, synthesize

ProgressCallback = Callable[[str, int], None]

# 一个自动删段单元最多覆盖多长，以及一条删段最多占全片多少。
# 2026-11-22 实测：一个语义事件把 6 句证据并成 28s，再与相邻视觉问题串成 43s，
# 90 秒素材被删掉一半，人工修不动。删段必须按证据簇拆开，并限制合并长度。
MAX_CUT_UNIT_MS = 8000
MAX_MERGED_CUT_MS = 12000
MAX_AUTO_CUT_RATIO = 0.30


class RenderError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def build_edit_plan(material_id: str) -> dict[str, Any] | None:
    report = get_report(material_id)
    if report is None:
        return None
    material = report["material"]
    duration_ms = material["duration_ms"] or 0
    replace_ops: list[dict[str, Any]] = []
    cut_ops: list[dict[str, Any]] = []
    manual_ops: list[dict[str, Any]] = []
    warnings: list[str] = []

    for finding in report["findings"]:
        suggestion = finding.get("suggestion")
        if not suggestion or suggestion.get("status") == "false_positive":
            continue
        if finding.get("needs_manual_time") or finding.get("start_ms") is None or finding.get("end_ms") is None:
            warnings.append(f"{finding['rule_id']} 缺少时间区间，修改前需要人工标时间")
            continue
        start_ms = int(finding["start_ms"])
        end_ms = int(finding["end_ms"])
        action = suggestion.get("action") or (
            "cut_segment" if suggestion["method"] == "CUT_BROLL" else "manual" if suggestion["method"] == "MANUAL" else "replace_script"
        )
        item = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "suggestion_id": suggestion["id"],
            "finding_ids": [finding["id"]],
            "confidence": float(
                suggestion.get("confidence") or finding.get("confidence") or 0
            ),
        }
        if action == "replace_script" and suggestion.get("after_text"):
            item.update(
                {
                    "text": suggestion["after_text"],
                    "method": suggestion["method"],
                    "abc_class": suggestion.get("abc_class"),
                    "estimated_duration_ms": suggestion.get("estimated_duration_ms") or 0,
                }
            )
            replace_ops.extend(_expand_replace_op_by_evidence(item, finding))
        elif action == "cut_segment":
            item.update(
                {
                    "instruction": suggestion.get("edit_instruction")
                    or suggestion.get("after_text")
                    or "删除该片段",
                    "evidence": finding.get("evidence") or [],
                }
            )
            confidence = float(
                suggestion.get("confidence") or finding.get("confidence") or 0
            )
            unverified = "未在该时段复现" in str(finding.get("rationale") or "")
            if unverified or confidence < 0.8:
                item["reason"] = (
                    "视觉判据置信度不足"
                    if confidence < 0.8
                    else "密集复核未复现该画面"
                ) + "，不自动删段，需人工确认"
                manual_ops.append(item)
                continue
            cut_ops.append(item)
        else:
            item.update(
                {
                    "reason": suggestion.get("edit_instruction")
                    or suggestion.get("after_text")
                    or "需要人工处理",
                }
            )
            manual_ops.append(item)

    if not AUTO_CUT_ENABLED:
        for op in cut_ops:
            manual_ops.append(
                {
                    **op,
                    "reason": "需人工在剪映处理："
                    + str(op.get("instruction") or "删除或替换该画面")[:120],
                }
            )
        cut_ops = []

    replace_ops, cross_manual, cross_warnings = _align_replace_ops_to_shots(
        material_id,
        material,
        replace_ops,
    )
    manual_ops.extend(cross_manual)
    warnings.extend(cross_warnings)
    replace_ops = _snap_replace_ops_to_shots(material_id, material, replace_ops)
    replace_ops = _merge_replace_ops(replace_ops)
    if not TTS_ROUTE_ENABLED:
        kept_ops: list[dict[str, Any]] = []
        for op in replace_ops:
            if op.get("route") == "wan":
                kept_ops.append(op)
            else:
                manual_ops.append(
                    {
                        **op,
                        "reason": "转人工："
                        + (op.get("route_reason") or "当前只自动执行 Wan 口播改写"),
                    }
                )
        replace_ops = kept_ops
    cut_ops = _split_cut_ops_by_evidence(cut_ops)
    cut_ops = _merge_cut_ops(cut_ops)
    replace_ops, cut_ops, conflict_warnings = _resolve_cut_conflicts(
        replace_ops, cut_ops
    )
    warnings.extend(conflict_warnings)
    replace_ops, cut_ops, absorbed_warnings = _absorb_short_replace_remnants(
        replace_ops,
        cut_ops,
        duration_ms,
    )
    cut_ops = _merge_cut_ops(cut_ops)
    cut_ops, oversize_manual = _cap_oversize_cuts(cut_ops, duration_ms)
    manual_ops.extend(oversize_manual)
    warnings.extend(absorbed_warnings)
    cut_total = sum(max(0, op["end_ms"] - op["start_ms"]) for op in cut_ops)
    base_after_cut = max(0, duration_ms - cut_total)
    extra = sum(
        max(0, (op.get("estimated_duration_ms") or 0) - (op["end_ms"] - op["start_ms"]))
        for op in replace_ops
    )
    estimated = base_after_cut + extra
    mode = (
        "mixed_segment"
        if replace_ops and cut_ops and SEGMENT_REPLACE_ENABLED
        else "segment_replace"
        if replace_ops and not cut_ops and SEGMENT_REPLACE_ENABLED
        else "mixed"
        if replace_ops and cut_ops
        else "A"
        if replace_ops
        else "B"
        if cut_ops
        else "manual"
        if manual_ops
        else "none"
    )
    route_counts = {
        route: sum(1 for op in replace_ops if op.get("route") == route)
        for route in ("wan", "tts", "manual")
    }
    if route_counts["wan"]:
        warnings.append(
            f"{route_counts['wan']} 个替换单元走 Wan 整镜重生成，每次约 3.5–4 分钟"
        )
    if route_counts["tts"]:
        warnings.append(
            f"{route_counts['tts']} 个替换单元走局部换音轨（短镜或无脸镜），画面不动"
        )
    if route_counts["manual"]:
        warnings.append(
            f"{route_counts['manual']} 个替换单元镜头类型待判定，提交前需人工确认"
        )
    if replace_ops and mode in {"segment_replace", "mixed_segment"}:
        scope = "先删段，再只替换剩余目标区间" if mode == "mixed_segment" else "只替换目标区间"
        warnings.append(f"当前为片段级音频替换：{scope}；区间外保留原音轨")
        warnings.append("字幕按原时间轴重排，TTS 实际节奏可能略有偏差")
    elif replace_ops:
        warnings.append("当前 A 类为整轨 TTS 替换，原视频 BGM/SFX 不保留")
        warnings.append("字幕按原时间轴重排，TTS 实际节奏可能略有偏差")
    if cut_ops:
        warnings.append("当前 B 类为删段拼接，B-roll 覆盖尚未接入")
    if report["material"].get("srt_path"):
        warnings.append("原视频如含硬字幕，当前无法自动擦除")
    if manual_ops:
        warnings.append(f"{len(manual_ops)} 项需要人工处理，不会自动渲染")

    operations = [
        {**op, "op": "replace_script"} for op in replace_ops
    ] + [
        {**op, "op": "cut"} for op in cut_ops
    ] + [
        {**op, "op": "manual"} for op in manual_ops
    ]
    return {
        "material_id": material_id,
        "dry_run": True,
        "mode": mode,
        "operations": operations,
        "cut_intervals": [
            {"start_ms": op["start_ms"], "end_ms": op["end_ms"]} for op in cut_ops
        ],
        "estimated_duration_ms": estimated,
        "route_counts": route_counts,
        "warnings": list(dict.fromkeys(warnings)),
    }


def execute_edit_plan(
    material_id: str,
    plan: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not RENDER_ENABLED:
        raise RenderError("视频渲染未启用")
    plan = plan or build_edit_plan(material_id)
    if plan is None:
        raise RenderError("素材不存在")
    report = get_report(material_id)
    if report is None:
        raise RenderError("素材不存在")
    material = report["material"]
    video_path = Path(_load_video_path(material_id))
    if not video_path.is_file():
        raise RenderError("视频文件不存在")
    if FFMPEG_PATH is None:
        raise RenderError("未找到 FFmpeg")

    work_dir = video_path.parent / "render"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "output.mp4"
    srt_path = work_dir / "output.srt"
    for old in (output_path, srt_path, work_dir / "tts.mp3"):
        old.unlink(missing_ok=True)

    wan_ops = [
        op
        for op in plan.get("operations", [])
        if op.get("op") == "replace_script" and op.get("route") == "wan"
    ]
    if wan_ops and not TTS_ROUTE_ENABLED:
        return _execute_wan_replace(
            material_id,
            video_path,
            wan_ops,
            srt_path,
            output_path,
            work_dir,
            progress,
            plan,
        )

    duration_ms = material["duration_ms"] or probe_duration_ms(video_path)
    cut_intervals = [
        (int(item["start_ms"]), int(item["end_ms"]))
        for item in plan.get("cut_intervals", [])
    ]
    kept = _kept_intervals(duration_ms, cut_intervals)
    if not kept:
        raise RenderError("删除后没有剩余视频片段")
    if progress:
        progress("准备素材", 10)

    base_video = _cut_video(video_path, kept, work_dir, progress)
    cut_duration = sum(end - start for start, end in kept)
    segments = _load_segments(material_id)
    replace_ops = [op for op in plan.get("operations", []) if op.get("op") == "replace_script"]
    cut_ops = [op for op in plan.get("operations", []) if op.get("op") == "cut"]
    if not replace_ops and not cut_ops:
        raise RenderError("没有可执行的修改操作")
    _write_adjusted_srt(
        segments,
        cut_intervals,
        replace_ops,
        srt_path,
    )

    mode = str(plan.get("mode") or "")
    if mode in {"segment_replace", "mixed_segment"}:
        active_replace_ops = replace_ops
        segment_warnings = list(plan.get("warnings", []))
        if mode == "mixed_segment":
            active_replace_ops, mapping_warnings = _map_replace_ops_after_cuts(
                replace_ops,
                cut_intervals,
                duration_ms,
            )
            segment_warnings.extend(mapping_warnings)
        if not active_replace_ops:
            if not cut_ops:
                raise RenderError("替换区间都被删段覆盖，没有可执行的片段替换")
            if progress:
                progress("剪辑拼接", 72)
            if base_video == video_path:
                shutil.copy2(base_video, output_path)
            else:
                base_video.replace(output_path)
            return {
                "material_id": material_id,
                "output_path": str(output_path),
                "srt_path": str(srt_path),
                "duration_ms": cut_duration,
                "mode": "B",
                "warnings": segment_warnings,
            }
        return _execute_segment_replace(
            material_id,
            base_video,
            active_replace_ops,
            srt_path,
            output_path,
            cut_duration,
            work_dir,
            progress,
            segment_warnings,
            mode=mode,
        )

    if replace_ops:
        if progress:
            progress("生成配音", 65)
        narration = _build_narration(segments, replace_ops)
        tts_path = work_dir / "tts.mp3"
        _synthesize_tts(narration, tts_path)
        tts_duration = probe_duration_ms(tts_path)
        if progress:
            progress("音频合成", 82)
        _mux_tts(base_video, tts_path, output_path, cut_duration, tts_duration)
    else:
        if progress:
            progress("剪辑拼接", 72)
        if base_video == video_path:
            shutil.copy2(base_video, output_path)
        else:
            base_video.replace(output_path)

    if progress:
        progress("字幕", 94)
    if not srt_path.is_file():
        srt_path.write_text("", encoding="utf-8")
    return {
        "material_id": material_id,
        "output_path": str(output_path),
        "srt_path": str(srt_path),
        "duration_ms": cut_duration,
        "mode": plan.get("mode"),
        "warnings": plan.get("warnings", []),
    }


def _wan_preflight(op: dict[str, Any], original_text: str) -> str | None:
    """按 Wan 3.0 的硬约束判定能不能改；不合格的直接跳过，不浪费一次调用。"""
    from .shots import WAN_MIN_SECONDS
    from .wan import MAX_WAN_CHARS_PER_SECOND, planned_duration

    start = int(op["start_ms"])
    end = int(op["end_ms"])
    seconds = (end - start) / 1000
    if seconds < WAN_MIN_SECONDS:
        return f"镜头 {seconds:.1f}s 短于 Wan 最短 {WAN_MIN_SECONDS:.0f}s"
    text = str(op.get("text") or "").strip()
    if not text:
        return "缺少新台词"
    duration = planned_duration(seconds)
    if duration > seconds + 1e-6:
        return "Wan 输出时长会超过参考镜头"
    capacity = int(duration * MAX_WAN_CHARS_PER_SECOND)
    if len(text) > capacity:
        return f"文案 {len(text)} 字超过 {duration}s 容量 {capacity} 字"
    floor = int(seconds * 3.75)
    if len(text) < floor:
        return f"文案 {len(text)} 字偏短（下限 {floor} 字），时长会缩短超过 25%"
    def clean(value: str) -> str:
        return re.sub(r"[\s，。！？；：、,.!?;:：]", "", value or "")

    new_text = clean(text)
    old_text = clean(original_text)
    if len(new_text) >= 5 and len(old_text) >= 5 and (
        new_text[:5] == old_text[:5] or new_text[-5:] == old_text[-5:]
    ):
        return "新文案首尾与原句相同，容易复读原句"
    return None


def _execute_wan_replace(
    material_id: str,
    video_path: Path,
    ops: list[dict[str, Any]],
    srt_path: Path,
    output_path: Path,
    work_dir: Path,
    progress: ProgressCallback | None,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """逐镜跑 Wan 重生成；任何一段失败都保留原样，不影响其他段。"""
    from .replace import replace_shot

    ordered = sorted(ops, key=lambda item: int(item["start_ms"]))
    total = max(1, len(ordered))
    current = video_path
    segments = _load_segments(material_id)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for index, op in enumerate(ordered, start=1):
        original = "".join(
            segment.text
            for segment in segments
            if segment.end_ms > int(op["start_ms"]) and segment.start_ms < int(op["end_ms"])
        )
        blocked = _wan_preflight(op, original)
        if blocked:
            skipped.append(
                {
                    "start_ms": int(op["start_ms"]),
                    "end_ms": int(op["end_ms"]),
                    "reason": blocked,
                }
            )
            continue
        stage = work_dir / f"wan-{index:02d}.mp4"
        if progress:
            progress(f"Wan 重生成 {index}/{total}", 8 + int((index - 1) / total * 82))
        try:
            result = replace_shot(
                current,
                int(op["start_ms"]) / 1000,
                int(op["end_ms"]) / 1000,
                str(op.get("text") or ""),
                stage,
                work_dir=work_dir / f"wan-{index:02d}",
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "start_ms": int(op["start_ms"]),
                    "end_ms": int(op["end_ms"]),
                    "reason": str(exc)[:200],
                }
            )
            continue
        current = stage
        applied.append(
            {
                "start_ms": int(op["start_ms"]),
                "end_ms": int(op["end_ms"]),
                "text": op.get("text"),
                "padded_ms": result.get("padded_ms"),
            }
        )

    if progress:
        progress("输出视频", 92)
    if current == video_path:
        shutil.copy2(video_path, output_path)
    else:
        if output_path.exists():
            output_path.unlink()
        shutil.copy2(current, output_path)

    _write_adjusted_srt(segments, [], applied, srt_path)

    warnings = list(plan.get("warnings", []))
    if skipped:
        warnings.append(f"{len(skipped)} 段未能 Wan 重生成，已保留原样")
    return {
        "material_id": material_id,
        "output_path": str(output_path),
        "srt_path": str(srt_path),
        "duration_ms": probe_duration_ms(output_path),
        "mode": "wan_replace" if applied else "unchanged",
        "applied": applied,
        "skipped": skipped,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _execute_segment_replace(
    material_id: str,
    video_path: Path,
    replace_ops: list[dict[str, Any]],
    srt_path: Path,
    output_path: Path,
    duration_ms: int,
    work_dir: Path,
    progress: ProgressCallback | None,
    warnings: list[str],
    *,
    mode: str = "segment_replace",
) -> dict[str, Any]:
    current_video = video_path
    voice_sources: list[dict[str, Any]] = []
    all_warnings = list(warnings)
    background_audio: Path | None = None
    vocals_audio: Path | None = None
    bgm_preserved = False
    if SEGMENT_PRESERVE_BGM and replace_ops:
        if progress:
            progress("分离原人声与 BGM", 14)
        try:
            from .separate import separate_audio

            stems = separate_audio(video_path)
            background_audio = stems.background
            vocals_audio = stems.vocals
            bgm_preserved = True
        except Exception as exc:  # noqa: BLE001
            all_warnings.append(
                f"BGM 未保留：人声分离不可用（{str(exc)[:180]}）"
            )

    reference_audio: Path | None = None
    if replace_ops:
        stable_reference = (
            VOICE_CACHE_DIR / f"{material_id}-voice-reference-{VOICE_REFERENCE_TARGET_MS}.wav"
        )
        try:
            if not stable_reference.is_file() or stable_reference.stat().st_size < 100_000:
                from .separate import separate_audio

                original_video = Path(_load_video_path(material_id))
                original_stems = separate_audio(original_video)
                stable_reference = extract_stable_voice_reference(
                    original_stems.vocals,
                    stable_reference,
                )
            reference_audio = stable_reference
        except RenderError as exc:
            all_warnings.append(f"未生成克隆参考音：{exc}")
            first_op = min(replace_ops, key=lambda item: int(item["start_ms"]))
            try:
                reference_audio = extract_voice_reference(
                    video_path,
                    int(first_op["start_ms"]),
                    int(first_op["end_ms"]),
                    duration_ms,
                    work_dir / "voice-reference.wav",
                    source_audio=vocals_audio,
                )
            except RenderError:
                reference_audio = None
        except Exception as exc:  # noqa: BLE001
            all_warnings.append(f"稳定参考音不可用，改用临时参考：{str(exc)[:180]}")
            first_op = min(replace_ops, key=lambda item: int(item["start_ms"]))
            try:
                reference_audio = extract_voice_reference(
                    video_path,
                    int(first_op["start_ms"]),
                    int(first_op["end_ms"]),
                    duration_ms,
                    work_dir / "voice-reference.wav",
                    source_audio=vocals_audio,
                )
            except RenderError:
                reference_audio = None

    for index, op in enumerate(sorted(replace_ops, key=lambda item: item["start_ms"])):
        start_ms = int(op["start_ms"])
        end_ms = int(op["end_ms"])
        target_ms = max(1, end_ms - start_ms)
        if progress:
            progress(f"片段配音 {index + 1}/{len(replace_ops)}", 30 + int(index / max(1, len(replace_ops)) * 45))
        raw_audio = work_dir / f"segment-{index:02d}-tts.mp3"
        fitted_audio = work_dir / f"segment-{index:02d}-fit.mp3"
        try:
            tts_result = synthesize(
                op["text"],
                raw_audio,
                reference_audio=reference_audio,
            )
            fit_audio(raw_audio, fitted_audio, target_ms)
        except TTSError as exc:
            raise RenderError(f"片段 {index + 1} TTS 失败：{exc}") from exc
        voice_sources.append(
            {
                "provider": tts_result.provider,
                "voice_id": tts_result.voice_id,
                "voice_source": tts_result.voice_source,
            }
        )
        if tts_result.voice_source != "cloned":
            all_warnings.append(
                f"片段 {index + 1} 未使用原人物音色，当前为 Edge 测试音色"
            )
        all_warnings.extend(tts_result.warnings)

        staged = work_dir / f"segment-{index:02d}-video.mp4"
        replace_audio_interval(
            current_video,
            fitted_audio,
            start_ms,
            end_ms,
            duration_ms,
            staged,
            background_audio=background_audio,
        )
        current_video = staged

    if current_video == video_path:
        raise RenderError("没有发生片段替换")
    if current_video != output_path:
        if output_path.exists():
            output_path.unlink()
        current_video.replace(output_path)
    if not srt_path.is_file():
        srt_path.write_text("", encoding="utf-8")
    if progress:
        progress("字幕", 94)
    return {
        "material_id": material_id,
        "output_path": str(output_path),
        "srt_path": str(srt_path),
        "duration_ms": duration_ms,
        "mode": mode,
        "warnings": list(dict.fromkeys(all_warnings)),
        "voice_sources": voice_sources,
        "bgm_preserved": bgm_preserved,
        "voice_reference": str(reference_audio) if reference_audio else None,
    }


def replace_audio_interval(
    video_path: Path,
    tts_path: Path,
    start_ms: int,
    end_ms: int,
    duration_ms: int,
    output_path: Path,
    *,
    background_audio: Path | None = None,
) -> None:
    if FFMPEG_PATH is None:
        raise RenderError("未找到 FFmpeg")
    parts: list[str] = []
    inputs: list[str] = []
    if start_ms > 0:
        parts.append(
            f"[0:a]atrim=0:{start_ms / 1000:.3f},aresample=48000,asetpts=PTS-STARTPTS[a0]"
        )
        inputs.append("[a0]")
    if background_audio is not None:
        parts.append(
            f"[2:a]atrim={start_ms / 1000:.3f}:{end_ms / 1000:.3f},"
            "aresample=48000,asetpts=PTS-STARTPTS[bg]"
        )
        parts.append("[1:a]aresample=48000,asetpts=PTS-STARTPTS[voice]")
        parts.append(
            "[bg][voice]amix=inputs=2:duration=first:normalize=0[a1]"
        )
    else:
        parts.append("[1:a]aresample=48000,asetpts=PTS-STARTPTS[a1]")
    inputs.append("[a1]")
    if end_ms < duration_ms:
        parts.append(
            f"[0:a]atrim={end_ms / 1000:.3f}:{duration_ms / 1000:.3f},aresample=48000,asetpts=PTS-STARTPTS[a2]"
        )
        inputs.append("[a2]")
    filter_complex = ";".join(parts) + f";{''.join(inputs)}concat=n={len(inputs)}:v=0:a=1[a]"
    command = [
        str(FFMPEG_PATH),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(tts_path),
    ]
    if background_audio is not None:
        command.extend(["-i", str(background_audio)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RenderError(f"片段音频回贴失败：{completed.stderr[-400:]}")


def extract_voice_reference(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    duration_ms: int,
    output_path: Path,
    *,
    source_audio: Path | None = None,
) -> Path | None:
    """Extract a clean-ish same-video voice sample, preferring outside the edit."""
    if FFMPEG_PATH is None:
        raise RenderError("未找到 FFmpeg，无法提取克隆参考音")
    duration_ms = max(duration_ms, end_ms)
    before_ms = max(0, start_ms)
    after_ms = max(0, duration_ms - end_ms)
    if after_ms >= before_ms and after_ms > 0:
        ref_start = end_ms
        ref_end = min(duration_ms, end_ms + VOICE_REFERENCE_MAX_MS)
    elif before_ms > 0:
        ref_start = max(0, start_ms - VOICE_REFERENCE_MAX_MS)
        ref_end = start_ms
    else:
        ref_start = 0
        ref_end = min(duration_ms, VOICE_REFERENCE_MIN_MS)

    if ref_end - ref_start < VOICE_REFERENCE_MIN_MS:
        ref_start = max(0, min(ref_start, duration_ms - VOICE_REFERENCE_MIN_MS))
        ref_end = min(duration_ms, ref_start + VOICE_REFERENCE_MAX_MS)
    if ref_end - ref_start < 1000:
        return None

    source = source_audio or video_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{ref_start / 1000:.3f}",
            "-t",
            f"{(ref_end - ref_start) / 1000:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RenderError(f"克隆参考音提取失败：{completed.stderr[-300:]}")
    return output_path


def extract_stable_voice_reference(
    source_audio: Path,
    output_path: Path,
    *,
    target_ms: int | None = None,
) -> Path:
    """Create a reusable, level-normalized reference from contiguous vocals."""
    if FFMPEG_PATH is None:
        raise RenderError("未找到 FFmpeg，无法生成稳定参考音")
    if not source_audio.is_file():
        raise RenderError("克隆参考音源不存在")
    target_ms = max(VOICE_REFERENCE_MIN_MS, target_ms or VOICE_REFERENCE_TARGET_MS)
    duration_ms = probe_duration_ms(source_audio)
    available_ms = min(target_ms, duration_ms)
    if available_ms < VOICE_REFERENCE_MIN_MS:
        raise RenderError("音色参考音频太短，无法稳定复刻")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            f"{available_ms / 1000:.3f}",
            "-i",
            str(source_audio),
            "-af",
            "silenceremove=start_periods=1:start_duration=0.1:"
            "start_threshold=-45dB:detection=peak,"
            "loudnorm=I=-18:TP=-2:LRA=7",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RenderError(f"稳定参考音生成失败：{completed.stderr[-300:]}")
    return output_path


def _load_video_path(material_id: str) -> str:
    from .db import get_db

    with get_db() as conn:
        row = conn.execute("SELECT video_path FROM material WHERE id = ?", (material_id,)).fetchone()
    if row is None or not row["video_path"]:
        raise RenderError("视频文件不存在")
    return row["video_path"]


def _load_segments(material_id: str) -> list[Segment]:
    from .db import get_db

    with get_db() as conn:
        row = conn.execute("SELECT srt_path FROM material WHERE id = ?", (material_id,)).fetchone()
    if row is None or not row["srt_path"] or not Path(row["srt_path"]).is_file():
        return []
    return parse_srt(read_text(Path(row["srt_path"])))


def _load_plan_shots(material_id: str, material: dict[str, Any]) -> list[Any]:
    duration_ms = int(material.get("duration_ms") or 0)
    if duration_ms <= 0:
        return []
    try:
        from .shots import cached_manifest

        return cached_manifest(
            Path(_load_video_path(material_id)),
            _load_segments(material_id),
            duration_ms,
            DATA_DIR / "shots" / f"{material_id}.json",
        )
    except Exception:  # noqa: BLE001
        return []


MIN_PIECE_MS = 120
WAN_TEXT_DENSITY_MIN = 4.0
WAN_TEXT_DENSITY_MAX = 6.5


def _filler_text(run: list[Any], start_ms: int, end_ms: int) -> str:
    """合并进来的合规镜头保留原话，用来填满 Wan 的时长空档。"""
    parts = [
        (shot.text or "").strip()
        for shot in run
        if (shot.end_ms <= start_ms or shot.start_ms >= end_ms) and (shot.text or "").strip()
    ]
    return "，".join(parts)


def _merged_short_span(
    shots: list[Any],
    start: int,
    end: int,
    *,
    boundary_ok: Callable[[Any, Any], bool],
    min_seconds: float,
    max_seconds: float,
) -> tuple[int, int, list[Any], int] | None:
    """把含短镜的区间向相邻镜头补齐到 Wan 最短时长。

    只并入有口播、且切点两侧画面连续的镜头。跨真实换景的切点不合并，
    否则 Wan 会把两个场景糊成一个镜头。
    """
    indexes = [
        index
        for index, shot in enumerate(shots)
        if shot.end_ms > start and shot.start_ms < end
    ]
    if not indexes:
        return None
    first, last = indexes[0], indexes[-1]
    span = shots[last].end_ms - shots[first].start_ms
    if span >= min_seconds * 1000:
        return None
    left, right = first - 1, last + 1
    while span < min_seconds * 1000:
        grown = False
        if right < len(shots):
            candidate = shots[right]
            candidate_span = candidate.end_ms - shots[first].start_ms
            if (
                candidate.chars > 0
                and candidate_span <= max_seconds * 1000
                and boundary_ok(shots[right - 1], candidate)
            ):
                last = right
                right += 1
                span = candidate_span
                grown = True
        if span < min_seconds * 1000 and left >= 0:
            candidate = shots[left]
            candidate_span = shots[last].end_ms - candidate.start_ms
            if (
                candidate.chars > 0
                and candidate_span <= max_seconds * 1000
                and boundary_ok(candidate, shots[left + 1])
            ):
                first = left
                left -= 1
                span = candidate_span
                grown = True
        if not grown:
            return None
    if span > max_seconds * 1000:
        return None
    return shots[first].start_ms, shots[last].end_ms, shots[first:last + 1], first


def _cross_shot_route(
    pieces: list[tuple[int, Any, int]],
) -> tuple[str, str]:
    """跨镜区间先判路由：TTS 只动音轨，不需要对齐切点；只有 Wan 才必须按镜头拆。"""
    from .shots import WAN_MIN_SECONDS

    short = [shot for _, shot, _ in pieces if shot.seconds < WAN_MIN_SECONDS]
    if short:
        return (
            "tts",
            f"跨 {len(pieces)} 个镜头且含 {len(short)} 个短镜头，整段只换音轨，画面不动",
        )
    weak = [
        shot
        for _, shot, _ in pieces
        if shot.kind in {"voiceover", "unknown"}
    ]
    if weak:
        return (
            "tts",
            f"跨 {len(pieces)} 个镜头且有无脸或待判定镜头，整段转人工",
        )
    return "wan", f"跨 {len(pieces)} 个口播镜头，按镜头拆开逐个重生成"


def _fit_op_to_shot(
    op: dict[str, Any],
    shot: Any,
    shot_index: int,
    *,
    piece_start: int | None = None,
    piece_end: int | None = None,
) -> dict[str, Any] | None:
    """Attach a route to one replacement unit.

    只有 Wan 需要对齐硬切并吸附到整镜；TTS 只改音轨，跨切点没关系，
    所以保留原区间，避免把文案裁到装不下的长度。
    """
    from .shots import route_for_span

    start = int(op["start_ms"]) if piece_start is None else int(piece_start)
    end = int(op["end_ms"]) if piece_end is None else int(piece_end)
    overlap_start = max(start, shot.start_ms)
    overlap_end = min(end, shot.end_ms)
    if overlap_end <= overlap_start:
        return None
    span = max(1, shot.end_ms - shot.start_ms)
    coverage = min(1.0, (overlap_end - overlap_start) / span)
    decision = route_for_span(shot, overlap_start, overlap_end)
    if decision.route == "wan":
        start, end = shot.start_ms, shot.end_ms
        coverage = 1.0
    return {
        **op,
        "start_ms": start,
        "end_ms": end,
        "shot_index": shot_index,
        "shot_seconds": round(shot.seconds, 2),
        "coverage": round(coverage, 3),
        "route": decision.route,
        "route_reason": decision.reason,
    }


def _align_replace_ops_to_shots(
    material_id: str,
    material: dict[str, Any],
    ops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split cross-shot replacements per shot and attach a Wan/TTS route.

    Wan 只能整镜重生成，一个替换单元横跨多个镜头时不能整段提交：要么文案被
    摊到错误的镜头，要么一次违规变成 2–4 次调用。拆不出每段文案的，直接转人工。
    """
    if not ops:
        return [], [], []
    shots = _load_plan_shots(material_id, material)
    if not shots:
        return ops, [], []

    aligned: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, op in enumerate(ops, start=1):
        start = int(op["start_ms"])
        end = int(op["end_ms"])
        touched = [
            (shot_index, shot, max(0, min(end, shot.end_ms) - max(start, shot.start_ms)))
            for shot_index, shot in enumerate(shots)
            if shot.end_ms > start and shot.start_ms < end
        ]
        if not touched:
            aligned.append(op)
            continue
        dominant = max(touched, key=lambda item: item[2])
        # 只擦到镜头边缘几十毫秒的相邻镜头不该单独改写，否则会切出无意义碎段。
        significant = [item for item in touched if item[2] >= MIN_PIECE_MS]

        from .shots import WAN_MERGE_MAX_SECONDS, WAN_MIN_SECONDS, can_merge_across

        needs_padding = (end - start) < WAN_MIN_SECONDS * 1000 or any(
            shot.seconds < WAN_MIN_SECONDS for _, shot, _ in significant
        )
        if needs_padding:
            try:
                video_path: Path | None = Path(_load_video_path(material_id))
            except Exception:  # noqa: BLE001
                video_path = None
            merged = (
                _merged_short_span(
                    shots,
                    start,
                    end,
                    boundary_ok=lambda _prev, nxt: can_merge_across(
                        video_path, nxt.start_ms  # type: ignore[arg-type]
                    ),
                    min_seconds=WAN_MIN_SECONDS,
                    max_seconds=WAN_MERGE_MAX_SECONDS,
                )
                if video_path is not None
                else None
            )
            if merged is not None:
                merged_start, merged_end, run, run_index = merged
                seconds = (merged_end - merged_start) / 1000
                text = str(op.get("text") or "").strip()
                density = len(text) / seconds if seconds > 0 else 0.0
                if density < WAN_TEXT_DENSITY_MIN:
                    filler = _filler_text(run, start, end)
                    if filler:
                        text = f"{text.rstrip('，。！？；：,.!?;:')}，{filler}"
                        density = len(text) / seconds
                if density <= WAN_TEXT_DENSITY_MAX:
                    aligned.append(
                        {
                            **op,
                            "text": text,
                            "start_ms": merged_start,
                            "end_ms": merged_end,
                            "shot_index": run_index,
                            "shot_seconds": round(seconds, 2),
                            "coverage": 1.0,
                            "route": "wan",
                            "route_reason": (
                                f"合并 {len(run)} 个相邻镜头补齐到 {seconds:.1f}s，"
                                f"原区间 {(end - start) / 1000:.1f}s"
                            ),
                            "merged_shots": len(run),
                        }
                    )
                    warnings.append(
                        f"替换片段 {index} 短于 Wan 最短时长，已合并 {len(run)} 个相邻镜头到 "
                        f"{seconds:.1f}s；原片内部切点会被生成的一镜替代"
                    )
                    continue
                warnings.append(
                    f"替换片段 {index} 合并到 {seconds:.1f}s 后文案密度 {density:.1f} 字/秒，"
                    "仍超出上限，改走局部换音轨"
                )

        if len(significant) <= 1:
            fitted = _fit_op_to_shot(op, dominant[1], dominant[0])
            if fitted is not None:
                aligned.append(fitted)
            continue

        cross_route, cross_reason = _cross_shot_route(significant)
        if cross_route != "wan":
            span = max(1, end - start)
            coverage = sum(item[2] for item in significant) / span
            aligned.append(
                {
                    **op,
                    "start_ms": start,
                    "end_ms": end,
                    "shot_index": dominant[0],
                    "shot_seconds": round(dominant[1].seconds, 2),
                    "coverage": round(min(1.0, coverage), 3),
                    "route": "tts",
                    "route_reason": cross_reason,
                    "cross_shots": len(significant),
                }
            )
            warnings.append(f"替换片段 {index} {cross_reason}")
            continue

        chunks = _split_text_by_duration(
            str(op.get("text") or ""),
            [item[2] for item in significant],
            chars_per_second=5.5,
        )
        if any(len(chunk.strip()) < 2 for chunk in chunks):
            manual.append(
                {
                    **op,
                    "reason": (
                        f"违规点跨 {len(significant)} 个镜头，文案拆不到每段，"
                        "需要人工按镜头改写"
                    ),
                }
            )
            warnings.append(
                f"替换片段 {index} 跨 {len(significant)} 个镜头且文案拆不开，已转人工"
            )
            continue
        pieces = []
        for (shot_index, shot, _), chunk in zip(significant, chunks):
            fitted = _fit_op_to_shot(
                {**op, "text": chunk.strip()},
                shot,
                shot_index,
                piece_start=max(start, shot.start_ms),
                piece_end=min(end, shot.end_ms),
            )
            if fitted is not None:
                pieces.append(fitted)
        warnings.append(
            f"替换片段 {index} 跨 {len(significant)} 个镜头，已按镜头拆成 {len(pieces)} 段"
        )
        aligned.extend(pieces)
    return aligned, manual, warnings


def _snap_replace_ops_to_shots(
    material_id: str,
    material: dict[str, Any],
    ops: list[dict[str, Any]],
    *,
    max_trim_ms: int = 160,
) -> list[dict[str, Any]]:
    """Expand replacements that sit inside one shot to the full shot boundaries."""
    if not ops:
        return []
    duration_ms = int(material.get("duration_ms") or 0)
    if duration_ms <= 0:
        return ops
    try:
        from .shots import cached_manifest

        video_path = Path(_load_video_path(material_id))
        segments = _load_segments(material_id)
        shots = cached_manifest(
            video_path,
            segments,
            duration_ms,
            DATA_DIR / "shots" / f"{material_id}.json",
        )
    except Exception:  # noqa: BLE001
        return ops

    assignments: dict[int, list[int]] = {}
    for index, op in enumerate(ops):
        start = int(op["start_ms"])
        end = int(op["end_ms"])
        candidates = [
            (shot_index, shot)
            for shot_index, shot in enumerate(shots)
            if shot.end_ms > start and shot.start_ms < end
        ]
        if not candidates:
            continue
        shot_index, shot = max(
            candidates,
            key=lambda item: max(
                0,
                min(end, item[1].end_ms) - max(start, item[1].start_ms),
            ),
        )
        outside_ms = max(0, shot.start_ms - start) + max(0, end - shot.end_ms)
        if outside_ms > max_trim_ms:
            continue
        assignments.setdefault(shot_index, []).append(index)

    snap_to_shot: dict[int, Any] = {}
    for shot_index, op_indexes in assignments.items():
        shot = shots[shot_index]
        pieces = sorted(
            (
                max(int(ops[index]["start_ms"]), shot.start_ms),
                min(int(ops[index]["end_ms"]), shot.end_ms),
            )
            for index in op_indexes
        )
        covered = 0
        cursor = shot.start_ms
        for piece_start, piece_end in pieces:
            covered += max(0, piece_end - max(piece_start, cursor))
            cursor = max(cursor, piece_end)
        duration = max(1, shot.end_ms - shot.start_ms)
        if covered / duration >= 0.45:
            for index in op_indexes:
                snap_to_shot[index] = shot

    snapped: list[dict[str, Any]] = []
    for index, op in enumerate(ops):
        shot = snap_to_shot.get(index)
        if shot is None:
            snapped.append(op)
            continue
        snapped.append(
            {
                **op,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
            }
        )
    return snapped


def _merge_replace_ops(
    ops: list[dict[str, Any]],
    *,
    max_gap_ms: int = 250,
) -> list[dict[str, Any]]:
    if not ops:
        return []
    ordered = sorted(ops, key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    merged: list[dict[str, Any]] = [dict(ordered[0])]
    for op in ordered[1:]:
        current = merged[-1]
        current_shot = current.get("shot_index")
        op_shot = op.get("shot_index")
        if current_shot is not None and op_shot is not None and current_shot != op_shot:
            merged.append(dict(op))
            continue
        if int(op["start_ms"]) > int(current["end_ms"]) + max_gap_ms:
            merged.append(dict(op))
            continue
        current["end_ms"] = max(int(current["end_ms"]), int(op["end_ms"]))
        current["text"] = _merge_spoken_text(
            str(current.get("text") or ""),
            str(op.get("text") or ""),
        )
        current["finding_ids"] = list(
            dict.fromkeys(
                [
                    *(current.get("finding_ids") or []),
                    *(op.get("finding_ids") or []),
                ]
            )
        )
        current["estimated_duration_ms"] = max(
            int(current.get("estimated_duration_ms") or 0),
            int(op.get("estimated_duration_ms") or 0),
        )
    return merged


def _merge_spoken_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left.rstrip('，。！？；：,.!?;:')}，{right.lstrip('，。！？；：,.!?;:')}"


def _expand_replace_op_by_evidence(
    op: dict[str, Any],
    finding: dict[str, Any],
    *,
    max_gap_ms: int = 1500,
    chars_per_second: float = 5.5,
) -> list[dict[str, Any]]:
    """Avoid replacing a long span when evidence only covers a few short clusters."""
    evidence = finding.get("evidence") or []
    intervals: list[tuple[int, int]] = []
    for item in evidence:
        try:
            start = int(item.get("start_ms"))
            end = int(item.get("end_ms"))
        except (TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return [op]
    intervals.sort()
    clusters: list[tuple[int, int]] = []
    for start, end in intervals:
        if clusters and start <= clusters[-1][1] + max_gap_ms:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], end))
        else:
            clusters.append((start, end))
    if len(clusters) < 2:
        return [op]
    span = int(op["end_ms"]) - int(op["start_ms"])
    covered = sum(end - start for start, end in clusters)
    if span <= 0 or covered >= span * 0.85:
        return [op]
    chunks = _split_text_by_duration(
        str(op["text"]),
        [end - start for start, end in clusters],
        chars_per_second=chars_per_second,
    )
    expanded: list[dict[str, Any]] = []
    for (start, end), text in zip(clusters, chunks):
        if not text.strip():
            continue
        expanded.append(
            {
                **op,
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "estimated_duration_ms": max(
                    800,
                    int(len(text) * 1000 / chars_per_second),
                ),
            }
        )
    return expanded or [op]


def _split_text_by_duration(
    text: str,
    durations: list[int],
    *,
    chars_per_second: float,
) -> list[str]:
    clean = text.strip()
    if not clean or not durations:
        return [clean]
    total_duration = sum(max(1, value) for value in durations)
    punctuation = set("，。！？；：,.!?;:")
    split_points = [
        index + 1
        for index, char in enumerate(clean)
        if char in punctuation and index + 1 < len(clean)
    ]
    chunks: list[str] = []
    cursor = 0
    cumulative = 0
    for index, duration in enumerate(durations):
        cumulative += max(1, duration)
        if index == len(durations) - 1:
            chunk = clean[cursor:]
        else:
            target = round(len(clean) * cumulative / total_duration)
            budget = max(1, int(duration / 1000 * chars_per_second))
            max_point = min(len(clean) - 1, cursor + budget)
            candidates = [
                point
                for point in split_points
                if cursor < point <= max_point
            ]
            point = (
                min(candidates, key=lambda item: abs(item - target))
                if candidates
                else max(cursor + 1, min(target, max_point))
            )
            chunk = clean[cursor:point]
            cursor = point
        chunk = chunk.lstrip("，。！？；：,.!?;:").strip()
        chunks.append(chunk)
    return chunks[: len(durations)]


def _split_cut_ops_by_evidence(
    ops: list[dict[str, Any]],
    *,
    max_span_ms: int = MAX_CUT_UNIT_MS,
    max_gap_ms: int = 1500,
) -> list[dict[str, Any]]:
    """把一个跨多句的删段按证据时间簇拆开。

    语义事件可以把多句同类违规收进同一条证据链，但落到剪辑动作上必须逐簇执行，
    否则一条 finding 会变成"删掉半分钟"。
    """
    split: list[dict[str, Any]] = []
    for op in ops:
        evidence: list[tuple[int, int]] = []
        for item in op.get("evidence") or []:
            try:
                start = int(item["start_ms"])
                end = int(item["end_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                evidence.append((start, end))
        if len(evidence) < 2:
            split.append(op)
            continue
        evidence.sort()
        clusters: list[tuple[int, int]] = []
        for start, end in evidence:
            if (
                clusters
                and start <= clusters[-1][1] + max_gap_ms
                and max(end, clusters[-1][1]) - clusters[-1][0] <= max_span_ms
            ):
                clusters[-1] = (clusters[-1][0], max(clusters[-1][1], end))
            else:
                clusters.append((start, end))
        if len(clusters) <= 1:
            split.append(op)
            continue
        for index, (start, end) in enumerate(clusters, start=1):
            split.append(
                {
                    **op,
                    "start_ms": start,
                    "end_ms": end,
                    "instruction": (
                        f"{op.get('instruction') or '删除该片段'}"
                        f"（按证据簇 {index}/{len(clusters)}）"
                    ),
                }
            )
    return split


def _cap_oversize_cuts(
    ops: list[dict[str, Any]],
    duration_ms: int,
    *,
    max_ratio: float = MAX_AUTO_CUT_RATIO,
    max_span_ms: int = MAX_MERGED_CUT_MS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """单个自动删段不得过长，否则转人工，避免一条规则删掉半条片子。"""
    # 短视频里删掉 3 秒是正常的，比例上限至少给到 4 秒，只约束长片里的整段删除。
    ratio_limit = max(4000, int(duration_ms * max_ratio)) if duration_ms > 0 else max_span_ms
    limit = min(max_span_ms, ratio_limit)
    kept: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for op in ops:
        span = int(op["end_ms"]) - int(op["start_ms"])
        if span > limit:
            manual.append(
                {
                    **op,
                    "reason": (
                        f"连续删段 {span / 1000:.1f}s 超过自动上限 {limit / 1000:.1f}s，"
                        "需要人工确认删哪几段"
                    ),
                }
            )
        else:
            kept.append(op)
    return kept, manual


def _merge_cut_ops(
    ops: list[dict[str, Any]],
    *,
    max_span_ms: int = MAX_MERGED_CUT_MS,
    max_gap_ms: int = 200,
) -> list[dict[str, Any]]:
    if not ops:
        return []
    ordered = sorted(ops, key=lambda item: (item["start_ms"], item["end_ms"]))
    merged: list[dict[str, Any]] = [dict(ordered[0])]
    for op in ordered[1:]:
        current = merged[-1]
        combined_span = max(int(current["end_ms"]), int(op["end_ms"])) - int(current["start_ms"])
        if op["start_ms"] <= current["end_ms"] + max_gap_ms and combined_span <= max_span_ms:
            current["end_ms"] = max(current["end_ms"], op["end_ms"])
            current["finding_ids"].extend(op["finding_ids"])
            current["instruction"] = "；".join(
                dict.fromkeys([current.get("instruction", ""), op.get("instruction", "")])
            )
        else:
            merged.append(dict(op))
    return merged


HIGH_VISUAL_CONF = 0.85
LOW_SPEECH_CONF = 0.95


def _subtract_spans(
    start: int,
    end: int,
    spans: list[tuple[int, int]],
    *,
    min_piece_ms: int = 300,
) -> list[tuple[int, int]]:
    cursor = start
    pieces: list[tuple[int, int]] = []
    for span_start, span_end in sorted(spans):
        if span_end <= cursor or span_start >= end:
            continue
        if span_start > cursor:
            pieces.append((cursor, min(span_start, end)))
        cursor = max(cursor, span_end)
        if cursor >= end:
            break
    if cursor < end:
        pieces.append((cursor, end))
    return [(a, b) for a, b in pieces if b - a >= min_piece_ms]


def _resolve_cut_conflicts(
    replace_ops: list[dict[str, Any]],
    cut_ops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """视觉删段与口播改写重叠时的裁决。

    只有"高置信视觉 + 不够可信的口播改写"才让删段吞掉改写；其余组合保留改写，
    把删段缩到不重叠的部分。画面违规改写修不了，所以视觉置信度高时仍以删段为准。
    """
    if not replace_ops or not cut_ops:
        return replace_ops, cut_ops, []
    kept: list[dict[str, Any]] = []
    cut_priority = 0
    rewrite_priority = 0
    for cut in cut_ops:
        cut_start, cut_end = int(cut["start_ms"]), int(cut["end_ms"])
        overlapping = [
            op
            for op in replace_ops
            if int(op["end_ms"]) > cut_start and int(op["start_ms"]) < cut_end
        ]
        if not overlapping:
            kept.append(cut)
            continue
        visual_conf = float(cut.get("confidence") or 0)
        speech_conf = max(float(op.get("confidence") or 0) for op in overlapping)
        if visual_conf >= HIGH_VISUAL_CONF and speech_conf < LOW_SPEECH_CONF:
            kept.append(cut)
            cut_priority += 1
            continue
        pieces = _subtract_spans(
            cut_start,
            cut_end,
            [(int(op["start_ms"]), int(op["end_ms"])) for op in overlapping],
        )
        if len(pieces) != 1 or pieces[0] != (cut_start, cut_end):
            rewrite_priority += 1
        for piece_start, piece_end in pieces:
            kept.append({**cut, "start_ms": piece_start, "end_ms": piece_end})
    warnings: list[str] = []
    if cut_priority or rewrite_priority:
        warnings.append(
            f"视觉删段与口播改写重叠：{cut_priority} 段删段优先、"
            f"{rewrite_priority} 段改写优先"
        )
    return replace_ops, kept, warnings


def _absorb_short_replace_remnants(
    replace_ops: list[dict[str, Any]],
    cut_ops: list[dict[str, Any]],
    duration_ms: int,
    *,
    chars_per_second: float = 5.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Delete a whole replacement window when a cut leaves too little room for its line."""
    if not replace_ops or not cut_ops:
        return replace_ops, cut_ops, []
    active_cuts = [dict(op) for op in cut_ops]
    active_replacements: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, op in enumerate(replace_ops, start=1):
        cuts = [(int(item["start_ms"]), int(item["end_ms"])) for item in active_cuts]
        kept = _kept_intervals(duration_ms, cuts)
        start = int(op["start_ms"])
        end = int(op["end_ms"])
        remaining_ms = sum(
            max(0, min(end, kept_end) - max(start, kept_start))
            for kept_start, kept_end in kept
        )
        if remaining_ms <= 0:
            active_replacements.append(op)
            continue
        required_ms = max(800, int(len(str(op.get("text") or "")) * 1000 / chars_per_second))
        if remaining_ms >= required_ms:
            active_replacements.append(op)
            continue

        overlapping = [
            item
            for item in active_cuts
            if int(item["end_ms"]) > start and int(item["start_ms"]) < end
        ]
        if not overlapping:
            active_replacements.append(op)
            continue

        cut_start = min([start, *(int(item["start_ms"]) for item in overlapping)])
        cut_end = max([end, *(int(item["end_ms"]) for item in overlapping)])
        merged_cut = {
            "start_ms": cut_start,
            "end_ms": cut_end,
            "suggestion_id": next(
                (str(item.get("suggestion_id")) for item in overlapping if item.get("suggestion_id")),
                "",
            ),
            "finding_ids": list(
                dict.fromkeys(
                    finding_id
                    for item in overlapping
                    for finding_id in (item.get("finding_ids") or [])
                )
            ),
            "instruction": "；".join(
                dict.fromkeys(
                    [
                        *(str(item.get("instruction") or "") for item in overlapping),
                        "删段扩展以覆盖无法容纳替换文案的残留区间",
                    ]
                )
            ),
        }
        active_cuts = [
            item
            for item in active_cuts
            if not (int(item["end_ms"]) > cut_start and int(item["start_ms"]) < cut_end)
        ]
        active_cuts.append(merged_cut)
        active_cuts.sort(key=lambda item: int(item["start_ms"]))
        warnings.append(
            f"替换片段 {index} 删段后只剩 {remaining_ms / 1000:.2f}s，"
            f"不足以容纳 {required_ms / 1000:.2f}s 文案，已并入删段"
        )

    return active_replacements, active_cuts, warnings


def _kept_intervals(duration_ms: int, cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(cuts):
        start = max(0, min(duration_ms, start))
        end = max(0, min(duration_ms, end))
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_ms:
        intervals.append((cursor, duration_ms))
    return [(start, end) for start, end in intervals if end > start]


def _map_replace_ops_after_cuts(
    ops: list[dict[str, Any]],
    cuts: list[tuple[int, int]],
    duration_ms: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Map source-time replacements onto the post-cut video timeline."""
    if not ops:
        return [], []
    kept = _kept_intervals(duration_ms, cuts)
    mapped: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, op in enumerate(ops, start=1):
        source_start = int(op["start_ms"])
        source_end = int(op["end_ms"])
        pieces: list[tuple[int, int, int]] = []
        for kept_start, kept_end in kept:
            start = max(source_start, kept_start)
            end = min(source_end, kept_end)
            if end <= start:
                continue
            mapped_start = start - _removed_before(start, cuts)
            mapped_end = end - _removed_before(end, cuts)
            pieces.append((start, mapped_start, mapped_end))
        if not pieces:
            warnings.append(f"替换片段 {index} 完全落在删段内，已跳过")
            continue
        _, mapped_start, mapped_end = max(
            pieces,
            key=lambda item: item[2] - item[1],
        )
        if len(pieces) > 1 or sum(mapped_end - mapped_start for _, mapped_start, mapped_end in pieces) < (
            source_end - source_start
        ):
            warnings.append(f"替换片段 {index} 与删段重叠，已按剩余最长区间缩短")
        if mapped_end - mapped_start < 500:
            warnings.append(f"替换片段 {index} 删段后不足 0.5 秒，已跳过")
            continue
        mapped.append(
            {
                **op,
                "start_ms": mapped_start,
                "end_ms": mapped_end,
            }
        )
    return mapped, warnings


def _cut_video(
    video_path: Path,
    kept: list[tuple[int, int]],
    work_dir: Path,
    progress: ProgressCallback | None,
) -> Path:
    duration = probe_duration_ms(video_path)
    if len(kept) == 1 and kept[0][0] == 0 and kept[0][1] >= duration - 50:
        return video_path
    parts: list[Path] = []
    for index, (start, end) in enumerate(kept):
        part = work_dir / f"part_{index:03d}.mp4"
        completed = subprocess.run(
            [
                str(FFMPEG_PATH),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start / 1000:.3f}",
                "-to",
                f"{end / 1000:.3f}",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(part),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0 or not part.is_file():
            raise RenderError(f"视频剪切失败：{completed.stderr[-300:]}")
        parts.append(part)
        if progress:
            progress(f"剪辑拼接 {index + 1}/{len(kept)}", 25 + int((index + 1) / len(kept) * 35))

    list_file = work_dir / "parts.txt"
    list_file.write_text(
        "\n".join(f"file '{part.name}'" for part in parts),
        encoding="utf-8",
    )
    output = work_dir / "cut.mp4"
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RenderError(f"视频拼接失败：{completed.stderr[-300:]}")
    return output


def _synthesize_tts(text: str, output_path: Path) -> None:
    if not text.strip():
        raise RenderError("没有可配音的文案")
    try:
        import edge_tts
    except ImportError as exc:
        raise RenderError("未安装 edge-tts，无法生成配音") from exc

    async def run() -> None:
        communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
        await communicate.save(str(output_path))

    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        raise RenderError(f"TTS 生成失败：{exc}") from exc
    if not output_path.is_file() or output_path.stat().st_size < 1000:
        raise RenderError("TTS 输出为空")


def _mux_tts(
    video_path: Path,
    tts_path: Path,
    output_path: Path,
    duration_ms: int,
    tts_duration_ms: int,
) -> None:
    factor = 1.0
    if duration_ms > 0 and tts_duration_ms > duration_ms:
        factor = min(1.8, max(1.0, tts_duration_ms / duration_ms))
    audio_filter = f"[1:a]atempo={factor:.3f},apad,atrim=0:{duration_ms / 1000:.3f}[a]"
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(tts_path),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RenderError(f"音视频合成失败：{completed.stderr[-300:]}")


def _build_narration(
    segments: list[Segment],
    replace_ops: list[dict[str, Any]],
) -> str:
    if not segments:
        return "\n".join(op["text"] for op in replace_ops if op.get("text"))
    lines: list[str] = []
    used: set[int] = set()
    for segment in segments:
        op_index = next(
            (
                index
                for index, op in enumerate(replace_ops)
                if segment.end_ms > op["start_ms"]
                and segment.start_ms < op["end_ms"]
            ),
            None,
        )
        if op_index is not None:
            if op_index not in used:
                used.add(op_index)
                lines.append(replace_ops[op_index]["text"])
            # Every segment covered by this replacement is skipped; otherwise
            # the original violating台词 would leak back into the TTS.
            continue
        else:
            lines.append(segment.text)
    for index, op in enumerate(replace_ops):
        if index not in used:
            lines.append(op["text"])
    return "\n".join(line for line in lines if line.strip())


def _write_adjusted_srt(
    segments: list[Segment],
    cuts: list[tuple[int, int]],
    replace_ops: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not segments:
        output_path.write_text("", encoding="utf-8")
        return
    kept_map = _segment_kept_ranges(segments, cuts)
    used_replacements: set[int] = set()
    lines: list[str] = []
    for segment in segments:
        kept_range = kept_map.get(segment.index)
        if kept_range is None:
            continue
        new_start, new_end = kept_range
        text = segment.text
        op_index = next(
            (
                index
                for index, op in enumerate(replace_ops)
                if segment.end_ms > op["start_ms"]
                and segment.start_ms < op["end_ms"]
            ),
            None,
        )
        if op_index is not None:
            if op_index in used_replacements:
                continue
            used_replacements.add(op_index)
            text = replace_ops[op_index]["text"]
            replacement_start = replace_ops[op_index]["start_ms"] - _removed_before(
                replace_ops[op_index]["start_ms"], cuts
            )
            replacement_end = replace_ops[op_index]["end_ms"] - _removed_before(
                replace_ops[op_index]["end_ms"], cuts
            )
            new_start = min(new_start, replacement_start)
            new_end = max(new_end, replacement_end)
        lines.append(
            f"{len(lines) + 1}\n{format_timestamp(new_start)} --> {format_timestamp(max(new_start + 200, new_end))}\n{text}\n"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _removed_before(time_ms: int, cuts: list[tuple[int, int]]) -> int:
    return sum(max(0, min(time_ms, end) - start) for start, end in cuts)


def _segment_kept_ranges(
    segments: list[Segment],
    cuts: list[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for segment in segments:
        midpoint = (segment.start_ms + segment.end_ms) // 2
        if any(start <= midpoint < end for start, end in cuts):
            continue
        removed = sum(
            max(0, min(segment.end_ms, end) - max(segment.start_ms, start))
            for start, end in cuts
        )
        start_shift = sum(
            max(0, min(segment.start_ms, end) - start)
            for start, end in cuts
        )
        ranges[segment.index] = (
            max(0, segment.start_ms - start_shift),
            max(0, segment.end_ms - start_shift - removed),
        )
    return ranges

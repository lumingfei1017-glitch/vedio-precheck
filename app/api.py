from __future__ import annotations

import json
import mimetypes
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import jobs
from .asr import AsrError, probe_duration_ms, transcribe_video
from .config import (
    BATCH_RENDER_ENABLED,
    DASHSCOPE_ASR_MODEL,
    DATA_DIR,
    TTS_ROUTE_ENABLED,
    UPLOAD_DIR,
)
from .db import get_db
from .replace import ReplaceError, plan_local_tts, plan_replace
from .review import (
    confirm_material,
    dry_run_plan,
    get_report,
    get_suggestion,
    now_iso,
    recheck_material,
    review_material_content,
    store_review,
    update_suggestion,
    update_suggestion_rewrite,
)
from .rewrite import RewriteError, rewrite_one
from .schemas import (
    BatchCreate,
    ConfirmRequest,
    DryRunResponse,
    FeedbackCreate,
    RefreshRequest,
    ShotReplaceRequest,
    SuggestionUpdate,
)
from .shots import cached_manifest
from .srt import parse_srt, read_text, render_srt, text_from_segments
from .suggest import refresh_text

router = APIRouter(prefix="/api")


def _material_or_404(material_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM material WHERE id = ?", (material_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return dict(row)


def _batch_or_404(batch_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM batch WHERE id = ?", (batch_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="批次不存在")
    return dict(row)


def _material_id_for_suggestion(suggestion_id: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT f.material_id FROM suggestion s
            JOIN finding f ON s.finding_id = f.id
            WHERE s.id = ?
            """,
            (suggestion_id,),
        ).fetchone()
    return row["material_id"] if row else None


def _run_transcription(material: dict) -> str | None:
    """Run ASR for one material and persist transcript/SRT plus failure state."""
    video_path = material.get("video_path")
    if not video_path or not Path(video_path).is_file():
        message = "视频文件不存在，无法识别"
        with get_db() as conn:
            conn.execute(
                "UPDATE material SET status = 'asr_failed', asr_error = ? WHERE id = ?",
                (message, material["id"]),
            )
        return None

    try:
        result = transcribe_video(Path(video_path), Path(video_path).parent)
    except AsrError as exc:
        with get_db() as conn:
            conn.execute(
                """
                UPDATE material
                SET status = 'asr_failed', asr_error = ?, asr_model = ?
                WHERE id = ?
                """,
                (str(exc), DASHSCOPE_ASR_MODEL, material["id"]),
            )
        return None

    srt_path = Path(video_path).parent / "subtitle.srt"
    srt_path.write_text(render_srt(result.segments), encoding="utf-8")
    transcript = text_from_segments(result.segments)
    duration_ms = max(material.get("duration_ms") or 0, result.duration_ms)
    with get_db() as conn:
        conn.execute(
            """
            UPDATE material
            SET srt_path = ?, transcript = ?, status = 'uploaded', duration_ms = ?,
                transcript_source = 'asr', asr_model = ?, asr_error = NULL,
                asr_request_id = ?, word_count = ?
            WHERE id = ?
            """,
            (
                str(srt_path),
                transcript,
                duration_ms,
                result.model,
                result.request_id,
                len(result.words),
                material["id"],
            ),
        )
    return transcript


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/jobs")
def list_job_status(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return jobs.list_jobs(limit)


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@router.post("/batches")
def create_batch(payload: BatchCreate) -> dict:
    batch_id = uuid.uuid4().hex[:12]
    with get_db() as conn:
        conn.execute(
            "INSERT INTO batch (id, name, status, created_at) VALUES (?, ?, 'draft', ?)",
            (batch_id, payload.name.strip(), now_iso()),
        )
    return {"id": batch_id, "name": payload.name.strip(), "status": "draft"}


@router.get("/batches")
def list_batches() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT b.*, COUNT(m.id) AS material_count
            FROM batch b LEFT JOIN material m ON m.batch_id = b.id
            GROUP BY b.id
            ORDER BY b.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@router.post("/batches/{batch_id}/materials")
def upload_material(
    batch_id: str,
    video: UploadFile = File(...),
    srt: UploadFile | None = File(None),
    script: str = Form(""),
    has_face: str = Form("unknown"),
    category: str = Form("shapewear"),
    background: bool = Form(False),
) -> dict:
    _batch_or_404(batch_id)
    # The product line is shapewear-only; ignore legacy category values.
    category = "shapewear"
    material_id = uuid.uuid4().hex[:12]
    material_dir = UPLOAD_DIR / material_id
    material_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = material_dir / f"video{suffix}"
    with video_path.open("wb") as target:
        shutil.copyfileobj(video.file, target)

    segments = []
    srt_path: Path | None = None
    transcript = script.strip()
    transcript_source = "script" if transcript else None
    if srt is not None and srt.filename:
        srt_path = material_dir / "subtitle.srt"
        with srt_path.open("wb") as target:
            shutil.copyfileobj(srt.file, target)
        srt_text = read_text(srt_path)
        segments = parse_srt(srt_text)
        if not segments:
            raise HTTPException(status_code=400, detail="SRT 解析失败或没有有效字幕")
        transcript = text_from_segments(segments)
        transcript_source = "srt"

    has_face_value = {"true": 1, "false": 0}.get(has_face.strip().lower())
    duration_ms = probe_duration_ms(video_path)
    if segments:
        duration_ms = max(duration_ms, segments[-1].end_ms)
    status = "queued" if background else ("uploaded" if transcript else "transcribing")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO material
            (id, batch_id, filename, video_path, srt_path, transcript, has_face,
             category, status, duration_ms, transcript_source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                material_id,
                batch_id,
                video.filename or f"video{suffix}",
                str(video_path),
                str(srt_path) if srt_path else None,
                transcript,
                has_face_value,
                category,
                status,
                duration_ms,
                transcript_source,
                now_iso(),
            ),
        )

    material = _material_or_404(material_id)
    if not transcript and not background:
        _run_transcription(material)
        material = _material_or_404(material_id)

    if background:
        job = jobs.create_material_job(material_id)
        return JSONResponse(
            status_code=202,
            content={
                "id": material_id,
                "batch_id": batch_id,
                "filename": video.filename,
                "status": "queued",
                "has_srt": srt_path is not None,
                "duration_ms": material["duration_ms"],
                "transcript_source": material["transcript_source"],
                "asr_error": material["asr_error"],
                "job_id": job["id"],
            },
        )

    return {
        "id": material_id,
        "batch_id": batch_id,
        "filename": video.filename,
        "has_srt": srt_path is not None,
        "status": material["status"],
        "duration_ms": material["duration_ms"],
        "transcript_source": material["transcript_source"],
        "asr_error": material["asr_error"],
    }


@router.get("/materials")
def list_materials(batch_id: str | None = Query(default=None)) -> list[dict]:
    with get_db() as conn:
        if batch_id:
            rows = conn.execute(
                """
                SELECT m.*, COUNT(f.id) AS finding_count
                FROM material m LEFT JOIN finding f ON f.material_id = m.id
                WHERE m.batch_id = ?
                GROUP BY m.id
                ORDER BY m.created_at DESC
                """,
                (batch_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT m.*, COUNT(f.id) AS finding_count
                FROM material m LEFT JOIN finding f ON f.material_id = m.id
                GROUP BY m.id
                ORDER BY m.created_at DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


@router.get("/materials/{material_id}")
def material_report(material_id: str) -> dict:
    report = get_report(material_id)
    if report is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return report


@router.post("/materials/{material_id}/review")
def review_material(material_id: str, background: bool = Query(default=False)) -> dict:
    material = _material_or_404(material_id)
    if background:
        job = jobs.create_material_job(material_id, kind="material_review")
        return JSONResponse(
            status_code=202,
            content={"job_id": job["id"], "status": job["status"], "material_id": material_id},
        )
    if material.get("status") == "transcribing":
        raise HTTPException(status_code=409, detail="字幕正在识别，请稍后再试")
    transcript = material.get("transcript") or ""
    if not transcript.strip():
        detail = material.get("asr_error") or "素材没有字幕或脚本，请先识别字幕或上传 SRT"
        raise HTTPException(status_code=400, detail=detail)

    segments = []
    srt_path = material.get("srt_path")
    if srt_path and Path(srt_path).is_file():
        segments = parse_srt(read_text(Path(srt_path)))

    result = review_material_content(material, transcript, segments)
    store_review(material_id, result)
    report = get_report(material_id)
    if report is None:
        raise HTTPException(status_code=500, detail="审核结果保存失败")
    return report


@router.post("/batches/{batch_id}/review")
def review_batch(batch_id: str, background: bool = Query(default=False)) -> dict:
    _batch_or_404(batch_id)
    if background:
        job = jobs.create_batch_job(batch_id)
        return JSONResponse(
            status_code=202,
            content={"job_id": job["id"], "status": job["status"], "batch_id": batch_id},
        )
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM material WHERE batch_id = ?", (batch_id,)).fetchall()

    reviewed = 0
    failed: list[dict] = []
    for row in rows:
        material = dict(row)
        transcript = material.get("transcript") or ""
        if not transcript.strip():
            reason = material.get("asr_error") or "缺少字幕或脚本"
            failed.append({"id": material["id"], "filename": material["filename"], "reason": reason})
            continue
        segments = []
        srt_path = material.get("srt_path")
        if srt_path and Path(srt_path).is_file():
            segments = parse_srt(read_text(Path(srt_path)))
        result = review_material_content(material, transcript, segments)
        store_review(material["id"], result)
        reviewed += 1

    return {
        "batch_id": batch_id,
        "total": len(rows),
        "reviewed": reviewed,
        "failed": failed,
    }


@router.post("/materials/{material_id}/transcribe")
def transcribe_material(material_id: str) -> dict:
    material = _material_or_404(material_id)
    if material.get("status") == "transcribing":
        raise HTTPException(status_code=409, detail="字幕正在识别，请稍后再试")
    video_path = material.get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(status_code=404, detail="视频文件不存在")

    with get_db() as conn:
        conn.execute(
            "UPDATE material SET status = 'transcribing', asr_error = NULL WHERE id = ?",
            (material_id,),
        )
    _run_transcription(_material_or_404(material_id))
    report = get_report(material_id)
    if report is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return report


@router.patch("/suggestions/{suggestion_id}")
def patch_suggestion(suggestion_id: str, payload: SuggestionUpdate) -> dict:
    suggestion = update_suggestion(
        suggestion_id,
        after_text=payload.after_text,
        method=payload.method,
        abc_class=payload.abc_class,
        status=payload.status,
    )
    if suggestion is None:
        raise HTTPException(status_code=404, detail="建议不存在")
    material_id = _material_id_for_suggestion(suggestion_id)
    recheck = recheck_material(material_id) if material_id else None
    return {"suggestion": suggestion, "recheck": recheck}


@router.post("/suggestions/{suggestion_id}/refresh")
def refresh_suggestion(suggestion_id: str, payload: RefreshRequest) -> dict:
    suggestion = get_suggestion(suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="建议不存在")
    material_id = _material_id_for_suggestion(suggestion_id)
    material = _material_or_404(material_id) if material_id else None
    report = get_report(material_id) if material_id else None
    finding = None
    if report:
        finding = next(
            (item for item in report["findings"] if item["id"] == suggestion["finding_id"]),
            None,
        )

    updated = None
    if material and finding:
        segments = []
        srt_path = material.get("srt_path")
        if srt_path and Path(srt_path).is_file():
            segments = parse_srt(read_text(Path(srt_path)))
        instruction = "；".join(
            value for value in (payload.constraint, payload.instruction) if value
        )
        try:
            rewrite = rewrite_one(
                material.get("transcript") or "",
                segments,
                finding,
                suggestion,
                has_face=(
                    bool(material["has_face"]) if material.get("has_face") is not None else None
                ),
                instruction=instruction,
            )
        except RewriteError:
            rewrite = None
        if rewrite is not None:
            updated = update_suggestion_rewrite(suggestion_id, rewrite)

    if updated is None:
        new_text = refresh_text(suggestion["after_text"], payload.constraint, payload.instruction)
        updated = update_suggestion(suggestion_id, after_text=new_text)
    recheck = recheck_material(material_id) if material_id else None
    return {"suggestion": updated, "recheck": recheck}


@router.post("/suggestions/{suggestion_id}/feedback")
def suggestion_feedback(suggestion_id: str, payload: FeedbackCreate) -> dict:
    suggestion = get_suggestion(suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="建议不存在")

    feedback_id = uuid.uuid4().hex
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (id, suggestion_id, action, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            (feedback_id, suggestion_id, payload.action, payload.reason, now_iso()),
        )
        if payload.action == "false_positive":
            conn.execute(
                "UPDATE suggestion SET status = 'false_positive', updated_at = ? WHERE id = ?",
                (now_iso(), suggestion_id),
            )

    material_id = _material_id_for_suggestion(suggestion_id)
    report = get_report(material_id) if material_id else None
    return {
        "feedback_id": feedback_id,
        "suggestion": get_suggestion(suggestion_id),
        "report": report,
    }


@router.post("/materials/{material_id}/confirm")
def confirm(material_id: str, payload: ConfirmRequest | None = None) -> dict:
    report = confirm_material(material_id)
    if report is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return report


@router.post("/materials/{material_id}/apply")
def apply_material(material_id: str, dry_run: bool = Query(default=True)) -> dict:
    plan = dry_run_plan(material_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    if dry_run:
        return plan
    if not BATCH_RENDER_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="批量渲染已停用：当前只支持单镜 Wan 重生成，画面问题请导出人工清单到剪映处理",
        )
    job = jobs.create_render_job(material_id)
    return JSONResponse(
        status_code=202,
        content={"job_id": job["id"], "status": job["status"], "material_id": material_id},
    )


@router.get("/materials/{material_id}/shots")
def material_shots(material_id: str, refresh: bool = Query(default=False)) -> dict:
    material = _material_or_404(material_id)
    srt_path = material.get("srt_path")
    if not srt_path or not Path(srt_path).is_file():
        return {"material_id": material_id, "shots": [], "hint": "先识别字幕，才能按镜头对齐口播"}
    segments = parse_srt(read_text(Path(srt_path)))
    video_path = material.get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    shots = cached_manifest(
        Path(video_path),
        segments,
        material.get("duration_ms") or 0,
        Path(DATA_DIR) / "shots" / f"{material_id}.json",
        refresh=refresh,
    )
    return {
        "material_id": material_id,
        "shots": [shot.as_dict() for shot in shots],
        "wan_count": sum(1 for shot in shots if shot.route == "wan"),
        "tts_count": sum(1 for shot in shots if shot.route == "tts"),
    }


@router.post("/materials/{material_id}/shots/replace")
def replace_material_shot(material_id: str, payload: ShotReplaceRequest) -> JSONResponse:
    _material_or_404(material_id)
    if payload.route == "tts" and not TTS_ROUTE_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="局部换音已停用，请转人工处理或在剪映里完成",
        )
    try:
        planner = plan_local_tts if payload.route == "tts" else plan_replace
        plan = planner(payload.start_ms / 1000, payload.end_ms / 1000, payload.text)
    except ReplaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.dry_run:
        return JSONResponse(
            content={"material_id": material_id, **plan.as_dict()}
        )
    job = jobs.create_shot_job(
        material_id,
        payload.start_ms,
        payload.end_ms,
        payload.text,
        route=payload.route,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job["id"], "status": job["status"], "material_id": material_id},
    )


@router.post("/audits/sync")
def sync_audit_feed(kind: str = Query(default="reject"), days: int | None = Query(default=None)) -> dict:
    from .audit_feed import AuditFeedError, sync_audits

    try:
        return sync_audits(kind=kind, days=days)
    except AuditFeedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/audits/stats")
def audit_feed_stats() -> dict:
    from .audit_feed import recent_patterns, stats

    payload = stats()
    payload["patterns"] = recent_patterns(5)
    return payload


@router.get("/materials/{material_id}/worklist.md")
def material_worklist_markdown(material_id: str) -> PlainTextResponse:
    from .worklist import build_worklist

    worklist = build_worklist(material_id)
    if worklist is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return PlainTextResponse(
        worklist["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="worklist-{material_id}.md"'},
    )


@router.get("/materials/{material_id}/worklist.srt")
def material_worklist_srt(material_id: str) -> PlainTextResponse:
    from .worklist import build_worklist

    worklist = build_worklist(material_id)
    if worklist is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return PlainTextResponse(
        worklist["srt"],
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="worklist-{material_id}.srt"'},
    )


@router.get("/materials/{material_id}/output")
def material_output(material_id: str) -> FileResponse:
    material = _material_or_404(material_id)
    output_path = material.get("render_output_path")
    if not output_path or not Path(output_path).is_file():
        raise HTTPException(status_code=404, detail="修改版视频尚未生成")
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"modified-{material['filename']}",
    )


@router.get("/materials/{material_id}/output.srt")
def material_output_srt(material_id: str) -> FileResponse:
    material = _material_or_404(material_id)
    output_path = material.get("render_output_path")
    if not output_path:
        raise HTTPException(status_code=404, detail="修改版字幕尚未生成")
    srt_path = Path(output_path).with_name("output.srt")
    if not srt_path.is_file():
        raise HTTPException(status_code=404, detail="修改版字幕尚未生成")
    return FileResponse(srt_path, media_type="text/plain; charset=utf-8")


@router.get("/materials/{material_id}/video")
def material_video(material_id: str) -> FileResponse:
    material = _material_or_404(material_id)
    video_path = material.get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    media_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    return FileResponse(video_path, media_type=media_type)


@router.get("/materials/{material_id}/srt")
def material_srt(material_id: str) -> FileResponse:
    material = _material_or_404(material_id)
    srt_path = material.get("srt_path")
    if not srt_path or not Path(srt_path).is_file():
        raise HTTPException(status_code=404, detail="SRT 文件不存在")
    return FileResponse(srt_path, media_type="text/plain; charset=utf-8")


@router.get("/materials/{material_id}/report.json")
def material_report_json(material_id: str) -> JSONResponse:
    report = get_report(material_id)
    if report is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="report-{material_id}.json"'},
    )

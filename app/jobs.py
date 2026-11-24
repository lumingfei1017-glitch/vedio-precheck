from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import DATA_DIR, SHOT_TTS_FALLBACK_ENABLED
from .db import get_db

ProgressCallback = Callable[[str, int], None]

_executor: ThreadPoolExecutor | None = None
_started = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_worker() -> None:
    global _executor, _started
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="precheck-job")
    if _started:
        return
    _started = True
    with get_db() as conn:
        conn.execute(
            """
            UPDATE job
            SET status = 'failed', stage = 'interrupted',
                error = '服务重启，任务已中断，请重新提交',
                updated_at = ?
            WHERE status IN ('queued', 'running')
            """,
            (now_iso(),),
        )


def shutdown_worker() -> None:
    global _executor, _started
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=False)
    _executor = None
    _started = False


def create_material_job(material_id: str, kind: str = "material_pipeline") -> dict[str, Any]:
    start_worker()
    existing = _find_active_job("material_id", material_id)
    if existing is not None:
        return existing

    job_id = _insert_job(kind=kind, material_id=material_id)
    _submit(_run_material_job, job_id, material_id)
    return get_job(job_id)


def create_batch_job(batch_id: str) -> dict[str, Any]:
    start_worker()
    existing = _find_active_job("batch_id", batch_id)
    if existing is not None:
        return existing

    job_id = _insert_job(kind="batch_review", batch_id=batch_id)
    _submit(_run_batch_job, job_id, batch_id)
    return get_job(job_id)


def create_render_job(material_id: str) -> dict[str, Any]:
    start_worker()
    existing = _find_active_job("material_id", material_id)
    if existing is not None and existing["kind"] == "render_material":
        return existing

    job_id = _insert_job(kind="render_material", material_id=material_id)
    _submit(_run_render_job, job_id, material_id)
    return get_job(job_id)


def create_shot_job(
    material_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
    *,
    route: str = "wan",
) -> dict[str, Any]:
    start_worker()
    job_id = _insert_job(kind="shot_replace", material_id=material_id)
    _submit(_run_shot_job, job_id, material_id, start_ms, end_ms, text, route)
    return get_job(job_id)


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM job ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def _run_material_job(job_id: str, material_id: str) -> None:
    _update_job(job_id, status="running", stage="starting", progress=3)
    try:
        result = _process_material(
            material_id,
            lambda stage, progress: _update_job(
                job_id,
                status="running",
                stage=stage,
                progress=progress,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc)[:500],
            progress=100,
        )
        return

    _update_job(
        job_id,
        status="succeeded",
        stage="done",
        progress=100,
        result=result,
        error=None,
    )


def _run_batch_job(job_id: str, batch_id: str) -> None:
    _update_job(job_id, status="running", stage="batch", progress=3)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename FROM material WHERE batch_id = ? ORDER BY created_at",
            (batch_id,),
        ).fetchall()
    if not rows:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error="批次没有素材",
            progress=100,
        )
        return

    reviewed = 0
    failed: list[dict[str, str]] = []
    total = len(rows)
    for index, row in enumerate(rows):
        material_id = row["id"]
        filename = row["filename"]

        def progress(stage: str, local_progress: int, index: int = index) -> None:
            overall = 5 + int((index + local_progress / 100) / total * 90)
            _update_job(
                job_id,
                status="running",
                stage=f"{index + 1}/{total} {stage}",
                progress=min(95, overall),
            )

        try:
            _process_material(material_id, progress, stage_prefix=f"{filename} · ")
            reviewed += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": material_id, "filename": filename, "reason": str(exc)[:300]})

    status = "succeeded" if not failed else ("partial" if reviewed else "failed")
    _update_job(
        job_id,
        status=status,
        stage="done" if not failed else "finished_with_errors",
        progress=100,
        result={
            "batch_id": batch_id,
            "total": total,
            "reviewed": reviewed,
            "failed": failed,
        },
        error=None if not failed else f"{len(failed)} 条失败",
    )


def _run_render_job(job_id: str, material_id: str) -> None:
    from .edit import build_edit_plan, execute_edit_plan

    _update_job(job_id, status="running", stage="build_plan", progress=5)
    try:
        plan = build_edit_plan(material_id)
        if plan is None:
            raise RuntimeError("素材不存在")
        with get_db() as conn:
            conn.execute(
                "UPDATE material SET render_status = 'running', render_error = NULL WHERE id = ?",
                (material_id,),
            )

        def progress(stage: str, value: int) -> None:
            _update_job(
                job_id,
                status="running",
                stage=stage,
                progress=max(5, min(98, value)),
            )

        result = execute_edit_plan(material_id, plan, progress)
    except Exception as exc:  # noqa: BLE001
        with get_db() as conn:
            conn.execute(
                "UPDATE material SET render_status = 'failed', render_error = ? WHERE id = ?",
                (str(exc)[:500], material_id),
            )
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc)[:500],
            progress=100,
        )
        return

    with get_db() as conn:
        conn.execute(
            """
            UPDATE material
            SET render_status = 'succeeded', render_output_path = ?, render_error = NULL
            WHERE id = ?
            """,
            (result["output_path"], material_id),
        )
    _update_job(
        job_id,
        status="succeeded",
        stage="done",
        progress=100,
        result=result,
        error=None,
    )


def _run_shot_job(
    job_id: str,
    material_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
    route: str,
) -> None:
    from .replace import replace_shot, replace_shot_tts

    _update_job(job_id, status="running", stage="prepare", progress=3)
    material = _load_material(material_id)
    if material is None:
        _update_job(job_id, status="failed", stage="failed", error="素材不存在", progress=100)
        return
    video_path = material.get("video_path")
    if not video_path or not Path(video_path).is_file():
        _update_job(job_id, status="failed", stage="failed", error="视频文件不存在", progress=100)
        return

    out_dir = Path(DATA_DIR) / "render" / material_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "shot-output.mp4"

    def progress(stage: str, value: int) -> None:
        _update_job(job_id, status="running", stage=stage, progress=max(3, min(98, value)))

    with get_db() as conn:
        conn.execute(
            "UPDATE material SET render_status = 'running', render_error = NULL WHERE id = ?",
            (material_id,),
        )
    try:
        if route == "tts":
            result = replace_shot_tts(
                Path(video_path),
                start_ms / 1000,
                end_ms / 1000,
                text,
                output,
                work_dir=out_dir / "work",
                progress=progress,
            )
        else:
            try:
                result = replace_shot(
                    Path(video_path),
                    start_ms / 1000,
                    end_ms / 1000,
                    text,
                    output,
                    work_dir=out_dir / "work",
                    progress=progress,
                )
            except Exception as wan_error:  # noqa: BLE001
                if not SHOT_TTS_FALLBACK_ENABLED:
                    raise
                progress("Wan 失败，切换局部 TTS", 24)
                result = replace_shot_tts(
                    Path(video_path),
                    start_ms / 1000,
                    end_ms / 1000,
                    text,
                    output,
                    work_dir=out_dir / "work",
                    progress=progress,
                )
                result["fallback_from"] = "wan"
                result["fallback_reason"] = str(wan_error)[:300]
                result["manual_review_required"] = True
                result["warnings"] = list(
                    dict.fromkeys(
                        [
                            *(result.get("warnings") or []),
                            "Wan 重生成失败后改用局部 TTS，正脸镜头需人工确认口型",
                        ]
                    )
                )
    except Exception as exc:  # noqa: BLE001
        with get_db() as conn:
            conn.execute(
                "UPDATE material SET render_status = 'failed', render_error = ? WHERE id = ?",
                (str(exc)[:500], material_id),
            )
        _update_job(job_id, status="failed", stage="failed", error=str(exc)[:500], progress=100)
        return

    with get_db() as conn:
        conn.execute(
            "UPDATE material SET render_status = 'succeeded', render_output_path = ?, render_error = NULL WHERE id = ?",
            (result["output"], material_id),
        )
    _update_job(job_id, status="succeeded", stage="done", progress=100, result=result, error=None)


def _process_material(
    material_id: str,
    progress: ProgressCallback,
    *,
    stage_prefix: str = "",
) -> dict[str, Any]:
    from .api import _run_transcription
    from .review import review_material_content, store_review
    from .srt import parse_srt, read_text

    progress(f"{stage_prefix}准备", 8)
    material = _load_material(material_id)
    if material is None:
        raise RuntimeError("素材不存在")

    if not (material.get("transcript") or "").strip():
        progress(f"{stage_prefix}识别口播", 18)
        _run_transcription(material)
        material = _load_material(material_id)
        if material is None:
            raise RuntimeError("素材不存在")
        if material.get("status") == "asr_failed" or not (material.get("transcript") or "").strip():
            raise RuntimeError(material.get("asr_error") or "ASR 识别失败")

    segments = []
    srt_path = material.get("srt_path")
    if srt_path and Path(srt_path).is_file():
        segments = parse_srt(read_text(Path(srt_path)))
    transcript = material.get("transcript") or ""

    def review_progress(stage: str, value: int) -> None:
        mapped = {
            "rules": 58,
            "semantic": 72,
            "visual": 84,
            "rewrite": 94,
            "merge": 96,
        }.get(stage, value)
        progress(f"{stage_prefix}{stage}", mapped)

    result = review_material_content(
        material,
        transcript,
        segments,
        progress=review_progress,
    )
    progress(f"{stage_prefix}保存结果", 94)
    store_review(material_id, result)
    return {
        "material_id": material_id,
        "filename": material.get("filename"),
        "findings": result.summary.get("total", 0),
        "semantic_count": result.summary.get("semantic_count", 0),
        "semantic_status": result.summary.get("semantic_status"),
    }


def _load_material(material_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM material WHERE id = ?", (material_id,)).fetchone()
    return dict(row) if row else None


def _insert_job(
    *,
    kind: str,
    material_id: str | None = None,
    batch_id: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex
    timestamp = now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO job
            (id, kind, material_id, batch_id, status, stage, progress, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'queued', 'queued', 0, ?, ?)
            """,
            (job_id, kind, material_id, batch_id, timestamp, timestamp),
        )
    return job_id


def _find_active_job(field: str, value: str) -> dict[str, Any] | None:
    if field not in {"material_id", "batch_id"}:
        raise ValueError(field)
    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM job
            WHERE {field} = ? AND status IN ('queued', 'running')
            ORDER BY created_at DESC LIMIT 1
            """,
            (value,),
        ).fetchone()
    return _row_to_job(row) if row else None


def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    fields = ["updated_at = ?"]
    values: list[Any] = [now_iso()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if progress is not None:
        fields.append("progress = ?")
        values.append(max(0, min(100, progress)))
    if result is not None:
        fields.append("result_json = ?")
        values.append(json.dumps(result, ensure_ascii=False))
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    values.append(job_id)
    with get_db() as conn:
        conn.execute(f"UPDATE job SET {', '.join(fields)} WHERE id = ?", values)


def _row_to_job(row: Any) -> dict[str, Any]:
    result = None
    if row["result_json"]:
        try:
            result = json.loads(row["result_json"])
        except json.JSONDecodeError:
            result = None
    return {
        "id": row["id"],
        "kind": row["kind"],
        "material_id": row["material_id"],
        "batch_id": row["batch_id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "result": result,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _submit(function: Callable[..., None], *args: Any) -> None:
    start_worker()
    if _executor is None:
        raise RuntimeError("后台任务线程池未启动")
    _executor.submit(function, *args)

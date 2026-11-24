from fastapi.testclient import TestClient

import app.api as api_module
from app.asr import AsrError, AsrResult, AsrWord
from app.main import app
from app.srt import Segment


SRT = """1
00:00:00,000 --> 00:00:03,000
姐妹们，上身显瘦十几斤。

2
00:00:03,000 --> 00:00:06,000
正面看是蚂蚁腰。
"""


def test_review_flow():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "测试批次"}).json()
        files = {
            "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
            "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
        }
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files=files,
            data={"has_face": "false", "category": "shapewear"},
        )
        assert upload.status_code == 200, upload.text
        material_id = upload.json()["id"]

        reviewed = client.post(f"/api/materials/{material_id}/review")
        assert reviewed.status_code == 200, reviewed.text
        report = reviewed.json()
        assert report["summary"]["blocking"] >= 2
        assert len(report["findings"]) >= 2
        assert report["recheck"]["status"] in {"passed", "needs_revision"}

        suggestion_id = report["findings"][0]["suggestion"]["id"]
        patched = client.patch(
            f"/api/suggestions/{suggestion_id}",
            json={"after_text": "上身包裹感更明显。", "method": "TTS_SUBTITLE"},
        )
        assert patched.status_code == 200, patched.text

        refreshed = client.post(
            f"/api/suggestions/{suggestion_id}/refresh",
            json={"constraint": "more_conservative"},
        )
        assert refreshed.status_code == 200, refreshed.text

        dry_run = client.post(f"/api/materials/{material_id}/apply?dry_run=true")
        assert dry_run.status_code == 200, dry_run.text
        assert dry_run.json()["dry_run"] is True
        assert dry_run.json()["plan"]

        confirmed = client.post(f"/api/materials/{material_id}/confirm", json={"accepted": True})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["material"]["status"] == "confirmed"


def _fake_asr_result() -> AsrResult:
    segments = [
        Segment(index=1, start_ms=0, end_ms=3000, text="姐妹们，上身显瘦十几斤。"),
        Segment(index=2, start_ms=3000, end_ms=6000, text="正面看是蚂蚁腰。"),
    ]
    return AsrResult(
        text="姐妹们，上身显瘦十几斤。\n正面看是蚂蚁腰。",
        segments=segments,
        words=[
            AsrWord(text="姐妹们", start_ms=0, end_ms=800, punctuation="，"),
            AsrWord(text="上身显瘦十几斤", start_ms=800, end_ms=3000, punctuation="。"),
            AsrWord(text="正面看是蚂蚁腰", start_ms=3000, end_ms=6000, punctuation="。"),
        ],
        duration_ms=6000,
        request_id="fake-request",
        model="qwen-audio-3.0-asr-flash",
    )


def test_upload_without_srt_runs_asr(monkeypatch):
    monkeypatch.setattr(api_module, "transcribe_video", lambda *args, **kwargs: _fake_asr_result())
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "自动识别批次"}).json()
        files = {"video": ("demo.mp4", b"fake video bytes", "video/mp4")}
        response = client.post(
            f"/api/batches/{batch['id']}/materials",
            files=files,
            data={"has_face": "unknown", "category": "shapewear"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "uploaded"
        assert body["transcript_source"] == "asr"

        reviewed = client.post(f"/api/materials/{body['id']}/review")
        assert reviewed.status_code == 200, reviewed.text
        report = reviewed.json()
        assert report["material"]["transcript_source"] == "asr"
        assert len(report["segments"]) == 2
        assert report["segments"][0]["start_ms"] == 0
        assert report["segments"][1]["end_ms"] == 6000
        assert report["findings"]
        assert report["findings"][0]["start_ms"] is not None


def test_upload_asr_failure_keeps_material_for_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise AsrError("未识别到人声", code="ASR_RESPONSE_HAVE_NO_WORDS")

    monkeypatch.setattr(api_module, "transcribe_video", fail)
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "识别失败批次"}).json()
        files = {"video": ("demo.mp4", b"fake video bytes", "video/mp4")}
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files=files,
            data={"has_face": "unknown", "category": "shapewear"},
        )
        assert upload.status_code == 200, upload.text
        body = upload.json()
        assert body["status"] == "asr_failed"
        assert "未识别到人声" in body["asr_error"]

        review = client.post(f"/api/materials/{body['id']}/review")
        assert review.status_code == 400
        assert "未识别到人声" in review.json()["detail"]

        monkeypatch.setattr(api_module, "transcribe_video", lambda *args, **kwargs: _fake_asr_result())
        retried = client.post(f"/api/materials/{body['id']}/transcribe")
        assert retried.status_code == 200, retried.text
        assert retried.json()["material"]["status"] == "uploaded"
        assert retried.json()["material"]["transcript_source"] == "asr"


def test_batch_review_reviews_all_materials():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "批量审核批次"}).json()
        for index in range(2):
            files = {
                "video": (f"demo-{index}.mp4", b"fake video bytes", "video/mp4"),
                "srt": (f"demo-{index}.srt", SRT.encode("utf-8"), "text/plain"),
            }
            response = client.post(
                f"/api/batches/{batch['id']}/materials",
                files=files,
                data={"has_face": "false", "category": "shapewear"},
            )
            assert response.status_code == 200, response.text

        reviewed = client.post(f"/api/batches/{batch['id']}/review")
        assert reviewed.status_code == 200, reviewed.text
        body = reviewed.json()
        assert body["reviewed"] == 2
        assert body["failed"] == []

        materials = client.get(f"/api/materials?batch_id={batch['id']}").json()
        assert len(materials) == 2
        assert all(material["finding_count"] >= 1 for material in materials)


def test_false_positive_feedback_updates_report():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "误报反馈批次"}).json()
        files = {
            "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
            "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
        }
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files=files,
            data={"has_face": "false", "category": "shapewear"},
        )
        assert upload.status_code == 200, upload.text
        report = client.post(f"/api/materials/{upload.json()['id']}/review").json()
        suggestion_id = report["findings"][0]["suggestion"]["id"]
        blocking_before = report["summary"]["blocking"]

        response = client.post(
            f"/api/suggestions/{suggestion_id}/feedback",
            json={"action": "false_positive", "reason": "测试误报"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["suggestion"]["status"] == "false_positive"
        assert body["report"]["summary"]["false_positive"] == 1
        assert body["report"]["summary"]["blocking"] == blocking_before - 1

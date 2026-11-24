from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app


SRT = """1
00:00:00,000 --> 00:00:03,000
姐妹们，上身显瘦十几斤。

2
00:00:03,000 --> 00:00:06,000
正面看是蚂蚁腰。
"""


def _wait_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"succeeded", "partial", "failed"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {job_id}")


def test_background_upload_and_review():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "后台上传批次"}).json()
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files={
                "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
                "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
            },
            data={"has_face": "false", "background": "true"},
        )
        assert upload.status_code == 202, upload.text
        body = upload.json()
        assert body["job_id"]

        job = _wait_job(client, body["job_id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["material_id"] == body["id"]

        report = client.get(f"/api/materials/{body['id']}")
        assert report.status_code == 200, report.text
        assert report.json()["material"]["status"] == "reviewed"
        assert report.json()["findings"]


def test_background_single_review():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "后台单条批次"}).json()
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files={
                "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
                "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
            },
            data={"has_face": "false"},
        )
        material_id = upload.json()["id"]

        response = client.post(f"/api/materials/{material_id}/review?background=true")
        assert response.status_code == 202, response.text
        job = _wait_job(client, response.json()["job_id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["findings"] >= 1


def test_background_batch_review():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "后台批量批次"}).json()
        for index in range(2):
            upload = client.post(
                f"/api/batches/{batch['id']}/materials",
                files={
                    "video": (f"demo-{index}.mp4", b"fake video bytes", "video/mp4"),
                    "srt": (f"demo-{index}.srt", SRT.encode("utf-8"), "text/plain"),
                },
                data={"has_face": "false"},
            )
            assert upload.status_code == 200, upload.text

        response = client.post(f"/api/batches/{batch['id']}/review?background=true")
        assert response.status_code == 202, response.text
        job = _wait_job(client, response.json()["job_id"], timeout=15)
        assert job["status"] == "succeeded", job
        assert job["result"]["reviewed"] == 2


def test_shot_job_falls_back_to_local_tts(monkeypatch):
    import app.replace as replace_module

    def fail_wan(*args, **kwargs):
        raise RuntimeError("wan unavailable")

    def succeed_tts(video, start, end, text, output, **kwargs):
        return {
            "output": str(output),
            "mode": "tts",
            "voice_source": "cloned",
            "warnings": [],
        }

    monkeypatch.setattr(replace_module, "replace_shot", fail_wan)
    monkeypatch.setattr(replace_module, "replace_shot_tts", succeed_tts)

    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "兜底批次"}).json()
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files={
                "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
                "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
            },
            data={"has_face": "true"},
        )
        material_id = upload.json()["id"]
        response = client.post(
            f"/api/materials/{material_id}/shots/replace",
            json={
                "start_ms": 0,
                "end_ms": 3000,
                "text": "这款平时穿也合适。",
                "route": "wan",
                "dry_run": False,
            },
        )
        assert response.status_code == 202, response.text
        job = _wait_job(client, response.json()["job_id"], timeout=15)

    assert job["status"] == "succeeded", job
    assert job["result"]["fallback_from"] == "wan"
    assert job["result"]["manual_review_required"] is True

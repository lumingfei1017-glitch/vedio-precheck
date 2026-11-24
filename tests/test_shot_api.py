from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

SRT = """1
00:00:00,000 --> 00:00:03,000
坚持穿下去，你的身材能塑形定型。

2
00:00:03,000 --> 00:00:06,000
越穿越有型。
"""


def _material(client: TestClient) -> str:
    batch = client.post("/api/batches", json={"name": "镜头替换"}).json()
    upload = client.post(
        f"/api/batches/{batch['id']}/materials",
        files={
            "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
            "srt": ("demo.srt", SRT.encode("utf-8"), "text/plain"),
        },
        data={"has_face": "true"},
    )
    assert upload.status_code == 200, upload.text
    return upload.json()["id"]


def test_shot_replace_dry_run_plan():
    with TestClient(app) as client:
        material_id = _material(client)
        response = client.post(
            f"/api/materials/{material_id}/shots/replace",
            json={"start_ms": 0, "end_ms": 5500, "text": "这款平时穿也合适，喜欢的姐妹可以试试看。", "dry_run": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["route"] == "wan"
        assert body["duration"] == 5
        # 文案按 Wan 实际生成时长（5 秒）× 5 字/秒 配，不按 5.5 秒参考配
        assert body["char_budget"] == 25
        assert body["ref_seconds"] == 5.5
        assert body["warnings"]


def test_shot_replace_rejects_empty_text():
    with TestClient(app) as client:
        material_id = _material(client)
        response = client.post(
            f"/api/materials/{material_id}/shots/replace",
            json={"start_ms": 0, "end_ms": 3000, "text": "", "dry_run": True},
        )
        assert response.status_code == 422


def test_shot_replace_rejects_too_short_reference():
    with TestClient(app) as client:
        material_id = _material(client)
        response = client.post(
            f"/api/materials/{material_id}/shots/replace",
            json={"start_ms": 0, "end_ms": 1500, "text": "这款版型挺利落。", "dry_run": True},
        )
        assert response.status_code == 400
        assert "超过参考片段" in response.json()["detail"]


def test_local_tts_route_is_disabled():
    with TestClient(app) as client:
        material_id = _material(client)
        response = client.post(
            f"/api/materials/{material_id}/shots/replace",
            json={
                "start_ms": 0,
                "end_ms": 1500,
                "text": "这款版型挺利落。",
                "route": "tts",
                "dry_run": True,
            },
        )
        assert response.status_code == 409, response.text
        assert "已停用" in response.json()["detail"]


def test_shots_requires_subtitle():
    with TestClient(app) as client:
        batch = client.post("/api/batches", json={"name": "无字幕"}).json()
        upload = client.post(
            f"/api/batches/{batch['id']}/materials",
            files={"video": ("demo.mp4", b"fake video bytes", "video/mp4")},
        )
        material_id = upload.json()["id"]
        response = client.get(f"/api/materials/{material_id}/shots")
        assert response.status_code == 200
        assert response.json()["shots"] == []

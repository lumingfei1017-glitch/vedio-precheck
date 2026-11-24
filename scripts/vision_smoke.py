from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VIDEO_PRECHECK_DATA_DIR", str(ROOT / "data"))

from app import config  # noqa: E402,F401


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "vision-smoke.jpg"
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_path.read_bytes()).decode()}"
    base_url = os.environ.get(
        "BASE_URL1", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip().rstrip("/")
    model = os.environ.get("VISUAL_MODEL", "qwen3.8-flash").strip()
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('DASHSCOPE_API_KEY', '').strip()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是服装广告视觉审核员。判断图片是否有低俗擦边、敏感部位特写、"
                                "透视/暴露、未授权品牌logo、极端拉伸演示或穿前穿后身材对比。"
                                "只输出JSON：{\"description\":\"...\",\"categories\":[\"none\"]}。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )
    print(json.dumps({"status": response.status_code, "model": model}, ensure_ascii=True))
    print(response.text[:2000])
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())

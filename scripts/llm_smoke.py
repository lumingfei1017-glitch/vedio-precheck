from __future__ import annotations

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
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base_url = os.environ.get(
        "BASE_URL1", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip().rstrip("/")
    model = os.environ.get("SEMANTIC_MODEL", "deepseek-v4.1-flash-1017").strip()
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": 'Return exactly {"ok": true}.'},
            ],
            "temperature": 0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    print(json.dumps({"status": response.status_code, "model": model}, ensure_ascii=True))
    print(response.text[:1200])
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())

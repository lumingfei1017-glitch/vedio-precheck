"""整镜替换离线脚本：把一段口播镜头换成 Wan 重生成的版本。

用法：
    python scripts/replace_shot.py \
        --video input.mp4 --start 78.2 --end 83.7 \
        --text "这款平时穿也合适，喜欢的姐妹可以试试看。" \
        --out output.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.replace import cut_reference, cut_reference_audio, plan_replace, replace_shot  # noqa: E402
from app.wan import build_shot_prompt  # noqa: E402

DEFAULT_SCENE = "室内竖屏服装展示口播"
DEFAULT_ACTION = "她面向镜头站立介绍身上的产品，手部动作自然，镜头景别保持不变"


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan 整镜替换")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", required=True, type=float, help="镜头起点，秒")
    parser.add_argument("--end", required=True, type=float, help="镜头终点，秒")
    parser.add_argument("--text", required=True, help="替换后的口播台词")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--action", default=None)
    parser.add_argument("--resolution", default="720P", choices=["480P", "720P", "1080P"])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只切参考片段，不调用接口")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        raise SystemExit(f"视频不存在：{video}")
    plan = plan_replace(args.start, args.end, args.text)
    work = Path(args.work_dir) if args.work_dir else ROOT / "data" / "replace" / time.strftime("%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)

    print(f"镜头 {plan.start:.2f}-{plan.end:.2f}s，文案 {plan.chars} 字，预算 {plan.budget}，输出 {plan.duration}s")
    for warning in plan.warnings:
        print(f"警告：{warning}")

    ref = cut_reference(video, plan.start, plan.end, work)
    audio = cut_reference_audio(video, plan.start, plan.end, work)
    print(f"参考片段：{ref.name}，参考音：{audio.name}")

    if args.dry_run:
        prompt = build_shot_prompt(
            plan.text,
            scene=args.scene or DEFAULT_SCENE,
            action=args.action or DEFAULT_ACTION,
        )
        (work / "prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"dry-run：提示词已写入 {work / 'prompt.txt'}")
        return

    def progress(stage: str, value: int) -> None:
        print(f"  [{value:3d}%] {stage}")

    result = replace_shot(
        video, plan.start, plan.end, plan.text, Path(args.out).resolve(),
        work_dir=work, scene=args.scene, action=args.action,
        resolution=args.resolution, progress=progress,
    )
    print(f"完成，用时 {result['elapsed_seconds']}s")
    print(f"输出：{result['output']}")
    (work / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

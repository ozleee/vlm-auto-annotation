#!/usr/bin/env python3
"""反向发现：让视觉模型自由列举图中物体，不受类别清单限制。

用途：检查你的类别清单是否覆盖了实际场景。自动标注是闭集（只认清单里的类），
清单外的物体一律不标——这个脚本就是用来发现"清单外还有什么"。

    ./run.sh discover --images ~/yolo-pritice/datasets/desktop/raw --limit 10

输出：
    终端表格：每种物体出现在几张图里、平均每张几个
    discovered.json：完整结果
    classes_suggested.txt：按出现频率生成的候选类别清单（前 N 个）

注意：模型给的名字不会完全统一（"thermos" / "water bottle" / "flask" 可能混着来），
所以要人工归并。它的价值是告诉你"场景里还有什么"，不是直接产出可用清单。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from auto_annotate import build_headers, call_vlm, encode_image

PROMPT = """请列出这张图里所有你能认出的、有明确边界的物体。

要求：
1. 用英文短名，小写，多个词用下划线连接，例如 cell_phone、glue_stick、paper_box
2. 不要输出坐标，不要输出描述
3. 只列物体，不要列背景（墙、桌面、地板）
4. 同一物体只写一次，用 count 表示图中有几个
5. 最多 20 种，按显著程度排序

严格输出 JSON 数组：
[{{"label": "cell_phone", "count": 1}}]

第一个字符必须是 [，最后一个必须是 ]。"""


def parse(text: str) -> list[dict]:
    import re
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    for cand in (t, t[t.find("["): t.rfind("]") + 1] if "[" in t else ""):
        if not cand:
            continue
        try:
            out = json.loads(cand)
            if isinstance(out, list):
                return [x for x in out if isinstance(x, dict)]
        except json.JSONDecodeError:
            continue
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="自由发现图中的物体类别")
    ap.add_argument("--images", required=True, help="图片目录或通配符")
    ap.add_argument("--limit", type=int, default=10, help="抽多少张（默认 10）")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "gpt-4o"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--out", default="discover_out")
    ap.add_argument("--top", type=int, default=15, help="候选清单里输出前几个")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置 $VLM_API_KEY")

    src = Path(args.images).expanduser()
    if any(ch in str(src) for ch in "*?["):
        import glob
        files = sorted(Path(p) for p in glob.glob(str(src)))
    elif src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    else:
        raise SystemExit(f"找不到：{src}")
    files = files[: args.limit] if args.limit else files
    if not files:
        raise SystemExit("没有图片")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_hit = Counter()      # 出现在几张图
    obj_total = Counter()    # 总实例数
    per_image = []
    print(f"抽样 {len(files)} 张，模型 {args.model}\n")

    for i, f in enumerate(files, 1):
        b64, _ = encode_image(f, args.max_side)
        try:
            reply, _ = call_vlm(args.api_base, args.api_key, args.model, b64, PROMPT)
        except SystemExit as e:
            print(f"[{i}/{len(files)}] {f.name}: 失败 {e}")
            continue
        items = parse(reply)
        labels = []
        for it in items:
            lab = str(it.get("label") or "").strip().lower().replace(" ", "_").replace("-", "_")
            if not lab:
                continue
            try:
                n = max(1, int(it.get("count") or 1))
            except (TypeError, ValueError):
                n = 1
            labels.append(lab)
            obj_total[lab] += n
        for lab in set(labels):
            img_hit[lab] += 1
        per_image.append({"image": f.name, "labels": labels})
        print(f"[{i}/{len(files)}] {f.name}: {len(set(labels))} 种")

    (out / "discovered.json").write_text(json.dumps(
        {"model": args.model, "images": len(per_image),
         "image_hits": dict(img_hit), "instance_total": dict(obj_total),
         "per_image": per_image}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'物体名':28s} {'出现图数':>8s} {'占比':>7s} {'总实例':>7s}")
    print("-" * 56)
    for lab, c in img_hit.most_common():
        print(f"{lab:28s} {c:>8d} {c/len(per_image):>6.0%} {obj_total[lab]:>7d}")

    suggested = [lab for lab, _ in img_hit.most_common(args.top)]
    (out / "classes_suggested.txt").write_text(
        "# 由 discover.py 按出现频率生成的候选类别（需人工归并同义词）\n"
        + "\n".join(suggested) + "\n", encoding="utf-8")
    print(f"\n结果 → {out/'discovered.json'}")
    print(f"候选清单 → {out/'classes_suggested.txt'}（前 {len(suggested)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

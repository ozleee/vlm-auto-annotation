#!/usr/bin/env python3
"""横向评测多个视觉模型做自动标注的效果。

为什么需要工具而不是 shell 循环：
    单纯比"检出几个框"会得出错误结论——漏检少的模型可能框得歪。
    这个脚本同时给出四类信号，并把框画成并排图让人一眼判断。

测什么：
    1. 类别覆盖   —— 清单里有多少类被找到（漏检是主要失败模式）
    2. 实例数量   —— 每类几个，能看出重复框或漏框
    3. 自一致性   —— 同一张图跑两次，框的 IoU 是否稳定（不用标注真值就能测）
    4. 成本与耗时 —— token 用量、延迟

用法：
    export VLM_API_KEY=sk-xxx
    python benchmark_models.py \
        --image desk.jpg \
        --classes-file examples/classes.txt \
        --api-base https://opencode.ai/inference/go/openai/v1 \
        --models qwen4-max gemini-4-argon claude-haiku-5-5 \
        --repeat 2 --out bench/

产物（bench/）：
    side_by_side.jpg    各模型的框并排，直接肉眼比
    benchmark.md        对比表，可直接贴进 issue / PR
    benchmark.json      完整数据
    <model>/*.jpg       每个模型的单张预览
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# 复用主脚本里的实现，避免两处维护
from auto_annotate import (DEFAULT_PROMPT, call_vlm, encode_image,
                           normalize_boxes, parse_json_array)


def load_classes(classes: str | None, classes_file: str | None) -> list[str]:
    if classes_file:
        items = [ln.strip() for ln in Path(classes_file).read_text(encoding="utf-8").splitlines()]
    else:
        items = [c.strip() for c in (classes or "").split(",")]
    out = [c for c in items if c and not c.startswith("#")]
    if not out:
        raise SystemExit("没有类别：用 --classes 或 --classes-file")
    return out


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_iou(run_a: list[dict], run_b: list[dict]) -> float:
    """同一图两次结果的匹配 IoU：同类别的框做贪心配对，取平均。"""
    used = set()
    scores = []
    for a in run_a:
        best, best_j = 0.0, -1
        for j, b in enumerate(run_b):
            if j in used or a["label"] != b["label"]:
                continue
            iou = box_iou(a["bbox"], b["bbox"])
            if iou > best:
                best, best_j = iou, j
        if best_j >= 0:
            used.add(best_j)
            scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def draw_boxes(image: np.ndarray, dets: list[dict], title: str) -> np.ndarray:
    canvas = image.copy()
    colors = [(60, 180, 75), (230, 25, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        c = colors[i % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
        cv2.putText(canvas, d["label"], (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
    # 顶部标题条
    bar = np.zeros((36, canvas.shape[1], 3), np.uint8)
    bar[:] = (32, 32, 32)
    cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main() -> int:
    ap = argparse.ArgumentParser(description="横评多个视觉模型的自动标注效果")
    ap.add_argument("--image", nargs="+", required=True, help="一张或多张图（建议用最有代表性的）")
    ap.add_argument("--classes", help="类别清单，逗号分隔")
    ap.add_argument("--classes-file", help="类别清单文件")
    ap.add_argument("--models", nargs="+", required=True, help="要对比的模型名")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--repeat", type=int, default=2, help="每个模型每张图跑几次，用来算自一致性（默认 2）")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--out", default="bench")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置 $VLM_API_KEY 或传 --api-key")

    classes = load_classes(args.classes, args.classes_file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = [Path(p).expanduser() for p in args.image]

    results = {}          # model -> {image_name: {...}}
    side_rows = {}

    for model in args.models:
        print(f"\n=== {model} ===")
        results[model] = {}
        for img_path in images:
            b64, (w, h) = encode_image(img_path, args.max_side)
            prompt = DEFAULT_PROMPT.format(classes="\n".join(f"- {c}" for c in classes), w=w, h=h)

            runs, latencies, tokens = [], [], []
            for r in range(max(1, args.repeat)):
                t0 = time.time()
                try:
                    reply, usage = call_vlm(args.api_base, args.api_key, model, b64, prompt)
                except SystemExit as e:
                    print(f"  {img_path.name} 第 {r+1} 次失败：{e}")
                    reply, usage = "", {}
                latencies.append(time.time() - t0)
                tokens.append(usage.get("completion_tokens") or 0)
                dets = normalize_boxes(parse_json_array(reply), w, h, classes)
                runs.append(dets)
                print(f"  {img_path.name} 第 {r+1} 次：{len(dets)} 个框，{latencies[-1]:.1f}s")

            first = runs[0]
            counts: dict[str, int] = {}
            for d in first:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            coverage = len(counts) / len(classes)
            consistency = (float(np.mean([match_iou(runs[0], runs[i]) for i in range(1, len(runs))]))
                           if len(runs) > 1 else float("nan"))

            results[model][img_path.name] = {
                "boxes": len(first), "per_class": counts,
                "coverage": round(coverage, 3), "missing": [c for c in classes if c not in counts],
                "consistency_iou": None if np.isnan(consistency) else round(consistency, 3),
                "latency_s": [round(x, 1) for x in latencies],
                "completion_tokens": tokens,
                "detections": first,
            }
            print(f"  → 类别覆盖 {coverage:.0%} ｜ 自一致性 IoU "
                  f"{'n/a' if np.isnan(consistency) else f'{consistency:.2f}'} ｜ 漏检 {len(classes)-len(counts)} 类")

            # 单模型预览
            mdir = out / model
            mdir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(mdir / f"{img_path.stem}_boxes.jpg"),
                        draw_boxes(img, first, f"{model} | {img_path.name}"))

            # 并排图素材
            side_rows.setdefault(img_path.name, []).append(
                (model, draw_boxes(img, first, f"{model}  ({len(first)} boxes, "
                                              f"cov {coverage:.0%})")))

    # ---------- 并排图 ----------
    for img_name, rows in side_rows.items():
        target_w = 900
        scaled = []
        for _, canvas in rows:
            scale = target_w / canvas.shape[1]
            scaled.append(cv2.resize(canvas, (target_w, int(canvas.shape[0] * scale))))
        cv2.imwrite(str(out / f"side_by_side_{Path(img_name).stem}.jpg"), np.vstack(scaled))
    if len(side_rows) == 1:
        only = next(iter(side_rows))
        (out / f"side_by_side_{Path(only).stem}.jpg").rename(out / "side_by_side.jpg")

    # ---------- 报告 ----------
    (out / "benchmark.json").write_text(json.dumps(
        {"classes": classes, "api_base": args.api_base, "repeat": args.repeat, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 视觉模型横评", "", f"类别清单：{', '.join(classes)}", "",
             f"每模型跑 {args.repeat} 次/图（自一致性 = 两次结果的匹配 IoU）", ""]
    for img_name in side_rows:
        lines.append(f"## {img_name}")
        lines.append("")
        lines.append("| 模型 | 检出框 | 类别覆盖 | 漏检类别 | 自一致性 IoU | 平均耗时 | 输出 token |")
        lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
        ranked = sorted(args.models,
                        key=lambda m: (results[m][img_name]["coverage"],
                                       results[m][img_name]["consistency_iou"] or 0),
                        reverse=True)
        for m in ranked:
            r = results[m][img_name]
            cons = "-" if r["consistency_iou"] is None else f"{r['consistency_iou']:.2f}"
            lat = f"{np.mean(r['latency_s']):.0f}s"
            tok = f"{np.mean(r['completion_tokens']):.0f}"
            lines.append(f"| {m} | {r['boxes']} | {r['coverage']:.0%} | "
                         f"{', '.join(r['missing']) or '无'} | {cons} | {lat} | {tok} |")
        lines.append("")
    lines.append("> 类别覆盖高但自一致性低的模型，说明它「猜」的成分大，检出的框可能不稳定。")
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n产物目录：{out.resolve()}")
    print("先看 side_by_side.jpg，再对照 benchmark.md 的覆盖率和自一致性做决定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

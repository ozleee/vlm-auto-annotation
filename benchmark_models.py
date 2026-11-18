#!/usr/bin/env python3
"""横向评测多个视觉模型做自动标注的效果（识别 + 可选分割）。

为什么需要工具而不是 shell 循环：
    只比"检出几个框"会得出错误结论——覆盖率高但框歪的模型会被误选。
    这个脚本同时给四类信号，并把结果画成并排图让人一眼判断。

测什么：
    1. 类别覆盖   —— 清单里有多少类被找到（漏检是主要失败模式）
    2. 实例数量   —— 每类几个，能看出重复框或漏框
    3. 自一致性   —— 同一张图跑两次，框的 IoU 是否稳定（不用标注真值就能测）
    4. 成本与耗时 —— token 用量、延迟
    加 --with-sam 时再测：
    5. 掩码质量   —— 掩码占全图比例、每个物体的多边形段数（碎片化程度）

用法：
    # 只评识别（快，不占算力）
    python benchmark_models.py --image desk.jpg --classes-file examples/classes.txt \\
        --api-base https://opencode.ai/inference/go/openai/v1 \\
        --models qwen4-max claude-haiku-5-5 --repeat 2 --out bench/

    # 识别 + 分割一起评
    python benchmark_models.py --image desk.jpg --classes-file examples/classes.txt \\
        --models qwen3.8-max deepseek-v4.1-flash claude-haiku-5-5 \\
        --with-sam --out bench_seg/

产物（--out 目录）：
    side_by_side.jpg    各模型结果并排，肉眼比
    benchmark.md        对比表，可直接贴进 issue / PR
    benchmark.json      完整数据
    <model>/*.jpg       每个模型的单张预览
    <model>/coco_instances.json   仅 --with-sam 时生成，可直接转 YOLO
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
                           mask_to_polygons, normalize_boxes, parse_json_array)


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
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)) + (max(0.0, bx2 - bx1) * max(0.0, by2 - by1)) - inter
    return inter / ua if ua > 0 else 0.0


def match_iou(run_a: list[dict], run_b: list[dict]) -> float:
    """同一图两次结果的匹配 IoU：同类别的框贪心配对，取平均。"""
    used, scores = set(), []
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


def draw_panel(image: np.ndarray, dets: list[dict], masks: np.ndarray | None, title: str) -> np.ndarray:
    """画一张对比面板：框（细线）+ 掩码（半透明红）+ 顶部标题条。"""
    canvas = image.copy()
    if masks is not None and len(masks):
        for m in masks:
            sel = m > 0.5
            canvas[sel] = (canvas[sel] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
    colors = [(60, 180, 75), (0, 255, 255), (0, 130, 200), (245, 130, 48), (145, 30, 180), (255, 0, 255)]
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        c = colors[i % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 1)
        cv2.putText(canvas, d["label"], (x1, max(y1 - 5, 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
    bar[:] = (32, 32, 32)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main() -> int:
    ap = argparse.ArgumentParser(description="横评多个视觉模型的自动标注效果")
    ap.add_argument("--image", nargs="+", required=True, help="一张或多张图")
    ap.add_argument("--classes", help="类别清单，逗号分隔")
    ap.add_argument("--classes-file", help="类别清单文件")
    ap.add_argument("--models", nargs="+", required=True, help="要对比的模型名")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--repeat", type=int, default=2, help="每模型每图跑几次算自一致性（默认 2）")
    ap.add_argument("--with-sam", action="store_true", help="用 SAM2 出掩码，一并评分割效果")
    ap.add_argument("--seg-model", default="models/sam2.1_t.pt", help="SAM2 权重")
    ap.add_argument("--device", default="cpu", help="SAM2 设备：cpu 或 0")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--out", default="bench")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置 $VLM_API_KEY 或传 --api-key")

    classes = load_classes(args.classes, args.classes_file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = [Path(p).expanduser() for p in args.image]

    sam = None
    if args.with_sam:
        from ultralytics import SAM
        sam = SAM(args.seg_model)
        print(f"SAM2 已加载：{args.seg_model}")

    results: dict[str, dict] = {}
    panels: dict[str, list] = {}          # image_name -> [(model, panel)]

    for model in args.models:
        print(f"\n=== {model} ===")
        results[model] = {}
        mdir = out / model
        mdir.mkdir(parents=True, exist_ok=True)

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
                runs.append(normalize_boxes(parse_json_array(reply), w, h, classes))
                print(f"  {img_path.name} 第 {r+1} 次：{len(runs[-1])} 个框，{latencies[-1]:.1f}s")

            first = runs[0]
            counts: dict[str, int] = {}
            for d in first:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            coverage = len(counts) / len(classes)
            consistency = (float(np.mean([match_iou(runs[0], runs[i]) for i in range(1, len(runs))]))
                           if len(runs) > 1 else float("nan"))

            record = {
                "boxes": len(first), "per_class": counts, "coverage": round(coverage, 3),
                "missing": [c for c in classes if c not in counts],
                "consistency_iou": None if np.isnan(consistency) else round(consistency, 3),
                "latency_s": [round(x, 1) for x in latencies],
                "completion_tokens": tokens, "detections": first,
            }

            # ---------- 可选：SAM2 分割 ----------
            masks = None
            if sam is not None and first:
                pts = np.array([d["bbox"] for d in first], dtype=np.float32)
                sr = sam.predict(img_path, bboxes=pts, device=args.device, verbose=False)[0]
                masks = (sr.masks.data.cpu().numpy() if sr.masks is not None
                         else np.zeros((0, h, w), np.float32))
                if len(masks):
                    union = (masks > 0.5).any(axis=0)
                    area_ratio = float(union.sum()) / (w * h)
                    seg_parts = [len(mask_to_polygons((m > 0.5).astype(np.uint8), 0.0005 * w * h))
                                 for m in masks]
                else:
                    area_ratio, seg_parts = 0.0, []
                record["masks"] = len(masks)
                record["mask_area_ratio"] = round(area_ratio, 3)
                record["segments_per_object"] = seg_parts
                record["fragmented"] = sum(1 for s in seg_parts if s > 1)
                print(f"  → SAM2: {len(masks)} 个掩码 ｜ 占全图 {area_ratio:.1%} ｜ "
                      f"碎片化 {record['fragmented']} 个")

                # 生成可训练的 COCO
                cats = {c: i + 1 for i, c in enumerate(classes)}
                coco = {
                    "images": [{"id": 1, "file_name": img_path.name,
                                "file_path": str(img_path.resolve()), "width": w, "height": h}],
                    "categories": [{"id": i + 1, "name": c, "supercategory": "object"}
                                   for i, c in enumerate(classes)],
                    "annotations": [],
                }
                aid = 1
                for mask, d in zip(masks, first):
                    polys = mask_to_polygons((mask > 0.5).astype(np.uint8), 0.0005 * w * h)
                    if not polys:
                        continue
                    ys, xs = np.where(mask > 0.5)
                    coco["annotations"].append({
                        "id": aid, "image_id": 1, "category_id": cats[d["label"]],
                        "segmentation": polys, "area": float((mask > 0.5).sum()),
                        "bbox": [float(xs.min()), float(ys.min()),
                                 float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                        "iscrowd": 0})
                    aid += 1
                (mdir / f"{img_path.stem}_coco.json").write_text(
                    json.dumps(coco, ensure_ascii=False), encoding="utf-8")

            results[model][img_path.name] = record

            # ---------- 预览 ----------
            img = cv2.imread(str(img_path))
            subtitle = (f"{model}  boxes {len(first)}  cov {coverage:.0%}"
                        + (f"  mask {record.get('mask_area_ratio', 0):.0%}" if masks is not None else ""))
            panel = draw_panel(img, first, masks, subtitle)
            cv2.imwrite(str(mdir / f"{img_path.stem}_panel.jpg"), panel)
            panels.setdefault(img_path.name, []).append((model, panel))

    # ---------- 并排图 ----------
    for img_name, rows in panels.items():
        target_w = 900
        scaled = []
        for _, canvas in rows:
            scale = target_w / canvas.shape[1]
            scaled.append(cv2.resize(canvas, (target_w, int(canvas.shape[0] * scale))))
        name = "side_by_side.jpg" if len(panels) == 1 else f"side_by_side_{Path(img_name).stem}.jpg"
        cv2.imwrite(str(out / name), np.vstack(scaled))

    # ---------- 报告 ----------
    (out / "benchmark.json").write_text(json.dumps(
        {"classes": classes, "api_base": args.api_base, "repeat": args.repeat,
         "with_sam": args.with_sam, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 视觉模型横评（识别" + (" + 分割" if args.with_sam else "") + "）", "",
             f"类别清单：{', '.join(classes)}", "",
             f"每模型跑 {args.repeat} 次/图（自一致性 = 两次结果的匹配 IoU）", ""]
    for img_name in panels:
        lines += [f"## {img_name}", ""]
        head = "| 模型 | 检出框 | 类别覆盖 | 漏检 | 自一致性 | 耗时 |"
        sep = "| --- | ---: | ---: | --- | ---: | ---: |"
        if args.with_sam:
            head += " 掩码占比 | 碎片化 |"
            sep += " ---: | ---: |"
        lines += [head, sep]
        ranked = sorted(args.models,
                        key=lambda m: (results[m][img_name]["coverage"],
                                       results[m][img_name]["consistency_iou"] or 0), reverse=True)
        for m in ranked:
            r = results[m][img_name]
            cons = "-" if r["consistency_iou"] is None else f"{r['consistency_iou']:.2f}"
            row = (f"| {m} | {r['boxes']} | {r['coverage']:.0%} | "
                   f"{', '.join(r['missing']) or '无'} | {cons} | {np.mean(r['latency_s']):.0f}s |")
            if args.with_sam:
                row += f" {r.get('mask_area_ratio', 0):.0%} | {r.get('fragmented', 0)} |"
            lines.append(row)
        lines.append("")
    lines.append("> 覆盖高但自一致性低 = 在猜；掩码占比异常大 = 框太松，SAM 把背景也切进去了。")
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n产物目录：{out.resolve()}")
    print("先看 side_by_side.jpg，再对照 benchmark.md 做决定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

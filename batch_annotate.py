#!/usr/bin/env python3
"""批量自动标注：遍历一个目录里的所有图片，逐张跑 VLM + SAM2。

相比 for 循环调 auto_annotate.py，这个脚本做了三件事：
    1. SAM2 只加载一次（省掉每张图重新加载的 10~30 秒）
    2. 断点续跑（--resume）：已经出过 coco 的图自动跳过
    3. 单张失败不中断整批，最后汇总哪些失败

用法：
    export VLM_API_KEY=sk-xxx
    export VLM_MAX_TOKENS=16000        # 用推理模型时必需

    # 先跑 3 张看看
    python batch_annotate.py --images ~/yolo-pritice/datasets/desktop/raw \\
        --classes-file examples/classes.txt --model claude-haiku-5-5 \\
        --api-base https://opencode.ai/inference/go/openai/v1 \\
        --out runs --limit 3

    # 全量（中断后重跑会自动跳过已完成的）
    python batch_annotate.py --images ... --out runs --resume

产物：
    runs/<图片名>/coco_instances.json    单张 COCO
    runs/<图片名>/preview.jpg            框 + 掩码
    runs/<图片名>/detections.json        原始识别结果
    runs/batch_report.json               整批统计

之后转成训练集：
    python coco2yolo.py --coco "runs/*/coco_instances.json" \\
        --classes-file examples/classes.txt --val-split 0.2 --out dataset/
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from auto_annotate import (DEFAULT_PROMPT, call_vlm, encode_image,
                           mask_to_polygons, normalize_boxes, parse_json_array)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_classes(classes: str | None, classes_file: str | None) -> list[str]:
    if classes_file:
        items = [ln.strip() for ln in Path(classes_file).read_text(encoding="utf-8").splitlines()]
    else:
        items = [c.strip() for c in (classes or "").split(",")]
    out = [c for c in items if c and not c.startswith("#")]
    if not out:
        raise SystemExit("没有类别：用 --classes 或 --classes-file")
    return out


def annotate_one(img_path: Path, classes: list[str], *, api_base: str, api_key: str,
                 model: str, sam, seg_model: str, device: str, max_side: int,
                 prompt_template: str, out_dir: Path, dry_run: bool) -> dict:
    """跑一张图：识别 → 描边 → 写 COCO。返回统计 dict。"""
    b64, (w, h) = encode_image(img_path, max_side)
    prompt = prompt_template.format(classes="\n".join(f"- {c}" for c in classes), w=w, h=h)

    t0 = time.time()
    reply, usage = call_vlm(api_base, api_key, model, b64, prompt)
    latency = time.time() - t0

    dets = normalize_boxes(parse_json_array(reply), w, h, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(img_path))

    counts: dict[str, int] = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    coco = {
        "images": [{"id": 1, "file_name": img_path.name,
                    "file_path": str(img_path.resolve()), "width": w, "height": h}],
        "categories": [{"id": i + 1, "name": c, "supercategory": "object"}
                       for i, c in enumerate(classes)],
        "annotations": [],
    }

    masks = None
    if not dry_run and dets and sam is not None:
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        sr = sam.predict(img_path, bboxes=boxes, device=device, verbose=False)[0]
        masks = (sr.masks.data.cpu().numpy() if sr.masks is not None
                 else np.zeros((0, h, w), np.float32))
        cats = {c: i + 1 for i, c in enumerate(classes)}
        aid = 1
        for mask, d in zip(masks, dets):
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
        (out_dir / "coco_instances.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    # 预览
    canvas = image.copy()
    if masks is not None and len(masks):
        for m in masks:
            sel = m > 0.5
            canvas[sel] = (canvas[sel] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(canvas, d["label"], (x1, max(y1 - 5, 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "preview.jpg"), canvas)
    (out_dir / "detections.json").write_text(json.dumps({
        "image": img_path.name, "model": model, "size": [w, h],
        "counts": counts, "detections": dets, "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "image": img_path.name, "boxes": len(dets), "masks": 0 if masks is None else len(masks),
        "objects": len(coco["annotations"]), "counts": counts,
        "missing": [c for c in classes if c not in counts],
        "latency_s": round(latency, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="批量自动标注")
    ap.add_argument("--images", required=True, help="图片目录，或通配符如 'raw/*.jpg'")
    ap.add_argument("--out", default="runs", help="输出根目录，每张图一个子目录")
    ap.add_argument("--classes", help="类别清单，逗号分隔")
    ap.add_argument("--classes-file", help="类别清单文件")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "gpt-4o"))
    ap.add_argument("--seg-model", default="models/sam2.1_t.pt")
    ap.add_argument("--device", default="cpu", help="SAM2 设备：cpu 或 0")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 = 全部）")
    ap.add_argument("--resume", action="store_true", help="跳过已生成 coco 的图")
    ap.add_argument("--dry-run", action="store_true", help="只识别，不跑 SAM2")
    ap.add_argument("--prompt-file", help="自定义提示词模板")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置 $VLM_API_KEY 或传 --api-key")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 收集图片
    src = Path(args.images).expanduser()
    if any(ch in str(src) for ch in "*?["):
        import glob
        files = sorted(Path(p) for p in glob.glob(str(src)))
        files = [f for f in files if f.suffix.lower() in IMG_EXT]
    elif src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    else:
        raise SystemExit(f"找不到：{src}")
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("没有找到图片")

    # 断点续跑
    pending, skipped = [], 0
    for f in files:
        target = out_root / f.stem / "coco_instances.json"
        if args.resume and target.exists():
            skipped += 1
            continue
        pending.append(f)

    print(f"图片总数 {len(files)} ｜ 已完成跳过 {skipped} ｜ 待处理 {len(pending)}")
    if not pending:
        print("没有需要处理的图（全部已完成）。")
        return 0

    # 只有真的要跑才要求类别清单，纯查状态时不必提供
    classes = load_classes(args.classes, args.classes_file)

    prompt_template = DEFAULT_PROMPT
    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    sam = None
    if not args.dry_run:
        from ultralytics import SAM
        sam = SAM(args.seg_model)
        print(f"SAM2 已加载（整批复用）：{args.seg_model}")

    report, t_start = [], time.time()
    for i, img in enumerate(pending, 1):
        tag = f"[{i}/{len(pending)}] {img.name}"
        try:
            r = annotate_one(img, classes, api_base=args.api_base, api_key=args.api_key,
                             model=args.model, sam=sam, seg_model=args.seg_model,
                             device=args.device, max_side=args.max_side,
                             prompt_template=prompt_template,
                             out_dir=out_root / img.stem, dry_run=args.dry_run)
            report.append({"ok": True, **r})
            print(f"{tag}: {r['boxes']} 框 / {r['objects']} 标注 / {r['latency_s']}s"
                  + (f" / 漏 {len(r['missing'])} 类" if r["missing"] else " / 无漏检"))
        except SystemExit as e:                 # call_vlm 的致命错误
            report.append({"ok": False, "image": img.name, "error": str(e)})
            print(f"{tag}: 失败 — {e}")
        except Exception as e:                 # 单张失败不影响整批
            report.append({"ok": False, "image": img.name, "error": f"{type(e).__name__}: {e}"})
            print(f"{tag}: 失败 — {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)

    ok = [r for r in report if r["ok"]]
    failed = [r for r in report if not r["ok"]]
    elapsed = time.time() - t_start
    coverage = {}
    for r in ok:
        for c in r.get("counts", {}):
            coverage[c] = coverage.get(c, 0) + 1

    summary = {
        "model": args.model, "api_base": args.api_base, "classes": classes,
        "total": len(files), "skipped": skipped, "processed": len(pending),
        "succeeded": len(ok), "failed": len(failed),
        "elapsed_s": round(elapsed, 1),
        "avg_per_image_s": round(elapsed / len(ok), 1) if ok else None,
        "detections_by_class": coverage,
        "never_detected": [c for c in classes if c not in coverage],
        "items": report,
    }
    (out_root / "batch_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：成功 {len(ok)} / 失败 {len(failed)} ｜ 总耗时 {elapsed/60:.1f} 分钟"
          f" ｜ 平均 {summary['avg_per_image_s']}s 每张")
    if coverage:
        print("各类别检出图片数：" + json.dumps(coverage, ensure_ascii=False))
    if summary["never_detected"]:
        print(f"从未检出的类别：{summary['never_detected']}（考虑改类名或换模型）")
    if failed:
        print(f"失败清单：{[r['image'] for r in failed][:10]}")
        print("修好后加 --resume 重跑，只会补失败的那几张。")
    print(f"\n报告：{out_root / 'batch_report.json'}")
    print(f"转训练集：python coco2yolo.py --coco \"{out_root}/*/coco_instances.json\" "
          f"--classes-file <你的类别文件> --val-split 0.2 --out dataset/")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""COCO 实例分割标注 → YOLO 分割数据集目录（可直接用于 ultralytics 训练）。

把 auto_annotate.py 产出的 coco_instances.json 转成这种结构：

    dataset_out/
    ├── images/train/         训练图片副本
    ├── images/val/           验证图片（--val-split > 0 时才有）
    ├── labels/train/         每张图一个同名 .txt
    ├── labels/val/
    ├── data.yaml             ultralytics 直接吃的配置
    └── convert_report.json   转换统计

单个 .txt 的格式（YOLO 分割规范）：
    <class_id> <x1> <y1> <x2> <y2> ...        # 坐标 0~1 归一化，不是像素
    一个物体一行，多边形顶点沿轮廓顺序排列。

用法：
    # 单张（val 复用 train，只够跑通流程）
    python coco2yolo.py --coco out/coco_instances.json --out dataset/

    # 多张 + 二八分（正式训练用这个）
    python coco2yolo.py --coco "runs/*/coco_instances.json" \\
        --classes-file examples/classes.txt --val-split 0.2 --out dataset/
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def expand_coco_args(patterns: list[str]) -> list[Path]:
    """支持直接传路径，也支持通配符。"""
    files: list[Path] = []
    for p in patterns:
        if any(ch in p for ch in "*?["):
            files.extend(Path(x) for x in sorted(glob.glob(p)))
        else:
            files.append(Path(p))
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit("没有找到任何 coco json 文件")
    return files


def load_class_order(coco_files: list[Path], classes_file: str | None) -> list[str]:
    """决定类别编号顺序。

    优先用 --classes-file（多次运行结果一致）；否则从 COCO 的 categories 里推。
    """
    if classes_file:
        items = [ln.split('#')[0].strip()
                 for ln in Path(classes_file).read_text(encoding="utf-8").splitlines()]
        return [c for c in items if c and not c.startswith("#")]

    seen: dict[int, str] = {}
    for f in coco_files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        for c in doc.get("categories", []):
            seen.setdefault(int(c["id"]), c["name"])
    return [seen[k] for k in sorted(seen)]


def polygon_to_normalized(flat: list[float], w: int, h: int) -> np.ndarray:
    """扁平像素点列 → (N,2) 归一化点列，夹到 [0,1]。"""
    pts = np.array(flat, dtype=float).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
    return pts


def dedup_points(pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """去掉相邻重复点，避免退化多边形。"""
    keep = [pts[0]]
    for pt in pts[1:]:
        if np.hypot(*(pt - keep[-1])) > tol:
            keep.append(pt)
    return np.array(keep)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="COCO 实例分割标注 → YOLO 分割数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python coco2yolo.py --coco out/coco_instances.json --out dataset/
  python coco2yolo.py --coco "runs/*/coco_instances.json" \\
      --classes-file examples/classes.txt --val-split 0.2 --out dataset/
""")
    ap.add_argument("--coco", nargs="+", required=True, help="一个或多个 coco json（支持通配符）")
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--classes-file", help="类别清单，决定类别编号顺序（强烈建议固定）")
    ap.add_argument("--image-dir", help="图片所在目录（COCO 里没记录 file_path 时用）")
    ap.add_argument("--val-list", help="验证集名单文件（每行一个图片 stem），"
                                        "适合按场景分层抽样的结果。优先于 --val-pattern")
    ap.add_argument("--val-pattern", nargs="+",
                    help="按文件名通配切验证集，如 \"desk_20261118_162*\"。"
                         "适合用不同日期/批次做真正的泛化测试（优先于 --val-split）")
    ap.add_argument("--val-split", type=float, default=0.0,
                    help="验证集比例，如 0.2。为 0 时 val 复用 train（只够跑通流程，指标无意义）")
    ap.add_argument("--seed", type=int, default=0, help="切分随机种子，保证可复现")
    ap.add_argument("--no-copy-images", action="store_true", help="只写标签，不复制图片")
    ap.add_argument("--skip-empty", action="store_true", help="没有标注的图片不纳入数据集")
    args = ap.parse_args()

    if not 0.0 <= args.val_split < 1.0:
        raise SystemExit("--val-split 必须在 [0, 1) 之间")

    coco_files = expand_coco_args(args.coco)

    # 同一张原图可能出现在多个 run 目录里（例如重复跑、或不同模型各跑一次）。
    # 按 COCO 里的 file_name 去重，保留标注数最多的那份，避免后写覆盖前写。
    picked: dict[str, tuple[Path, int]] = {}
    for cf in coco_files:
        try:
            doc = json.loads(cf.read_text(encoding="utf-8"))
            info = (doc.get("images") or [{}])[0]
            stem = Path(str(info.get("file_name") or cf.parent.name)).stem
            n = len(doc.get("annotations") or [])
        except Exception:
            picked[str(cf)] = (cf, 0)
            continue
        prev = picked.get(stem)
        if prev is None:
            picked[stem] = (cf, n)
        elif n > prev[1]:
            print(f"[去重] 同一张图 {stem}：用标注更多的 {cf}（{n} vs {prev[1]}），"
                  f"丢弃 {prev[0]}")
            picked[stem] = (cf, n)
        else:
            print(f"[去重] 同一张图 {stem}：保留 {prev[0]}（{prev[1]} 条），跳过 {cf}（{n} 条）")
    coco_files = [v[0] for v in picked.values()]

    classes = load_class_order(coco_files, args.classes_file)
    if not classes:
        raise SystemExit("拿不到类别清单")
    cls_id = {name: i for i, name in enumerate(classes)}

    out = Path(args.out)
    tr_img, tr_lbl = out / "images" / "train", out / "labels" / "train"
    va_img, va_lbl = out / "images" / "val", out / "labels" / "val"
    for d in (tr_img, tr_lbl):
        d.mkdir(parents=True, exist_ok=True)

    report, total_obj, skipped = [], 0, 0
    dropped_parts = [0]                          # 被丢弃的多余掩码段数
    pairs: list[tuple[Path | None, Path]] = []   # (图片, 标签) 供切分用

    for cf in coco_files:
        doc = json.loads(cf.read_text(encoding="utf-8"))
        if not doc.get("images"):
            print(f"[跳过] {cf.name}：没有 images 段")
            continue
        info = doc["images"][0]
        w, h = int(info["width"]), int(info["height"])
        fname = info["file_name"]
        stem = Path(fname).stem

        # 找原图：file_path → image-dir → coco 同级目录
        src = None
        for cand in (info.get("file_path"),
                     args.image_dir and Path(args.image_dir) / fname,
                     cf.parent / fname):
            if cand and Path(cand).exists():
                src = Path(cand)
                break

        cat_name = {int(c["id"]): c["name"] for c in doc.get("categories", [])}
        lines = []
        for ann in doc.get("annotations", []):
            name = cat_name.get(int(ann["category_id"]))
            if name is None or name not in cls_id:
                print(f"  [跳过] 未知类别 id={ann['category_id']} ({name})")
                continue

            polys = [q for q in ann.get("segmentation", [])
                     if isinstance(q, list) and len(q) >= 6]
            if not polys:
                continue
            if len(polys) > 1:
                # YOLO 分割每行只能一个多边形；多段掩码拆成多行会被当成多个实例。
                # 保留面积最大的那段（通常是主体），其余丢弃并计数。
                def area(q):
                    return abs(cv2.contourArea(np.array(q, dtype=np.float32).reshape(-1, 1, 2)))
                polys = [max(polys, key=area)]
                dropped_parts[0] += 1

            for poly in polys:
                pts = dedup_points(polygon_to_normalized(poly, w, h))
                if len(pts) < 3:
                    continue
                flat = pts.reshape(-1)
                lines.append(f"{cls_id[name]} " + " ".join(f"{v:.6f}" for v in flat))

        if not lines and args.skip_empty:
            skipped += 1
            print(f"[跳过] {cf.name}：没有标注")
            continue

        lbl = tr_lbl / f"{stem}.txt"
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        img_copy = None
        if src is None:
            print(f"[警告] {cf.name}：找不到原图 {fname}，只写了标签。用 --image-dir 指定图片目录。")
        elif not args.no_copy_images:
            img_copy = tr_img / fname
            shutil.copy2(src, img_copy)

        pairs.append((img_copy, lbl))
        total_obj += len(lines)
        report.append({"coco": str(cf), "image": fname, "objects": len(lines)})
        print(f"[{len(report)}] {fname}: {len(lines)} 个多边形")

    # ---------- 切分验证集 ----------
    if args.val_list:
        names = {ln.split('#')[0].strip() for ln in Path(args.val_list).read_text(encoding='utf-8').splitlines()}
        names = {n for n in names if n}
        for d in (va_img, va_lbl):
            d.mkdir(parents=True, exist_ok=True)
        moved = 0
        for img, lbl in list(pairs):
            if lbl.stem in names:
                if lbl.exists():
                    shutil.move(str(lbl), va_lbl / lbl.name)
                if img and img.exists():
                    shutil.move(str(img), va_img / img.name)
                moved += 1
        val_path = "images/val" if moved else "images/train"
        val_note = f"按名单切分：验证集 {moved} 张 / 训练集 {len(pairs) - moved} 张"
        print(f"\n切分：{val_note}")
        not_found = names - {lbl.stem for _, lbl in pairs}
        if not_found:
            print(f"[提示] 名单里有 {len(not_found)} 个名字在数据集中找不到（可能被 --skip-empty 排除）")
    elif args.val_pattern:
        from fnmatch import fnmatch
        for d in (va_img, va_lbl):
            d.mkdir(parents=True, exist_ok=True)
        moved = 0
        for img, lbl in list(pairs):
            name = lbl.stem + ".jpg"
            if any(fnmatch(name, pat) or fnmatch(lbl.stem, pat) for pat in args.val_pattern):
                if lbl.exists():
                    shutil.move(str(lbl), va_lbl / lbl.name)
                if img and img.exists():
                    shutil.move(str(img), va_img / img.name)
                moved += 1
        val_path = "images/val" if moved else "images/train"
        val_note = (f"按文件名通配切分：验证集 {moved} 张 / 训练集 {len(pairs) - moved} 张"
                    f"（patterns={args.val_pattern}）")
        print(f"\n切分：{val_note}")
        if moved in (0, len(pairs)):
            print("[警告] 通配符把全部或零张图切到了验证集，请检查 pattern")
    elif args.val_split > 0 and len(pairs) >= 5:
        rng = random.Random(args.seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(pairs) * args.val_split)))
        n_val = min(n_val, len(pairs) - 1)       # 至少留 1 张给训练
        for d in (va_img, va_lbl):
            d.mkdir(parents=True, exist_ok=True)
        for i in idx[:n_val]:
            img, lbl = pairs[i]
            if lbl.exists():
                shutil.move(str(lbl), va_lbl / lbl.name)
            else:
                print(f"[警告] 切分时找不到标签 {lbl.name}，跳过（可能是重复原图）")
            if img and img.exists():
                shutil.move(str(img), va_img / img.name)
        val_path = "images/val"
        val_note = f"验证集 {n_val} 张 / 训练集 {len(pairs) - n_val} 张（seed={args.seed}）"
        print(f"\n切分：{val_note}")
    elif args.val_split > 0:
        val_path = "images/train"
        val_note = f"只有 {len(pairs)} 张图，不够切分，val 复用 train"
        print(f"\n[提示] {val_note}")
    else:
        val_path = "images/train"
        val_note = "未切分，val 复用 train（只能跑通流程，指标无意义）"
        print("\n[警告] 没有 --val-split，val 复用 train —— 训练出来的指标是虚高的。"
              "\n        正式训练请加 --val-split 0.2，并准备足够多的图。")

    # ---------- data.yaml ----------
    yaml_lines = [
        f"path: {json.dumps(str(out.resolve()), ensure_ascii=False)}",
        "train: images/train",
        f"val: {val_path}",
        "names:",
    ]
    yaml_lines += [f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    (out / "convert_report.json").write_text(json.dumps({
        "classes": classes, "images": len(report), "polygons": total_obj,
        "skipped": skipped, "dropped_extra_parts": dropped_parts[0],
        "val_note": val_note, "details": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：{len(report)} 张图 / {total_obj} 个多边形 / {len(classes)} 个类别")
    if dropped_parts[0]:
        print(f"丢弃多余掩码段 {dropped_parts[0]} 处（YOLO 分割每行只能一个多边形）")
    if skipped:
        print(f"跳过空标注图片 {skipped} 张")
    print(f"数据集 → {out.resolve()}")
    print(f"训练：yolo segment train model=yolo26n-seg.pt data={out / 'data.yaml'} epochs=100")
    return 0


if __name__ == "__main__":
    sys.exit(main())

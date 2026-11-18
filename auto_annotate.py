#!/usr/bin/env python3
"""VLM + SAM2 自动标注：大模型负责"是什么、在哪"，SAM2 负责"边界画准"。

为什么这样分工：
    传统预标注模型（COCO 训练的 YOLO）类别是写死的 80 类，遇到"胶棒""指甲钳"
    这类目标只能往最接近的类上硬凑。视觉大模型靠常识能认对，但它给的框不够
    精确；SAM2 拿到框就能切出逐像素贴合物体的掩码。两者互补。

流程：
    图片 + 类别清单 ──→ VLM（OpenAI 兼容接口）──→ [{label, bbox}, ...]
                                                  │
                                                  ▼
                                            SAM2（框当提示）──→ 精确掩码
                                                  │
                                                  ▼
                                      COCO 多边形 JSON + 预览图

用法：
    export VLM_API_KEY=sk-xxx
    python auto_annotate.py --image photo.jpg \
        --api-base https://api.openai.com/v1 --model gpt-4o \
        --classes "monitor,cup,stapler" --out out/

    # 只验证大模型识别，不跑 SAM2（快、省算力）
    python auto_annotate.py ... --dry-run

产物（--out 目录下）：
    preview_boxes.jpg    只有大模型给的框，用来判断认类别准不准
    detections.json      大模型原始结果 + 解析后的框
    coco_instances.json  COCO 实例分割标注，可直接导入 CVAT / 转 YOLO
    preview.jpg          框 + SAM2 掩码叠加图
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PROMPT = """你是图像标注助手。请只输出 JSON，不要任何解释文字。

图中有这些类别，请找出每一类在图中出现的**每一个实例**：
{classes}

输出格式（严格 JSON 数组）：
[{{"label": "类别名（必须来自上面清单）", "bbox": [x1, y1, x2, y2]}}]

规则：
1. bbox 用 **0~1000 归一化坐标**：x 和 y 都按 0-1000 缩放，左上角 (0,0)、
   右下角 (1000,1000)（不看图片宽高比例，两个轴独立归一化）
2. 只标清单里的类别，图中没有的类别不要编造
3. 同一个类别的多个实例要分别列出
4. 框要贴紧物体外沿，宽度或高度不要小于 15（归一化后）
5. 不确定是不是目标时不要输出

只输出 JSON 数组，第一个字符必须是 [，最后一个必须是 ]。"""

PALETTE = [
    (75, 68, 125), (15, 157, 229), (32, 64, 192), (198, 56, 82), (120, 120, 0),
    (0, 96, 0), (142, 234, 0), (177, 73, 28), (192, 32, 128), (35, 186, 79),
]


# --------------------------------------------------------------------------- 输入


def load_classes(args) -> list[str]:
    """类别清单：--classes 逗号分隔，或 --classes-file 每行一个。"""
    if args.classes_file:
        text = Path(args.classes_file).read_text(encoding="utf-8")
        items = [ln.split('#')[0].strip() for ln in text.splitlines()]
    else:
        items = [c.strip() for c in (args.classes or "").split(",")]
    classes = [c for c in items if c and not c.startswith("#")]
    if not classes:
        raise SystemExit("没有类别：用 --classes 或 --classes-file 指定")
    return classes


def encode_image(path: Path, max_side: int) -> tuple[str, tuple[int, int]]:
    """读图 → base64 JPEG。限制最长边以控制 token 消耗。"""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读不了图：{path}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise SystemExit("图片编码失败")
    return base64.b64encode(buf.tobytes()).decode(), (w, h)


# --------------------------------------------------------------------------- VLM


def build_headers(api_base: str, api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # 有些网关（Cloudflare 前置）会按 UA 拦掉 python-urllib，返回 403 / error 1010
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    }
    # OpenCode Go 订阅通道要求会话 ID，缺了会报 MissingSessionID
    if "opencode.ai" in api_base:
        headers["x-opencode-session"] = str(uuid.uuid4())
    return headers


def call_vlm(api_base: str, api_key: str, model: str, b64: str, prompt: str,
             timeout: int = 180, retries: int = 2) -> tuple[str, dict]:
    """调一次 OpenAI 兼容的 chat/completions。返回 (回复文本, usage)。"""
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
        # 推理型模型（如 deepseek-v4.1-flash）会把输出预算先花在思考上，
        # 预算不够就只返回空正文 + finish_reason=length。用 VLM_MAX_TOKENS 调高。
        "max_tokens": int(os.environ.get("VLM_MAX_TOKENS", 4000)),
    }
    payload = json.dumps(body).encode()
    headers = build_headers(api_base, api_key)

    last_err = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            msg = data["choices"][0]["message"]
            # 推理型模型的正文可能在 content，也可能只给了 reasoning_content
            content = msg.get("content") or ""
            return content, data.get("usage") or {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:500]
            last_err = f"HTTP {e.code}: {detail}"
            # 4xx 里只有 429 值得重试
            if e.code != 429 and e.code < 500:
                break
        except Exception as e:  # 网络抖动
            last_err = str(e)
        if attempt < retries:
            wait = 2 ** attempt
            print(f"  请求失败（{last_err[:80]}），{wait}s 后重试…")
            time.sleep(wait)
    raise SystemExit(f"API 调用失败：{last_err}")


# --------------------------------------------------------------------------- 解析


def parse_json_array(text: str) -> list[dict]:
    """从模型回复里抠 JSON 数组。常带 ```json 围栏或前后废话。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        out = json.loads(text)
        if isinstance(out, list):
            return out
    except json.JSONDecodeError:
        pass
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        try:
            out = json.loads(text[i:j + 1])
            if isinstance(out, list):
                return out
        except json.JSONDecodeError:
            pass
    return []


def match_label(raw: str, valid: list[str]) -> str | None:
    """类别名宽松匹配：大小写、空格、下划线差异都算同一个。"""
    norm = lambda s: s.lower().replace(" ", "_").replace("-", "_")
    for v in valid:
        if norm(v) == norm(raw):
            return v
    for v in valid:
        if norm(v) in norm(raw) or norm(raw) in norm(v):
            return v
    return None


def normalize_boxes(items: list[dict], w: int, h: int, valid: list[str]) -> list[dict]:
    """统一各种坐标写法：0~1、0~1000、绝对像素。"""
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        label_raw = str(it.get("label") or it.get("class") or it.get("name") or "").strip()
        box = it.get("bbox") or it.get("box") or it.get("box_2d") or it.get("xyxy")
        if not label_raw or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        label = match_label(label_raw, valid)
        if label is None:
            print(f"  [跳过] 未知类别：{label_raw}")
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue

        mx = max(x1, y1, x2, y2)
        if mx <= 1.5:                      # 0~1
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
        elif mx <= 1000:                   # 0~1000（提示词约定的写法）
            x1, x2 = x1 / 1000 * w, x2 / 1000 * w
            y1, y2 = y1 / 1000 * h, y2 / 1000 * h
        # 大于 1000 视为像素坐标，原样使用

        x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
        y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return out


# --------------------------------------------------------------------------- SAM2


def mask_to_polygons(bins: np.ndarray, min_area: float) -> list[list[float]]:
    """二值掩码 → COCO 多边形（扁平点列）。过滤碎片、简化顶点。"""
    import supervision as sv

    polys = []
    for poly in sv.mask_to_polygons(bins.astype(bool)):
        if len(poly) < 3:
            continue
        pts = poly.reshape(-1, 1, 2).astype(np.float32)
        eps = 0.002 * cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, max(eps, 1.0), True)
        if len(approx) >= 3 and cv2.contourArea(approx) >= min_area:
            polys.append(approx.reshape(-1).astype(float).tolist())
    return polys


# --------------------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(
        description="VLM + SAM2 自动标注：大模型识别，SAM2 描边，输出 COCO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  export VLM_API_KEY=sk-xxx
  python auto_annotate.py --image desk.jpg --model gpt-4o \\
      --api-base https://api.openai.com/v1 --classes "monitor,cup,stapler" --out out/
""")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--classes", help="类别清单，逗号分隔")
    ap.add_argument("--classes-file", help="类别清单文件，每行一个（可带 # 注释）")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"),
                    help="OpenAI 兼容接口地址（含 /v1）")
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""),
                    help="API key，建议用环境变量 VLM_API_KEY")
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "gpt-4o"), help="视觉模型名")
    ap.add_argument("--seg-model", default="models/sam2.1_t.pt",
                    help="SAM2 权重路径；不存在会自动下载（tiny/small/base/large 换文件名即可）")
    ap.add_argument("--device", default="cpu", help="SAM2 设备：cpu 或 0（第一块 GPU）")
    ap.add_argument("--max-side", type=int, default=1280, help="送图前把最长边缩到这个值，控制 token")
    ap.add_argument("--out", default="autoanno_out", help="输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只跑大模型，不跑 SAM2")
    ap.add_argument("--save-raw", action="store_true", help="保存大模型原始回复，便于排查")
    ap.add_argument("--prompt-file", help="自定义提示词模板（可用 {classes} 和 {w}{h} 占位）")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置环境变量 VLM_API_KEY 或传 --api-key")

    classes = load_classes(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"图片不存在：{image_path}")

    b64, (w, h) = encode_image(image_path, args.max_side)
    print(f"图像 {w}x{h} | 类别 {len(classes)} 个 | 模型 {args.model}")

    # ---------- 第一步：大模型识别 ----------
    template = DEFAULT_PROMPT
    if args.prompt_file:
        template = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt = template.format(classes="\n".join(f"- {c}" for c in classes), w=w, h=h)

    print("[1/2] 调用视觉大模型识别…")
    reply, usage = call_vlm(args.api_base, args.api_key, args.model, b64, prompt)
    if args.save_raw:
        (out / "raw_reply.txt").write_text(reply or "", encoding="utf-8")
    if usage:
        print(f"  token: 输入 {usage.get('prompt_tokens', '?')} / 输出 {usage.get('completion_tokens', '?')}")

    raw_items = parse_json_array(reply)
    if not raw_items:
        print("模型没返回可用 JSON。原始回复前 800 字：")
        print((reply or "")[:800])
        return 1

    dets = normalize_boxes(raw_items, w, h, classes)
    print(f"  解析出 {len(dets)} 个有效框")

    # 框预览：判断"认类别"准不准
    preview = cv2.imread(str(image_path))
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(preview, d["label"], (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "preview_boxes.jpg"), preview)

    counts: dict[str, int] = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    (out / "detections.json").write_text(json.dumps({
        "model": args.model, "api_base": args.api_base, "image": image_path.name,
        "size": [w, h], "counts": counts, "detections": dets, "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  类别统计: {json.dumps(counts, ensure_ascii=False)}")
    print(f"  框预览 → {out / 'preview_boxes.jpg'}")
    missing = [c for c in classes if c not in counts]
    if missing:
        print(f"  未检出: {missing}")

    if args.dry_run:
        print("\n--dry-run：跳过 SAM2。先看 preview_boxes.jpg 判断类别和位置对不对。")
        return 0
    if not dets:
        print("没有可用框，无法跑 SAM2。")
        return 1

    # ---------- 第二步：SAM2 描边 ----------
    print("[2/2] SAM2 生成掩码…")
    from ultralytics import SAM

    sam = SAM(args.seg_model)
    boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
    result = sam.predict(image_path, bboxes=boxes, device=args.device, verbose=False)[0]
    masks = (result.masks.data.cpu().numpy() if result.masks is not None
             else np.zeros((0, h, w), np.float32))
    print(f"  SAM2 出 {len(masks)} 个掩码")

    cat_id = {c: i + 1 for i, c in enumerate(classes)}
    coco = {
        "info": {"description": f"VLM({args.model}) + SAM2 auto annotation",
                 "date_created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        # file_path 存绝对路径，方便 coco2yolo.py 找到原图（COCO 标准里没有这个字段，但无害）
        "images": [{"id": 1, "file_name": image_path.name,
                    "file_path": str(image_path.resolve()), "width": w, "height": h}],
        "categories": [{"id": i + 1, "name": c, "supercategory": "object"}
                       for i, c in enumerate(classes)],
        "annotations": [],
    }
    ann_id, stats = 1, {}
    min_area = 0.0005 * w * h          # 小于整图 0.05% 的连通块当噪声丢掉

    for mask, d in zip(masks, dets):
        bins = (mask > 0.5).astype(np.uint8)
        polys = mask_to_polygons(bins, min_area)
        if not polys:
            continue
        ys, xs = np.where(bins > 0)
        coco["annotations"].append({
            "id": ann_id, "image_id": 1, "category_id": cat_id[d["label"]],
            "segmentation": polys,          # 一个物体一条标注，多段掩码放同一个列表
            "area": float(bins.sum()),
            "bbox": [float(xs.min()), float(ys.min()),
                     float(xs.max() - xs.min()), float(ys.max() - ys.min())],
            "iscrowd": 0,
        })
        ann_id += 1
        stats[d["label"]] = stats.get(d["label"], 0) + 1

        color = PALETTE[cat_id[d["label"]] % len(PALETTE)]
        contours, _ = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, color, 2)

    cv2.imwrite(str(out / "preview.jpg"), preview)
    (out / "coco_instances.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print(f"\n写出 {len(coco['annotations'])} 条标注 → {out / 'coco_instances.json'}")
    print(f"预览 → {out / 'preview.jpg'}")
    print(f"最终统计: {json.dumps(stats, ensure_ascii=False)}")
    if missing:
        print(f"仍未检出: {missing}（可试着把类名写得更具体，或用更强的视觉模型）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

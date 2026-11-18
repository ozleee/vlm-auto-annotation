#!/usr/bin/env python3
"""列出接口上可用的模型，标出哪些能看图，方便挑模型。

    export VLM_API_KEY=sk-xxx
    python list_models.py --api-base https://opencode.ai/inference/go/openai/v1

    # 只看能看图的
    python list_models.py --api-base ... --vision-only

    # 机器可读
    python list_models.py --api-base ... --json

注意：不同网关返回的元数据详略不同。有的会给出 modalities（哪些输入类型支持），
有的啥都没有。**元数据只能当参考，最终要拿自己的图跑一次才算数**——见 README 的横评一节。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# 元数据缺失时的兜底推断：这些模型家族普遍支持图像输入
VISION_HINTS = ("vl", "vision", "gpt-4o", "gpt-5", "gpt-6", "gemini", "claude",
                "qwen", "glm-4v", "glm-5", "kimi", "mimo", "minimax", "grok",
                "pixtral", "llava", "internvl", "step", "muse", "hy4", "longcat")


def fetch_models(api_base: str, api_key: str, timeout: int = 30) -> list[dict]:
    url = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url} 返回 {e.code}：{e.read().decode()[:300]}")
    except Exception as e:
        raise SystemExit(f"请求失败：{e}")
    return data.get("data") or data.get("models") or []


def input_modalities(m: dict) -> list[str]:
    mod = (m.get("modalities") or {}).get("input")
    if isinstance(mod, list) and mod:
        return [str(x).split("/")[0] for x in mod]
    return []


def can_see_images(m: dict) -> bool:
    mods = input_modalities(m)
    if mods:
        return "image" in mods
    name = str(m.get("id") or m.get("name") or "").lower()
    return any(h in name for h in VISION_HINTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="列出可用模型并标出能看图的")
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--vision-only", action="store_true", help="只列能看图的模型")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺 API key：设置环境变量 VLM_API_KEY")

    models = fetch_models(args.api_base, args.api_key)
    rows = []
    for m in models:
        mid = m.get("id") or m.get("name") or "?"
        lim = m.get("limit") or {}
        rows.append({
            "id": mid,
            "name": m.get("name") or mid,
            "context": lim.get("context") or m.get("context_length") or 0,
            "inputs": input_modalities(m),
            "vision": can_see_images(m),
            "description": (m.get("description") or "")[:80],
        })
    rows.sort(key=lambda r: r["id"])

    if args.vision_only:
        rows = [r for r in rows if r["vision"]]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    total = len(models)
    n_vision = sum(1 for r in rows if r["vision"]) if args.vision_only else \
        sum(1 for m in models if can_see_images(m))
    print(f"接口：{args.api_base}")
    print(f"共 {total} 个模型，其中能看图 {n_vision} 个\n")

    if not rows:
        print("（没有匹配的模型）")
        return 0

    w = max(len(r["id"]) for r in rows) + 2
    print(f"{'模型 id'.ljust(w)} {'上下文':>10}  {'输入':28} 说明")
    print("-" * (w + 60))
    for r in rows:
        ctx = f"{r['context']:,}" if r["context"] else "-"
        inputs = ",".join(dict.fromkeys(r["inputs"])) or ("image?" if r["vision"] else "text")
        print(f"{r['id'].ljust(w)} {ctx:>10}  {inputs:28} {r['description']}")
    print("\n提示：把喜欢的模型名传给 auto_annotate.py 的 --model，或 benchmark_models.py 的 --models 对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

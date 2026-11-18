#!/usr/bin/env bash
# 训练入口：自动用本机 GPU，产物统一放 train_runs/（不会和标注结果 runs/ 混）。
#
# 用法：
#   ./train.sh --data dataset_v2/data.yaml
#   ./train.sh --data dataset_v2/data.yaml --model yolo26n-seg.pt --epochs 100 --batch 16
#   ./train.sh --data ... --device cpu        # 强制 CPU
#
# 会自动：
#   1. 挑一个装了 ultralytics 的 Python
#   2. 检查 CUDA 是否可用，可用就 device=0，否则退回 cpu 并提醒
#   3. 设好 LD_LIBRARY_PATH / YOLO_CONFIG_DIR / MPLCONFIGDIR
#   4. 输出到 train_runs/<时间戳>/

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 默认参数（命令行可覆盖）---
DATA=""
MODEL="yolo26n-seg.pt"
EPOCHS="100"
IMGSZ="640"
BATCH="16"
DEVICE=""
NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data)    DATA="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --imgsz)   IMGSZ="$2"; shift 2 ;;
    --batch)   BATCH="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --name)    NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [ -z "${DATA}" ]; then
  echo "缺 --data（dataset/data.yaml）" >&2
  exit 2
fi
if [ ! -f "${DATA}" ]; then
  echo "找不到 ${DATA}" >&2
  exit 2
fi

# --- Python ---
PY="${VLM_PY:-}"
if [ -z "${PY}" ]; then
  for cand in "$DIR/../.venv-cv/bin/python" "$DIR/.venv/bin/python" "$(command -v python3 || true)"; do
    if [ -n "${cand}" ] && [ -x "${cand}" ] && "${cand}" -c "import ultralytics" >/dev/null 2>&1; then
      PY="${cand}"; break
    fi
  done
fi
if [ -z "${PY}" ] || [ ! -x "${PY}" ]; then
  echo "找不到装了 ultralytics 的 Python" >&2
  exit 1
fi
YOLO_BIN="$(dirname "${PY}")/yolo"
if [ ! -x "${YOLO_BIN}" ]; then
  echo "找不到 yolo 命令：${YOLO_BIN}" >&2
  exit 1
fi

# --- 环境 ---
if [ -d "${HOME}/miniconda3/lib" ]; then
  export LD_LIBRARY_PATH="${HOME}/miniconda3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${DIR}/.yolo-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DIR}/.yolo-config}"
mkdir -p "${YOLO_CONFIG_DIR}" 2>/dev/null || true

# --- GPU 检测 ---
if [ -z "${DEVICE}" ]; then
  if "${PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="0"
    GPUNAME="$("${PY}" -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null || echo GPU)"
    echo "检测到 CUDA：${GPUNAME} → device=0"
  else
    DEVICE="cpu"
    echo "⚠️  未检测到可用 CUDA，退回 CPU 训练（会很慢）。"
    echo "   如果你确认有显卡，检查驱动：nvidia-smi"
  fi
fi

[ -z "${NAME}" ] && NAME="exp_$(date +%m%d_%H%M)"

echo "模型=${MODEL}  数据=${DATA}"
echo "epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH} device=${DEVICE}"
echo "输出=train_runs/${NAME}"
echo

exec "${YOLO_BIN}" segment train \
  model="${MODEL}" data="${DATA}" \
  epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" device="${DEVICE}" \
  project="${DIR}/train_runs" name="${NAME}" exist_ok=True

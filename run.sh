#!/usr/bin/env bash
# 统一入口：把环境问题都封在这里，你只需要记一条命令。
#
# 它会自动处理：
#   - 用哪个 Python（优先 ../.venv-cv，其次 $VLM_PY，最后系统 python3）
#   - LD_LIBRARY_PATH 指向 conda 的 lib（否则 scipy 会报 CXXABI_1.3.15 找不到）
#   - YOLO_CONFIG_DIR 指向可写目录（否则 ultralytics 每次往 /tmp 写配置）
#
# 用法：
#   ./run.sh models                       列出可用模型
#   ./run.sh one --image a.jpg --classes "cup,monitor"    单张
#   ./run.sh batch --images DIR --classes-file examples/classes.txt    批量
#   ./run.sh bench --image a.jpg --models m1 m2 --with-sam              横评
#   ./run.sh convert --coco "runs/*/coco_instances.json" --out dataset  转数据集
#
# 也可以直接给脚本名：
#   ./run.sh auto_annotate.py --image a.jpg --classes "cup"

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Python 解释器 ---
PY="${VLM_PY:-}"
if [ -z "${PY}" ]; then
  for cand in "$DIR/../.venv-cv/bin/python" "$DIR/.venv/bin/python" "$(command -v python3 || true)"; do
    if [ -n "${cand}" ] && [ -x "${cand}" ] && "${cand}" -c "import ultralytics" >/dev/null 2>&1; then
      PY="${cand}"; break
    fi
  done
fi
if [ -z "${PY}" ] || [ ! -x "${PY}" ]; then
  echo "找不到装了 ultralytics 的 Python。" >&2
  echo "先装依赖：pip install -r \"$DIR/requirements.txt\"" >&2
  echo "或用 VLM_PY=/path/to/python ./run.sh ... 指定解释器。" >&2
  exit 1
fi

# --- 运行环境 ---
# conda 的 scipy 需要它自带的 libstdc++，否则报 CXXABI_1.3.15 not found
if [ -d "${HOME}/miniconda3/lib" ]; then
  export LD_LIBRARY_PATH="${HOME}/miniconda3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# ultralytics 会往配置目录写设置；默认目录可能不可写
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${HOME}/.yolo-config}"
mkdir -p "${YOLO_CONFIG_DIR}" 2>/dev/null || true

# --- 子命令映射 ---
if [ $# -eq 0 ]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

NAME="$1"; shift
case "${NAME}" in
  one|single|auto)      SCRIPT="auto_annotate.py" ;;
  batch|all)            SCRIPT="batch_annotate.py" ;;
  bench|compare)        SCRIPT="benchmark_models.py" ;;
  convert|yolo)         SCRIPT="coco2yolo.py" ;;
  models|list)          SCRIPT="list_models.py" ;;
  discover|find)        SCRIPT="discover.py" ;;
  *.py)                 SCRIPT="${NAME}" ;;
  -h|--help|help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *)
    echo "未知子命令：${NAME}" >&2
    echo "可用：one / batch / bench / convert / models，或直接给 .py 文件名" >&2
    exit 2 ;;
esac

if [ ! -f "${DIR}/${SCRIPT}" ]; then
  echo "找不到脚本：${DIR}/${SCRIPT}" >&2
  exit 2
fi

exec "${PY}" "${DIR}/${SCRIPT}" "$@"

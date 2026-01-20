#!/usr/bin/env bash
set -euo pipefail

# ===========================
# 只改这里：参数总开关区
# ===========================

# 运行模式：dispatch（默认：自动拆 subset + 多 GPU 排队跑） / worker（只跑一个任务，调试用）
MODE="dispatch"

# dispatch 用：可用 GPU（物理 id，逗号分隔）
# 例：一张卡用 0；两张卡用 4,7；八卡想用 0,1,2,3,4,5,6,7
GPUS="1"

# dispatch 用：只跑哪些数据集（写 configs/default.json 里的 datasets key，逗号分隔；空=全跑）
# 例：ONLY="aime25" 或 ONLY="aime25,math_500"
# ONLY="aime25,aime24"
ONLY=""

# dispatch 用：本次批次 tag（空=自动时间戳 run_YYYYmmdd_HHMMSS）
# TAG="qwen3_4b_aime25_0120_1"
# TAG="qwen3_4b_all_0120_3"
# TAG="qwen3_4b_aime2425_0120_1"
TAG="ds_1.5b_all_0120_1"

# worker 用（仅 MODE=worker 时生效）：
# 跑哪个数据集（default.json 的 key）
DATASET="aime25"

# worker 用：跑哪个 subset（空=按 default.json 的 subset_list 或 default）
SUBSET="AIME2025-I"

# worker 用：绑定到哪张物理 GPU（单个整数）
GPU="0"

# ===========================
# 脚本主体
# ===========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/run_task.py"

echo "[RUN] mode=${MODE}"
echo "[RUN] runner=${PY}"

case "${MODE}" in
  dispatch)
    echo "[RUN] gpus=${GPUS}"
    [[ -n "${ONLY}" ]] && echo "[RUN] only=${ONLY}"
    [[ -n "${TAG}"  ]] && echo "[RUN] tag=${TAG}"
    echo

    cmd=(python "${PY}" --mode dispatch --gpus "${GPUS}")
    [[ -n "${ONLY}" ]] && cmd+=(--only "${ONLY}")
    [[ -n "${TAG}"  ]] && cmd+=(--tag  "${TAG}")

    echo "[CMD] ${cmd[*]}"
    exec "${cmd[@]}"
    ;;

  worker)
    echo "[RUN] dataset=${DATASET}"
    [[ -n "${SUBSET}" ]] && echo "[RUN] subset=${SUBSET}"
    [[ -n "${GPU}"    ]] && echo "[RUN] gpu=${GPU}"
    echo "[RUN] tag=${TAG:-<required for worker>}"
    echo

    if [[ -z "${DATASET}" ]]; then
      echo "[ERR] worker 模式必须设置 DATASET"
      exit 2
    fi
    if [[ -z "${TAG}" ]]; then
      echo "[ERR] worker 模式必须设置 TAG（用于输出目录名与日志名）"
      exit 2
    fi

    cmd=(python "${PY}" --mode worker --dataset "${DATASET}" --tag "${TAG}")
    [[ -n "${SUBSET}" ]] && cmd+=(--subset "${SUBSET}")
    [[ -n "${GPU}"    ]] && cmd+=(--gpu "${GPU}")

    echo "[CMD] ${cmd[*]}"
    exec "${cmd[@]}"
    ;;

  *)
    echo "[ERR] MODE 只能是 dispatch 或 worker，你给的是: ${MODE}"
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/run_task.py"
TAG="run_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Batch Tag: ${TAG}"
echo "[INFO] Runner: ${PY}"

python "${PY}" --dataset "math_500" --tag "${TAG}"
# python "${PY}" --dataset "aime24"   --tag "${TAG}"
# python "${PY}" --dataset "aime25"   --tag "${TAG}"

echo "[DONE] Finished."

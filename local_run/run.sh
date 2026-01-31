#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/run_task.py"
TAG="run_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Batch Tag: ${TAG}"
echo "[INFO] Runner: ${PY}"

python "${PY}" --dataset "aime25"   --tag "${TAG}"
python "${PY}" --dataset "aime24"   --tag "${TAG}"
python "${PY}" --dataset "math_500" --tag "${TAG}"

# python "${PY}" --dataset "aime25"   --tag "16k_max_tokens_0120_4"
# python "${PY}" --dataset "aime24"   --tag "16k_max_tokens_0120_1"
# python "${PY}" --dataset "aime25"   --tag "16k_max_tokens_0120_2"
# python "${PY}" --dataset "aime24"   --tag "16k_max_tokens_0120_2"
# python "${PY}" --dataset "aime25"   --tag "16k_max_tokens_0120_3"
# python "${PY}" --dataset "aime24"   --tag "16k_max_tokens_0120_3"
# python "${PY}" --dataset "math_500" --tag "16k_max_tokens_0120_1"

echo "[DONE] Finished."

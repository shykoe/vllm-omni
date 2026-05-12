#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p /efs/kwinsheng/vllm-omni/dev/output
export VLLM_OMNI_STORAGE_PATH=/efs/kwinsheng/vllm-omni/dev/output
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 必须运行真实 Python 文件, 不能使用 `python - <<'PY'` heredoc。
# vLLM-Omni worker 使用 multiprocessing spawn, 子进程会重新加载主入口文件;
# 如果主入口来自 stdin, 子进程会尝试加载 `${SCRIPT_DIR}/<stdin>` 并报 FileNotFoundError。
python "${SCRIPT_DIR}/serve_ltx2_3_t2av.py"
#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random checkpoints/SmolVLA_sim_click_bell 5 \
#   --pytorch_device cuda --headless true --eval_video_log true
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    if [[ -d "$WORKSPACE_ROOT/EmbodiChain" ]]; then
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    else
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    fi
fi
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

POLICY_NAME=smolvla

TASK_NAME="${1:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
CHECKPOINT_PATH="${3:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

# SmolVLA worker 解释器默认值 (10/2 坑根治第二层): worker 里的 lerobot 用 PEP 695
# 泛型语法需 py>=3.12, 未导 SMOLVLA_PYTHON 时此前回落 eval 主进程 (EmbodiChain
# py3.11) → worker SyntaxError → BrokenPipe。默认指向本机已知 3.13 venv;显式
# 导出永远优先;若默认候选也不在,保持原行为由 deploy_policy.py 版本预检兜底拦截。
if [[ -z "${SMOLVLA_PYTHON:-}" ]]; then
    _DEFAULT_SMOLVLA_PY="$WORKSPACE_ROOT/libero-act-project/.venv/bin/python"
    [[ -x "$_DEFAULT_SMOLVLA_PY" ]] && export SMOLVLA_PYTHON="$_DEFAULT_SMOLVLA_PY"
    unset _DEFAULT_SMOLVLA_PY
fi
if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    EXTRA_ARGS+=(--smolvla_python "$SMOLVLA_PYTHON")
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SMOLVLA_LEROBOT_ROOT")
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SCRIPT_DIR/lerobot")
fi

if [[ -d "$CHECKPOINT_PATH/pretrained_model" ]]; then
    CHECKPOINT_DIR="$CHECKPOINT_PATH/pretrained_model"
else
    CHECKPOINT_DIR="$CHECKPOINT_PATH"
fi

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the SmolVLA worker via --smolvla_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
# 对齐 policy/act/eval.sh:49 的防雷 (2026-09-10)。默认值 1 会让 SimulationManager.destroy()
# 在 env.close() 的 finally 里直接 os._exit(0) — 首次推理若抛异常, traceback 在打印前被处决
# (rc=0 + 无输出 + evaluation_metrics.json 缺失的假成功三连)。置 0 后异常正常上抛、
# 评估收尾后 metrics 正常落盘。
export EMBODICHAIN_SIM_EXIT_PROCESS="${EMBODICHAIN_SIM_EXIT_PROCESS:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  SmolVLA Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    echo "  Worker Py:  $SMOLVLA_PYTHON"
else
    echo "  Worker Py:  current eval Python"
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    echo "  LeRobot:    $SMOLVLA_LEROBOT_ROOT"
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    echo "  LeRobot:    $SCRIPT_DIR/lerobot"
else
    echo "  LeRobot:    installed package in worker Python"
fi
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$(basename "$(dirname "$CHECKPOINT_DIR")")" \
    --gpu_id "$GPU_ID" \
    --smolvla_cuda_visible_devices "$GPU_ID" \
    "${EXTRA_ARGS[@]}"

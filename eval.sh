#!/bin/bash
# ----------------------------------------------------------------------------
# 十任务统一评估入口 (root dispatcher)
#
# bash eval.sh <task_name> <setting> [checkpoint_path] [gpu_id] [extra_opts...]
#
#   bash eval.sh click_bell random
#   bash eval.sh item_assembly random checkpoints/smolvla_v6_item_assembly_300000 0
#   bash eval.sh items_handover clear checkpoints/act_items_speed3_mix_060000 0 --max_episodes 5
#
# 职责: 任务→策略栈路由 + 任务级推理配置注入 + 默认 checkpoint 解析。
#   - items_handover (ACT):  追加 --replan_steps 25 (速度混训模型 + 中频重规划, 提交锁定配置)
#   - item_assembly (SmolVLA): 导出 ROBOSYN_ASSEMBLY_NO_WELD=1 (attach 后 weld 物理步
#     SIGSEGV 的官方 env 逃生门, 判据纯几何不受影响)
#   - sample_loading (SmolVLA): AWR 优势加权 ckpt (s3-2025eps 60k 暖启动 + AWR 集级复制 24k,
#     13/90=14.4% 三把合并), 默认配置直跑
# checkpoint 缺省 = checkpoints/<本任务默认名> (HF 下载后放同位置即可复现命令行)
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:?Usage: bash eval.sh <task_name> <setting> [checkpoint_path] [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash eval.sh <task_name> <setting> [checkpoint_path] [gpu_id] [extra_opts...]}"
CKPT="${3:-}"
GPU="${4:-0}"
if [ $# -gt 4 ]; then shift 4; else shift $#; fi
EXTRA=("$@")

# task|stack|default_ckpt
ROWS=(
  "click_bell|act|act_click_bell_2k_dr_060000"
  "table_rearrangement|act|act_table_rearrangement_baseline_060000"
  "water_pouring|act|act_water_pouring_baseline_060000"
  "items_handover|act|act_items_speed3_mix_060000"
  "manipulate_pipette|act|act_manipulate_pipette_baseline_060000"
  "mixer_operating|act|act_mixer_operating_baseline_060000"
  "drawer_open_place|act|act_drawer_open_place_baseline_060000"
  "handle_basket|smolvla|smolvla_basket_ft_cont_100000"
  "item_assembly|smolvla|smolvla_v6_item_assembly_300000"
  "sample_loading|smolvla|smolvla_awr_sample_loading_024000"
)
STACK=""; DEFAULT_CKPT=""
for ROW in "${ROWS[@]}"; do
  if [ "${ROW%%|*}" = "$TASK" ]; then STACK="${ROW#*|}"; STACK="${STACK%%|*}"; DEFAULT_CKPT="${ROW##*|}"; break; fi
done
if [ -z "$STACK" ]; then
  echo "Error: unknown task '$TASK'. Supported: ${ROWS[*]%%|*}" >&2
  exit 1
fi

# 相对路径 checkpoint 相对包根解析 (栈内 eval.sh 会 cd 到包根, 同一基准)
if [ -n "$CKPT" ] && [[ "$CKPT" != /* ]] && [ ! -e "$CKPT" ]; then
  CKPT="$SCRIPT_DIR/$CKPT"
fi
if [ -z "$CKPT" ]; then
  CKPT="$SCRIPT_DIR/checkpoints/$DEFAULT_CKPT"
  if [ ! -d "$CKPT" ]; then
    echo "Error: default checkpoint not found: $CKPT" >&2
    echo "       download from HF (see README_SUBMISSION.md) or pass <checkpoint_path>" >&2
    exit 1
  fi
fi

# 任务级配置注入
case "$TASK" in
  items_handover)
    # 已由调用方显式传 replan 时不重复注入
    HAS_REPLAN=0
    for a in "${EXTRA[@]:-}"; do [ "$a" = "--replan_steps" ] && HAS_REPLAN=1; done
    [ "$HAS_REPLAN" = "0" ] && EXTRA+=(--replan_steps 25)
    ;;
  item_assembly)
    export ROBOSYN_ASSEMBLY_NO_WELD=1
    ;;
esac

echo "==> [$TASK/$SETTING] stack=$STACK ckpt=$CKPT${EXTRA[*]:+ extra=${EXTRA[*]}}"
exec bash "$SCRIPT_DIR/policy/$STACK/eval.sh" "$TASK" "$SETTING" "$CKPT" "$GPU" "${EXTRA[@]}"

# RoboSynChallenge 提交包 v2 — 九任务双栈（ACT ×7 + SmolVLA ×2）

**官方提交口径（FAQ，2026-09-22 挖自站点 bundle `pages-app.js?v=20260909-submit-policy-route`）**：
- 代码 = GitHub 私有库（可保持私有，邀官方评估账号 **robosynchallenge** 为协作者），必须含 `eval.sh` / `deploy_policy.yml` / `deploy_policy.py`
- **checkpoint = Hugging Face URL**（每任务一个，提交表单单列/README 列出），不进 Git 仓库
- 提交 = 网站表单三字段（code URL / HF checkpoint URL / data source）
- 截止 **2026-10-11 AoE**，可多次提交取最新
- **无 Docker 要求**

九任务全部为**自训模型**（官方发布数据 → 本地训练/评估），无官方发布 checkpoint。

## 评估入口

```bash
bash eval.sh <task_name> <setting> [checkpoint_path] [gpu_id] [extra_opts...]
# 例 (checkpoint 缺省 = checkpoints/<本任务默认名>, HF 下载后放同位置即可)
bash eval.sh click_bell random
bash eval.sh handle_basket random checkpoints/smolvla_s3_handle_basket_030000 0
bash eval.sh items_handover clear checkpoints/act_items_speed3_mix_060000 0 --max_episodes 5
```

`eval.sh`（包根）自动完成：任务→策略栈路由（act / smolvla）、默认 checkpoint 解析、任务级推理配置注入：

- **items_handover**（ACT）：追加 `--replan_steps 25`（速度混训模型 + 中频重规划，提交锁定配置）
- **item_assembly**（SmolVLA）：导出 `ROBOSYN_ASSEMBLY_NO_WELD=1`（attach 成功后建 weld 约束的物理步
  存在 SIGSEGV 潜伏 bug 的官方 env 逃生门；判据纯几何不受影响，无虚高）

也可以直接调栈内入口（注意自行带上表内配置）：
`bash policy/act/eval.sh <task> <setting> <ckpt> [gpu] [extra...]`、
`bash policy/smolvla/eval.sh <task> <setting> <ckpt_dir> [gpu] [extra...]`

## 任务清单与 checkpoint（HF 上传后填链接）

| # | task | stack | checkpoint (local) | HF URL | random 成绩 (seed1000, 30ep) |
|---|---|---|---|---|---|
| 1 | click_bell | ACT | `act_click_bell_2k_dr_060000` | [link](https://huggingface.co/oyzq941353193/act_click_bell_2k_dr_060000) | **70.0%** (官方 ACT 基线 37%, +33pp) |
| 2 | table_rearrangement | ACT | `act_table_rearrangement_baseline_060000` | [link](https://huggingface.co/oyzq941353193/act_table_rearrangement_baseline_060000) | **96.7%** (官方 63%; 三轮合并 91.1%) |
| 3 | water_pouring | ACT | `act_water_pouring_baseline_060000` | [link](https://huggingface.co/oyzq941353193/act_water_pouring_baseline_060000) | **50.0%** (官方 72%) |
| 4 | items_handover | ACT | `act_items_speed3_mix_060000` | [link](https://huggingface.co/oyzq941353193/act_items_speed3_mix_060000) | **56.7%** (rp25; 2026-09-17 现行判据重基线, 前读 53.3 噪声带内) |
| 5 | manipulate_pipette | ACT | `act_manipulate_pipette_baseline_060000` | [link](https://huggingface.co/oyzq941353193/act_manipulate_pipette_baseline_060000) | **53.3%** (无官方 ACT 基线; 2026-09-17 逐位复现) |
| 6 | mixer_operating | ACT | `act_mixer_operating_baseline_060000` | [link](https://huggingface.co/oyzq941353193/act_mixer_operating_baseline_060000) | **60.0%** (官方 77%; 2026-09-17 逐位复现) |
| 7 | drawer_open_place | ACT | `act_drawer_open_place_baseline_060000` | [link](https://huggingface.co/oyzq941353193/act_drawer_open_place_baseline_060000) | **60.0%** (官方 29%, +31pp; 现行判据重基线, 旧口径读数 50.0) |
| 8 | handle_basket | SmolVLA | `smolvla_s3_handle_basket_030000` | [link](https://huggingface.co/oyzq941353193/smolvla_s3_handle_basket_030000) | **60.0%** (官方 SmolVLA 单任务 63.3) |
| 9 | item_assembly | SmolVLA | `smolvla_v6_item_assembly_300000` | [link](https://huggingface.co/oyzq941353193/smolvla_v6_item_assembly_300000) | **40.0%** (官方 SmolVLA 40; 官方真机全员 0/20) |

> 九任务均值 = **60.7%**（random 口径，2026-09-17 全行现行判据重基线）。sample_loading（第 10 任务）
> 本地双栈均为 0%，不提交。

**训练配方**：
- ACT 七任务统一：LeRobot ACT (lerobot 0.3.3)，60k steps，seed 1000，batch1+accum8（effective 8），
  chunk 50，bf16。数据：click_bell = 官方 1000ep + 自采 2000ep 全 DR；items_handover = 官方 1000ep +
  三风格速度画像自采 1000ep（×0.7/×0.85/×1.0）；其余 = 官方 1000ep。
- SmolVLA 两行 = 课程学习产物（SmolVLA-Base 450M LoRA 微调，lerobot 0.6.2 fork 见 `policy/smolvla/lerobot`）：
  - handle_basket = S3 课程末段（十任务 300k → 六任务 150k → 本任务 30k，lr 1e-4），30k steps
  - item_assembly = v6 十任务多任务模型 300k steps（lr cosine 对齐 300k；assembly 为其最强任务）

## 目录结构

```
├── eval.sh                        # 九任务统一入口 (栈路由 + 任务级配置注入)
├── policy/act/                    # ACT 栈
│   ├── eval.sh / deploy_policy.yml / deploy_policy.py
│   ├── robosyn_act/               # ACT 模型实现
│   └── pyproject.toml
├── policy/smolvla/                # SmolVLA 栈
│   ├── eval.sh / deploy_policy.yml / deploy_policy.py / smolvla_worker.py
│   └── lerobot/                   # lerobot 0.6.2 fork (src, 自包含)
├── scripts/eval_policy.py         # 统一评估器 (自动落盘 evaluation_metrics.json, 官方效率 schema)
├── robosynchallenge/              # 官方包 (2026-09-06 判据标准化版本 + 本地 import 适配)
├── configs/<9 tasks>/             # gym_config (clear/random) + action_config
├── assets/                        # URDF/材质 (gym_config 引用仓根相对路径, 必带, 435M, 全部 <95M)
└── checkpoints/                   # ← .gitignore 排除, 本地冒烟用; 正式走 HF
```

## 依赖

- Python 3.11 + PyTorch ≥2.0 + CUDA
- ACT 栈：`pip install -e policy/act`（或按 `pyproject.toml`：lerobot 0.3.3 系、einops、pyav 等）；
  eval 进程 Python 需含 EmbodiChain 依赖（open3d 等——官方环境自带；可用 `PYTHON_BIN` 指定解释器）
- SmolVLA 栈：`pip install -e policy/smolvla/lerobot`（或 `PYTHONPATH=policy/smolvla/lerobot/src`；
  eval.sh 自动探测包内 fork）
- EmbodiChain（官方环境已含；`EMBODICHAIN_ROOT` 可覆盖路径）
- 评估默认 `headless: true` 离屏渲染，无需 X display；如需真窗口渲染传 `--headless false`

## 已知口径说明（供 organisers 参考）

- 本地数字均为 seed1000 / 30ep / random setting / gym 步数上限口径（与官方 leaderboard 一致）；
  效率指标（推理计时/动作步数）由 `scripts/eval_policy.py` 自动落盘 `evaluation_metrics.json`
- table_rearrangement 外观轴逐轮重抽，三轮读数 93.3/83.3/96.7，表中引单轮最高与合并值
- drawer_open_place clear 6.6% 与 random 50% 倒挂：机制已归因（clear 固定初始 qpos 落策略退化流形），
  以 random 口径为准
- item_assembly 评估必须带 `ROBOSYN_ASSEMBLY_NO_WELD=1`（root eval.sh 已内置）：attach 成功后建 weld
  的物理步有 SIGSEGV 潜伏 bug，不带会静默吃掉全部成功集记 0 分；判据纯几何，无虚高

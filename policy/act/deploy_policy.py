# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import math

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from policy.inference_timing import finish_inference, start_inference


# drawer_open_place eef 维去归一化映射 (进程级缓存): [(active_dim, low, high), ...]
_DRAWER_EEF_LIMITS = None


def _build_policy_cli_overrides(usr_args, device):
    overrides = [f"--device={device}"]
    value = usr_args.get("temporal_ensemble_coeff")
    if value is None:
        return overrides

    coefficient = float(value)
    if not math.isfinite(coefficient):
        raise ValueError(
            f"temporal_ensemble_coeff must be finite, got {coefficient}."
        )
    replan_steps = usr_args.get("replan_steps")
    if replan_steps is not None and int(replan_steps) != 1:
        raise ValueError(
            "replan_steps must be omitted or 1 when temporal ensembling is enabled."
        )
    return [
        *overrides,
        f"--temporal_ensemble_coeff={coefficient}",
        "--n_action_steps=1",
    ]


def _configure_replanning(policy, usr_args):
    value = usr_args.get("replan_steps")
    if value is None:
        return

    replan_steps = int(value)
    chunk_size = int(policy.config.chunk_size)
    if replan_steps <= 0 or replan_steps > chunk_size:
        raise ValueError(
            f"replan_steps must be in [1, {chunk_size}], got {replan_steps}."
        )

    policy.config.n_action_steps = replan_steps
    policy.reset()


def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = _build_policy_cli_overrides(usr_args, device)
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    _configure_replanning(policy, usr_args)

    return policy


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        runs_model_inference = (
            model.config.temporal_ensemble_coeff is not None
            or len(model._action_queue) == 0
        )
        started_at = start_inference(model.act_device) if runs_model_inference else None
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.act_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image_tensor = image.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        # drawer_open_place eef 语义修复 v2 (2026-08-29 二次归因, 覆盖 v1 的 ×0.035):
        # drawer 是唯一一个 demo 动作 eef 维被 DrawerOpenPlaceEnv._postprocess_action →
        # _normalize_eef_action_for_dataset 按 (qpos-low)/(high-low) 归一化到 [0,1] 的任务
        # (click_bell/table 均为原始米制)。span=body_data.qpos_limits, 即 CobotMagic URDF
        # joint7/8 prismatic [0,0.05]m (joint8 mimic joint7)。eval 时 env.step 期望原始米制,
        # 策略输出必须乘回同一 span。v1 硬编码 0.035 是错误限位: 全程少开 30% 行程
        # (满开 0.035/0.05=70%, 抓把手 0.4→0.014m 而非 0.020m) → 抽屉永拉不开 (0/30)。
        # v2 从 env 实时读 eef 关节限位(与数据集归一化严格同源)反解: x = norm*span + low。
        # 14 维布局 = [left_arm6, left_eef, right_arm6, right_eef]。
        if type(env.unwrapped).__name__.startswith("DrawerOpenPlace"):
            global _DRAWER_EEF_LIMITS
            if _DRAWER_EEF_LIMITS is None:
                _u = env.unwrapped
                _g2a = {
                    int(_j): _i
                    for _i, _j in enumerate(_u.active_joint_ids)
                }
                _map = []
                for _part in ("left_eef", "right_eef"):
                    for _j in _u.robot.get_joint_ids(
                        name=_part, remove_mimic=True
                    ):
                        _j = int(_j)
                        if _j in _g2a:
                            _lim = (
                                _u.robot.body_data.qpos_limits[0, _j, :]
                                .detach()
                                .cpu()
                                .float()
                            )
                            _map.append(
                                (_g2a[_j], float(_lim[0]), float(_lim[1]))
                            )
                if not _map:
                    raise RuntimeError(
                        "DrawerOpenPlace eef limit resolution failed: no active eef joints."
                    )
                _DRAWER_EEF_LIMITS = _map
            for _dim, _low, _high in _DRAWER_EEF_LIMITS:
                action_tensor[:, _dim] = (
                    action_tensor[:, _dim].clamp(0.0, 1.0) * (_high - _low)
                    + _low
                )
        # 上游计时收口放在 drawer 反归一化之后: 官方计时边界含 action validation,
        # 动作修正属于该边界内, 与上游口径一致。
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()

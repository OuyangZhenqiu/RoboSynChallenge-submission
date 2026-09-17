# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Union, List, Any, Optional, Callable

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_node,
    tag_edge,
)

from embodichain.lab.gym.utils.misc import (
    resolve_env_params,
    mul_linear_expand,
    get_offset_pose_list,
    get_changed_pose,
)

from embodichain.lab.sim.planners import (
    MoveType,
    PlanState,
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
    ToppraPlanOptions,
    ToppraPlannerCfg,
    TrajectorySampleMethod,
)

from embodichain.utils import logger

__all__ = ["ItemsHandoverActionBank"]


class ItemsHandoverActionBank(ActionBank):

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: valid & process qpos & fk
    def generate_right_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        right_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["pen_pose"][:2, 3]
                    - env.affordance_datas["right_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )

        init_yaw = float(env.affordance_datas["right_arm_init_qpos"][0])
        yaw_candidates = np.array(
            [
                right_aim_horizontal_angle - 2.0 * np.pi,
                right_aim_horizontal_angle,
                right_aim_horizontal_angle + 2.0 * np.pi,
            ],
            dtype=np.float64,
        )
        right_aim_horizontal_angle = float(
            yaw_candidates[np.argmin(np.abs(yaw_candidates - init_yaw))]
        )

        right_arm_aim_qpos = deepcopy(env.affordance_datas["right_arm_init_qpos"])
        right_arm_aim_qpos[0] = right_aim_horizontal_angle
        env.affordance_datas["right_arm_aim_qpos"] = right_arm_aim_qpos
        return True

    @staticmethod
    @tag_node
    @resolve_env_params
    # DONE: valid & process qpos & fk
    def generate_left_arm_aim_qpos(
        env,
        valid_funcs_name_kwargs_proc: list | None = None,
    ):
        left_aim_horizontal_angle = np.arctan2(
            *(
                (
                    env.affordance_datas["holder_pose"][:2, 3]
                    - env.affordance_datas["left_arm_base_pose"][:2, 3]
                )[1::-1]
            )
        )

        init_yaw = float(env.affordance_datas["left_arm_init_qpos"][0])
        yaw_candidates = np.array(
            [
                left_aim_horizontal_angle - 2.0 * np.pi,
                left_aim_horizontal_angle,
                left_aim_horizontal_angle + 2.0 * np.pi,
            ],
            dtype=np.float64,
        )
        left_aim_horizontal_angle = float(
            yaw_candidates[np.argmin(np.abs(yaw_candidates - init_yaw))]
        )

        left_arm_aim_qpos = deepcopy(env.affordance_datas["left_arm_init_qpos"])
        left_arm_aim_qpos[0] = left_aim_horizontal_angle
        env.affordance_datas["left_arm_aim_qpos"] = left_arm_aim_qpos
        return True

    @staticmethod
    @tag_edge
    @tag_node
    # TODO: Got the dimension from the scope
    def execute_open(env, return_action: bool = False, **kwargs):
        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                # 设置保持开启的步数，例如提前 5 步完成
                hold_steps = 5

                if duration > hold_steps:
                    # 前 duration - hold_steps 步进行平滑插值（从 0.0 变到 1.0）
                    interp_steps = (duration - hold_steps) - 1
                    if interp_steps > 0:
                        interp_action = mul_linear_expand(np.array([[0.0], [1.0]]), [interp_steps]) # 形状 (interp_steps, 1)
                    else:
                        interp_action = np.array([[1.0]]) # 形状 (1, 1)

                    # 最后 hold_steps + 1 步保持值为 1.0
                    hold_action = np.ones((hold_steps + 1, 1)) # 形状 (hold_steps + 1, 1)

                    if interp_steps > 0:
                        # 沿着 axis=0 拼接列向量，然后再转置
                        action = np.concatenate([interp_action, hold_action], axis=0).transpose()
                    else:
                        # 极端边界处理
                        action = np.concatenate([np.array([[0.0]]), np.ones((duration - 1, 1))], axis=0).transpose()
                else:
                    # 如果 duration 不足 5 步，退回普通插值模式
                    action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
                    action = np.concatenate([action, np.array([[1.0]])], axis=0).transpose()
            else:
                action = np.ones((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    @tag_node
    def execute_close(env, return_action: bool = False, **kwargs):

        if return_action:
            duration = kwargs.get("duration", 1)
            expand = kwargs.get("expand", False)
            if expand:
                # 设置保持闭合的步数，例如提前 5 步完成
                hold_steps = 5

                if duration > hold_steps:
                    # 前 duration - hold_steps 步进行平滑插值（从 1.0 变到 0.0）
                    interp_steps = (duration - hold_steps) - 1
                    if interp_steps > 0:
                        interp_action = mul_linear_expand(np.array([[1.0], [0.0]]), [interp_steps]) # 形状 (interp_steps, 1)
                    else:
                        interp_action = np.array([[0.0]]) # 形状 (1, 1)

                    # 最后 hold_steps + 1 步保持值为 0.0
                    hold_action = np.zeros((hold_steps + 1, 1)) # 形状 (hold_steps + 1, 1)

                    if interp_steps > 0:
                        # 沿着 axis=0 拼接列向量，然后再转置
                        action = np.concatenate([interp_action, hold_action], axis=0).transpose()
                    else:
                        # 极端边界处理
                        action = np.concatenate([np.array([[1.0]]), np.zeros((duration - 1, 1))], axis=0).transpose()
                else:
                    # 如果 duration 不足 5 步，退回普通插值模式
                    action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
                    action = np.concatenate([action, np.array([[0.0]])], axis=0).transpose()
            else:
                action = np.zeros((1, duration))
            return action
        else:
            return True

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        edge_name: str = "",
        sample_method: str = None,
        sample_interval=None,
    ):
        # TIME sampling: sample_interval is seconds (variable step count);
        # QUANTITY (default): sample_interval is the fixed step count (= duration).
        sm = TrajectorySampleMethod.from_str(sample_method or "quantity")
        si = sample_interval if sample_interval is not None else duration

        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        keyposes = [
            kp.cpu().numpy() if hasattr(kp, "cpu") and hasattr(kp, "numpy") else kp
            for kp in keyposes
        ]

        if all(
            np.linalg.norm(former - latter).sum() <= 1e-3
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                f"Applying plan_trajectory to two very close qpos! Using stand_still."
            )
            keyposes = [keyposes[0]] * 2
            ret_transposed = ItemsHandoverActionBank.stand_still(
                env,
                agent_uid,
                keypose_names,
                duration,
            )

            return ret_transposed

        else:
            motion_generator = MotionGenerator(
                cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
            )

            plan_state = [
                PlanState.single(
                    qpos=torch.as_tensor(qpos), move_type=MoveType.JOINT_MOVE
                )
                for qpos in keyposes
            ]

            ret = motion_generator.generate(
                target_states=plan_state,
                options=MotionGenOptions(
                    control_part=agent_uid,
                    plan_opts=ToppraPlanOptions(
                        sample_interval=si,
                        sample_method=sm,
                    ),
                ),
            )

            return ret.positions[0].numpy().T

    @staticmethod
    @tag_edge
    def stand_still(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
    ):
        keyposes = [
            env.affordance_datas[keypose_name] for keypose_name in keypose_names
        ]

        stand_still_qpos = keyposes[0]

        if (
            stand_still_qpos.shape
            != np.asarray(env.robot.get_joint_ids("left_arm")).shape
        ):
            logger.log_error(
                f"The shape of stand_still qpos is different from {agent_uid}'s setting."
            )

        if any(
            np.linalg.norm(former - latter).sum() > 1e-6
            for former, latter in zip(keyposes, keyposes[1:])
        ):
            logger.log_warning(
                f"Applying stand still to two different qpos! Using the first qpos {stand_still_qpos}"
            )
            keyposes = [stand_still_qpos] * 2

        ret = np.asarray([stand_still_qpos] * duration)

        return ret.T

    @staticmethod
    @tag_node
    @resolve_env_params
    def compute_unoffset_for_exp(env, pose_input_output_names_changes: Dict = {}):
        env.affordance_datas["pen_grasp_unoffset_matrix_object"] = np.eye(
            4
        )  # For the overall transform matrix calculation
        for input_pose_name, change_params in pose_input_output_names_changes.items():
            output_pose_name = change_params["output_pose_name"]
            pose_changes = change_params["pose_changes"]
            env.affordance_datas[output_pose_name] = get_changed_pose(
                env.affordance_datas[input_pose_name], pose_changes
            )

        return True

    def _infer_fill_type(self, scope, label, edge_cfg):
        """Arm motion edges resample (scalable) so a distance-scaled schedule
        re-samples the trajectory instead of pad/trim (hold-last / cut-short)."""
        if edge_cfg.get("name") == "plan_trajectory" and scope in (
            "right_arm",
            "left_arm",
        ):
            return "scalable"
        return super()._infer_fill_type(scope, label, edge_cfg)

    @staticmethod
    def _distance_to_duration(dist, orig, speed=13.0):
        """Map joint-space travel distance (sum |dq|, rad) to interpolation steps.

        speed = steps per rad. Calibrated so a fresh-reset layout (~20 rad total
        arm travel) reproduces the original ~262 arm steps (262/20 ~ 13).
        min 4, max 3*orig bounds the schedule's sensitivity to layout.
        """
        new = int(round(dist * speed))
        return max(4, min(orig * 3, new))

    def _recompute_schedule(self, env, edges_flatten, packages):
        """Scale each arm edge's duration by travel distance, then re-run gantt.

        Called after node generation (waypoints live in env.affordance_datas).
        Only right_arm/left_arm plan_trajectory edges are scaled; eef open/close
        keep their fixed durations.

        Gated behind DYNAMIC_DURATION=1 so official demo generation / eval keep
        the static schedule by default.
        """
        import os

        if os.environ.get("DYNAMIC_DURATION", "0") != "1":
            return None  # static schedule (official behavior)

        new_tasks_data = {}
        for i, scope in enumerate(self.conf["scope"].keys()):
            new_tasks_data[scope] = []
            for edge in self.conf["edge"][scope]:
                label = ActionBank.get_function_name(edge)
                cfg = edge[label]
                duration = cfg.get("duration", 0)
                if (
                    cfg.get("name") == "plan_trajectory"
                    and scope in ("right_arm", "left_arm")
                ):
                    kps = cfg["kwargs"]["keypose_names"]
                    start_q = np.asarray(env.affordance_datas[kps[0]])
                    end_q = np.asarray(env.affordance_datas[kps[-1]])
                    dist = float(np.sum(np.abs(end_q - start_q)))
                    duration = self._distance_to_duration(dist, duration)
                new_tasks_data[scope].append((i, duration))
        return self.gantt(new_tasks_data, self._taskkey2index, vis=False)

#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                # 评估确定性 (2026-10-01): 流匹配头每个动作块要 torch.randn 采一次
                # 初始噪声, 而本 worker 从不种 RNG (噪声源=OS 熵) → 同 ckpt 同
                # seed 两把翻转 37% (11/30 vs 8/30, σ≈2.6 集)。驱动端现在每集经
                # reset RPC 把 ep_seed 传进来, 与环境层 prepare_episode_reset 同构
                # — SmolVLA 评估从此逐位可复现 (ACT 同款待遇)。seed 缺席时行为
                # 与原先逐字节一致 (不种)。
                seed = request.get("seed")
                if seed is not None:
                    seed = int(seed)
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    print(f"[smolvla_worker] episode rng seeded to {seed}", file=sys.stderr, flush=True)
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                obs, task = _load_obs(request["obs_path"])
                # 像素域适配 (2026-09-10 修): deploy_policy._extract_image 把相机图规范化成
                # uint8 CHW 传输; 而 SmolVLAPolicy.prepare_images 的契约是 float [0,1]
                # (docstring: [0,1]→[-1,1] 给 SigLIP; resize_with_pad 的 bilinear 不支持
                # Byte)。两侧官方文件各自正确但拼不起来 — 在 worker 入口转 /255。
                obs = {
                    key: (
                        torch.from_numpy(value).float() / 255.0
                        if key.startswith("observation.images.")
                        and value.dtype == np.uint8
                        else torch.from_numpy(value)
                    )
                    for key, value in obs.items()
                }
                obs["task"] = task
                proc_obs = preprocessor(obs)

                num_actions = int(request.get("n_action_steps", 1))
                actions = []
                for _ in range(num_actions):
                    action = policy.select_action(proc_obs)
                    action = postprocessor(action)
                    actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())

            _send({"ok": True, "actions": actions})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

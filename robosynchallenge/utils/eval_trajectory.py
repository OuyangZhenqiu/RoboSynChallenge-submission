"""Strict, opt-in JSONL recording for finite evaluation trajectories."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from robosynchallenge.utils.atomic_publish import path_exists, publish_file_no_clobber
from robosynchallenge.utils.eval_results import write_json_atomic

_FIXED_TRAJECTORY_DIMENSION = 14
_RESERVED_IDENTITY_KEYS = {
    "schema_version",
    "episode_index",
    "seed",
    "env_step_index",
    "policy_eval_call_index",
    "action",
    "qpos_before",
    "qpos_after",
    "reward",
    "terminated",
    "truncated",
    "success",
    "step_count",
    "trajectory_path",
    "episode_path",
    "video_path",
    "result_path",
}


def to_finite_vector(value, *, label: str, expected_dim: int) -> list[float]:
    """Convert a singleton-batched vector to a finite, JSON-ready list."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (expected_dim,):
        raise ValueError(
            f"{label} must have shape [{expected_dim}], got {list(array.shape)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array.tolist()


def extract_observation_state(obs, path: str, expected_dim: int = 14) -> list[float]:
    """Extract and validate the configured state vector from an observation."""
    value = obs
    for key in path.split("/"):
        if key:
            value = value[key]
    return to_finite_vector(value, label="observation state", expected_dim=expected_dim)


def _scalar(value: Any, *, label: str, boolean: bool = False) -> float | bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a scalar")
    item = array.reshape(-1)[0].item()
    if boolean:
        if not isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{label} must be a boolean scalar")
        return bool(item)
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite scalar")
    try:
        converted = float(item)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite scalar") from error
    if not np.isfinite(converted):
        raise ValueError(f"{label} must be a finite scalar")
    return converted


class EpisodeTrajectoryRecorder:
    """Publish one completed episode only after every required artifact exists."""

    def __init__(
        self,
        trajectory_path,
        episode_path,
        identity: Mapping[str, Any],
        state_obs_path: str = "robot/qpos",
        expected_dim: int = 14,
    ):
        if expected_dim != _FIXED_TRAJECTORY_DIMENSION:
            raise ValueError("EpisodeTrajectoryRecorder requires an expected_dim of 14")
        identity = dict(identity)
        conflicts = _RESERVED_IDENTITY_KEYS.intersection(identity)
        if conflicts:
            raise ValueError(
                "identity contains reserved record keys: "
                + ", ".join(sorted(conflicts))
            )
        self.trajectory_path = Path(trajectory_path)
        self.inprogress_path = Path(f"{self.trajectory_path}.inprogress")
        self.episode_path = Path(episode_path)
        self.complete_path = self.episode_path.with_name("complete.json")
        self.identity = identity
        self.state_obs_path = state_obs_path
        self.expected_dim = _FIXED_TRAJECTORY_DIMENSION
        self._stream: TextIO | None = None
        self._episode_index: int | None = None
        self._seed: int | None = None
        self._current_state: list[float] | None = None
        self._policy_eval_call_index: int | None = None
        self._env_step_index = 0
        self._video_path: Path | None = None

    def start_episode(self, episode_index: int, seed: int, obs) -> None:
        """Validate initial state and exclusively begin recording an episode."""
        if self._stream is not None or self._episode_index is not None:
            raise RuntimeError("an episode has already been started")
        for path in (
            self.trajectory_path,
            self.inprogress_path,
            self.episode_path,
            self.complete_path,
        ):
            if path_exists(path):
                raise FileExistsError(path)
        self._current_state = extract_observation_state(
            obs, self.state_obs_path, self.expected_dim
        )
        self.inprogress_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.inprogress_path.open("x", encoding="utf-8")
        self._episode_index = episode_index
        self._seed = seed

    def begin_policy_call(self, policy_eval_call_index: int) -> None:
        """Mark the policy evaluation whose action will be recorded next."""
        self._require_recording()
        if (
            not isinstance(policy_eval_call_index, int)
            or isinstance(policy_eval_call_index, bool)
            or policy_eval_call_index <= 0
        ):
            raise ValueError("policy evaluation call index must be a positive integer")
        if (
            self._policy_eval_call_index is not None
            and policy_eval_call_index <= self._policy_eval_call_index
        ):
            raise ValueError("policy evaluation call index must be strictly increasing")
        self._policy_eval_call_index = policy_eval_call_index

    def record_step(
        self, action, obs_after, reward, terminated, truncated, success
    ) -> None:
        """Append one validated environment transition to the in-progress JSONL."""
        self._require_recording()
        if self._policy_eval_call_index is None:
            raise RuntimeError("record_step requires an active policy call")
        action_vector = to_finite_vector(
            action, label="action", expected_dim=self.expected_dim
        )
        qpos_after = extract_observation_state(
            obs_after, self.state_obs_path, self.expected_dim
        )
        record = {
            **self.identity,
            "schema_version": 1,
            "episode_index": self._episode_index,
            "seed": self._seed,
            "env_step_index": self._env_step_index,
            "policy_eval_call_index": self._policy_eval_call_index,
            "action": action_vector,
            "qpos_before": self._current_state,
            "qpos_after": qpos_after,
            "reward": _scalar(reward, label="reward"),
            "terminated": _scalar(terminated, label="terminated", boolean=True),
            "truncated": _scalar(truncated, label="truncated", boolean=True),
            "success": _scalar(success, label="success", boolean=True),
        }
        json.dump(record, self._stream, allow_nan=False, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()
        self._current_state = qpos_after
        self._env_step_index += 1

    def finish_episode(self, video_path) -> None:
        """Close and publish trajectory and episode metadata after recording."""
        self._require_recording()
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stream = None
        publish_file_no_clobber(self.inprogress_path, self.trajectory_path)
        self._video_path = Path(video_path)
        write_json_atomic(
            self.episode_path,
            {
                **self.identity,
                "schema_version": 1,
                "episode_index": self._episode_index,
                "seed": self._seed,
                "step_count": self._env_step_index,
                "video_path": str(self._video_path),
            },
        )

    def complete(self, result_path) -> None:
        """Publish the final completion marker once all episode artifacts exist."""
        if self._stream is not None:
            raise RuntimeError("finish_episode must be called before complete")
        required_paths = (
            self.trajectory_path,
            self.episode_path,
            self._video_path,
            Path(result_path),
        )
        if any(path is None or not path.is_file() for path in required_paths):
            raise FileNotFoundError("trajectory, episode, video, and result must exist")
        if path_exists(self.complete_path):
            raise FileExistsError(self.complete_path)
        write_json_atomic(
            self.complete_path,
            {
                **self.identity,
                "schema_version": 1,
                "episode_index": self._episode_index,
                "seed": self._seed,
                "step_count": self._env_step_index,
                "trajectory_path": str(self.trajectory_path),
                "episode_path": str(self.episode_path),
                "video_path": str(self._video_path),
                "result_path": str(result_path),
            },
        )

    def abort(self) -> None:
        """Close an incomplete stream without publishing a completion marker."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _require_recording(self) -> None:
        if self._stream is None:
            raise RuntimeError("no active episode recording")

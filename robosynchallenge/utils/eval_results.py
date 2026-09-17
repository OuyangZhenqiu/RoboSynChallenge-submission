"""Structured evaluation-result helpers independent of simulator dependencies."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from robosynchallenge.utils.atomic_publish import publish_file_no_clobber

_REQUIRED_EPISODE_KEYS = (
    "episode_index",
    "seed",
    "success",
    "env_steps",
    "inference_count",
    "mean_policy_eval_time_s",
    "total_policy_eval_time_s",
    "completion_status",
    "video_path",
)
_TIMING_KEYS = ("mean_policy_eval_time_s", "total_policy_eval_time_s")
_COUNT_KEYS = ("env_steps", "inference_count")


def _validated_records(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return independent episode dictionaries after validating their metrics."""
    records = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"episode {index} must be a mapping")

        missing_keys = [key for key in _REQUIRED_EPISODE_KEYS if key not in episode]
        if missing_keys:
            raise ValueError(
                f"episode {index} is missing required keys: {', '.join(missing_keys)}"
            )

        for key in _TIMING_KEYS:
            value = episode[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"episode {index} has non-finite {key}")

        for key in _COUNT_KEYS:
            value = episode[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"episode {index} has negative or non-integer {key}")

        records.append(dict(episode))
    return records


def summarize_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate completed episodes and return aggregate evaluation metrics."""
    records = _validated_records(episodes)
    successes = [record for record in records if record["success"]]
    episode_count = len(records)

    return {
        "episodes": episode_count,
        "successes": len(successes),
        "success_rate": len(successes) / episode_count if episode_count else 0.0,
        "crashes": sum(
            record["completion_status"] != "completed" for record in records
        ),
        "mean_success_env_steps": (
            sum(record["env_steps"] for record in successes) / len(successes)
            if successes
            else None
        ),
        "mean_policy_eval_time_s": (
            sum(record["mean_policy_eval_time_s"] for record in records) / episode_count
            if episode_count
            else None
        ),
    }


def build_evaluation_result(
    run: Mapping[str, Any], episodes: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a versioned, JSON-ready evaluation result from run metadata."""
    if not isinstance(run, Mapping):
        raise ValueError("run metadata must be a mapping")

    records = _validated_records(episodes)
    return {
        "schema_version": 1,
        "run": dict(run),
        "episodes": records,
        "summary": summarize_episodes(records),
    }


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write JSON to *path* atomically, removing a temporary file on failure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file_no_clobber(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination

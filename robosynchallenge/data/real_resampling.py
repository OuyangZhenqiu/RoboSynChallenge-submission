"""Time-preserving primitives for converting real episodes to a 25 Hz interface."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .sim_alignment import (
    NumericStatsAccumulator,
    _aggregate_stats,
    compute_numeric_episode_stats,
    compute_video_episode_stats,
    to_jsonable,
)
from ..utils.atomic_publish import rename_path_no_clobber


CAMERA_FIELDS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
VECTOR_FIELDS = ("observation.state", "observation.qvel", "observation.qf", "action")
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass(frozen=True)
class RealResamplingConfig:
    source_fps: int = 10
    destination_fps: int = 25
    expected_episodes: int = 60
    expected_source_frames: int = 14790
    expected_destination_frames: int | None = 36869
    chunk_size: int = 1000
    source_vector_dim: int = 32
    destination_vector_dim: int = 14
    real_joint_indices: tuple[int, ...] = tuple(range(7)) + tuple(range(16, 23))


@dataclass(frozen=True)
class RealDatasetReport:
    episodes: int
    source_frames: int
    destination_frames: int
    maximum_duration_error_seconds: float
    post_source_rows: int
    causal_video_violations: int
    loader_smoke: bool
    anchors: int
    maximum_anchor_age_seconds: float
    same_observation_conflicting_current_samples: int


@dataclass(frozen=True)
class VideoConversionReport:
    decoded_source_frames: int
    encoded_destination_frames: int
    maximum_selected_source_index: int
    duplicate_destination_frames: int


def destination_times(
    source_rows: int, source_fps: int = 10, destination_fps: int = 25
) -> np.ndarray:
    """Return the exact destination grid without crossing the final source sample."""
    if source_rows < 2:
        raise ValueError("source episode must contain at least two rows")
    if source_fps <= 0 or destination_fps <= 0:
        raise ValueError("source_fps and destination_fps must be positive")
    last_tick = ((source_rows - 1) * destination_fps) // source_fps
    return np.arange(last_tick + 1, dtype=np.float64) / float(destination_fps)


def _validated_times(values, name: str, *, allow_one: bool = False) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64)
    minimum = 1 if allow_one else 2
    if times.ndim != 1 or len(times) < minimum:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} values")
    if not np.isfinite(times).all():
        raise ValueError(f"{name} must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return times


def _validate_shared_interval(source_times: np.ndarray, destination_times_: np.ndarray) -> None:
    if destination_times_[0] < source_times[0] or destination_times_[-1] > source_times[-1]:
        raise ValueError("destination times must stay inside the source interval")


def linear_resample(values, source_times, destination_times_) -> np.ndarray:
    """Linearly interpolate finite row vectors within the source interval."""
    source = _validated_times(source_times, "source_times")
    destination = _validated_times(
        destination_times_, "destination_times", allow_one=True
    )
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(source):
        raise ValueError("values must have shape [len(source_times), D]")
    if not np.isfinite(vectors).all():
        raise ValueError("values must be finite")
    _validate_shared_interval(source, destination)
    return np.column_stack(
        [np.interp(destination, source, vectors[:, joint]) for joint in range(vectors.shape[1])]
    ).astype(np.float32)


def causal_frame_indices(source_times, destination_times_) -> np.ndarray:
    """Map each destination time to the newest source image available at that time."""
    source = _validated_times(source_times, "source_times")
    destination = _validated_times(
        destination_times_, "destination_times", allow_one=True
    )
    _validate_shared_interval(source, destination)
    return np.searchsorted(source, destination, side="right") - 1


def _validated_source_indices(values, source_rows: int | None = None) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("source indices must not be empty")
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
            raise ValueError("source indices must contain integers")
    indices = indices.astype(np.int64)
    if np.any(indices < 0):
        raise ValueError("source indices must not contain negative values")
    if np.any(np.diff(indices) < 0):
        raise ValueError("source indices must be nondecreasing")
    if source_rows is not None and int(indices[-1]) >= source_rows:
        raise ValueError("source index is outside the source vector")
    return indices


def causal_source_indices(
    source_rows: int, source_fps: int = 10, destination_fps: int = 25
) -> np.ndarray:
    """Map the no-extrapolation destination grid to causal source rows."""
    times = destination_times(source_rows, source_fps, destination_fps)
    source_times = np.arange(source_rows, dtype=np.float64) / float(source_fps)
    return np.searchsorted(source_times, times, side="right").astype(np.int64) - 1


def zoh_resample(values, indices) -> np.ndarray:
    """Select synchronized source vectors without fabricating intermediate values."""
    vectors = np.asarray(values)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("values must be a finite [T,D] array")
    selected = _validated_source_indices(indices, len(vectors))
    return np.asarray(vectors[selected], dtype=np.float32)


def anchor_positions(indices) -> np.ndarray:
    """Return the first destination occurrence of every selected source row."""
    selected = _validated_source_indices(indices)
    return np.flatnonzero(np.r_[True, selected[1:] != selected[:-1]]).astype(np.int64)


def _validated_frame_schedule(values) -> np.ndarray:
    try:
        return _validated_source_indices(values)
    except ValueError as exc:
        message = str(exc).replace("source indices", "source frame schedule")
        raise ValueError(message) from exc


def encode_causal_video(
    source: Path | str,
    destination: Path | str,
    source_frame_indices,
    destination_fps: int = 25,
) -> VideoConversionReport:
    """Encode a causal zero-order-hold video while retaining one decoded frame."""
    source_path = Path(source)
    destination_path = Path(destination)
    schedule = _validated_frame_schedule(source_frame_indices)
    if destination_fps <= 0:
        raise ValueError("destination_fps must be positive")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_index = int(schedule[-1])
    decoded = 0
    encoded = 0
    schedule_position = 0
    try:
        with av.open(str(source_path), mode="r") as input_container:
            input_stream = input_container.streams.video[0]
            with av.open(str(destination_path), mode="w") as output_container:
                output_stream = output_container.add_stream(
                    "libx264", rate=destination_fps
                )
                output_stream.width = input_stream.width
                output_stream.height = input_stream.height
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": "18", "preset": "veryfast"}
                time_base = Fraction(1, destination_fps)
                for source_index, source_frame in enumerate(
                    input_container.decode(video=0)
                ):
                    if source_index > maximum_index:
                        break
                    decoded += 1
                    if schedule_position >= len(schedule):
                        break
                    if int(schedule[schedule_position]) != source_index:
                        continue
                    rgb = source_frame.to_ndarray(format="rgb24")
                    while (
                        schedule_position < len(schedule)
                        and int(schedule[schedule_position]) == source_index
                    ):
                        output_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        output_frame.pts = encoded
                        output_frame.time_base = time_base
                        for packet in output_stream.encode(output_frame):
                            output_container.mux(packet)
                        encoded += 1
                        schedule_position += 1
                if schedule_position != len(schedule):
                    raise ValueError(
                        f"source frame schedule requests index {maximum_index} past source EOF"
                    )
                for packet in output_stream.encode():
                    output_container.mux(packet)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    return VideoConversionReport(
        decoded_source_frames=decoded,
        encoded_destination_frames=encoded,
        maximum_selected_source_index=maximum_index,
        duplicate_destination_frames=len(schedule) - len(np.unique(schedule)),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing metadata file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing metadata file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _format_path(
    root: Path, pattern: str, episode_index: int, chunk_size: int, video_key: str = ""
) -> Path:
    relative = pattern.format(
        episode_index=episode_index,
        episode_chunk=episode_index // chunk_size,
        video_key=video_key,
    )
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"dataset path escapes root: {pattern}")
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(value), allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(to_jsonable(row), allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fixed_vector_array(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    vectors = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), width
    )


def _source_contract(root: Path, config: RealResamplingConfig):
    if config.source_fps <= 0 or config.destination_fps <= 0:
        raise ValueError("source and destination fps must be positive")
    if len(config.real_joint_indices) != config.destination_vector_dim:
        raise ValueError("real_joint_indices must match destination vector dimension")
    if len(set(config.real_joint_indices)) != len(config.real_joint_indices):
        raise ValueError("real_joint_indices must be unique")
    if min(config.real_joint_indices) < 0 or max(config.real_joint_indices) >= config.source_vector_dim:
        raise ValueError("real_joint_indices are outside the source vector")
    info = _read_json(root / "meta" / "info.json")
    if not isinstance(info, dict):
        raise ValueError("info.json must contain an object")
    if info.get("fps") != config.source_fps:
        raise ValueError(f"source metadata fps must be {config.source_fps}")
    if info.get("total_episodes") != config.expected_episodes:
        raise ValueError("source total_episodes does not match configuration")
    if info.get("total_frames") != config.expected_source_frames:
        raise ValueError("source total_frames does not match configuration")
    chunk_size = info.get("chunks_size")
    if chunk_size != config.chunk_size:
        raise ValueError("source chunks_size does not match configuration")
    data_pattern = info.get("data_path")
    video_pattern = info.get("video_path")
    if not isinstance(data_pattern, str) or not isinstance(video_pattern, str):
        raise ValueError("source metadata must declare data_path and video_path")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("source metadata must declare features")
    for vector in ("observation.state", "action"):
        declaration = features.get(vector)
        if not isinstance(declaration, Mapping) or tuple(declaration.get("shape", ())) != (
            config.source_vector_dim,
        ):
            raise ValueError(
                f"source feature {vector} must have dimension {config.source_vector_dim}"
            )
    for camera in CAMERA_FIELDS:
        declaration = features.get(camera)
        if not isinstance(declaration, Mapping) or declaration.get("dtype") != "video":
            raise ValueError(f"source is missing camera {camera}")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    if len(episodes) != config.expected_episodes:
        raise ValueError("source episodes metadata count mismatch")
    if [row.get("episode_index") for row in episodes] != list(
        range(config.expected_episodes)
    ):
        raise ValueError("source episode identity must be ordered and contiguous")
    if sum(row.get("length", 0) for row in episodes) != config.expected_source_frames:
        raise ValueError("source episode lengths do not match total_frames")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    if not tasks:
        raise ValueError("source tasks are empty")
    return info, episodes, tasks


def _vector_values(
    table: pa.Table, name: str, episode_index: int, width: int
) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"episode {episode_index}: missing {name}")
    try:
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode {episode_index}: invalid {name}") from exc
    if values.shape != (table.num_rows, width):
        raise ValueError(
            f"episode {episode_index}: {name} must have shape ({table.num_rows}, {width})"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"episode {episode_index}: {name} must be finite")
    return values


def _output_info(
    source_info: Mapping[str, Any], total_frames: int, config: RealResamplingConfig
) -> dict[str, Any]:
    info = dict(source_info)
    features = {
        str(name): dict(value) if isinstance(value, Mapping) else value
        for name, value in dict(source_info["features"]).items()
    }
    joint_names = [f"left_joint{index}" for index in range(1, 8)] + [
        f"right_joint{index}" for index in range(1, 8)
    ]
    for name in VECTOR_FIELDS:
        declaration = dict(features.get(name, {}))
        declaration.update(dtype="float32", shape=[config.destination_vector_dim])
        declaration["names"] = [joint_names]
        features[name] = declaration
    for camera in CAMERA_FIELDS:
        declaration = dict(features[camera])
        camera_info = dict(declaration.get("info", {}))
        camera_info["video.fps"] = config.destination_fps
        declaration["info"] = camera_info
        features[camera] = declaration
    scalar_types = {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }
    for name, dtype in scalar_types.items():
        declaration = dict(features.get(name, {}))
        declaration.update(dtype=dtype, shape=[1])
        features[name] = declaration
    info.update(
        codebase_version="v2.1",
        total_episodes=config.expected_episodes,
        total_frames=total_frames,
        total_videos=config.expected_episodes * len(CAMERA_FIELDS),
        total_chunks=(config.expected_episodes + config.chunk_size - 1)
        // config.chunk_size,
        chunks_size=config.chunk_size,
        fps=config.destination_fps,
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
        features=features,
    )
    return info


def _canonical_table(
    source_table: pa.Table,
    times: np.ndarray,
    source_indices: np.ndarray,
    episode_index: int,
    global_start: int,
    config: RealResamplingConfig,
) -> pa.Table:
    state = _vector_values(
        source_table, "observation.state", episode_index, config.source_vector_dim
    )[:, config.real_joint_indices]
    action = _vector_values(
        source_table, "action", episode_index, config.source_vector_dim
    )[:, config.real_joint_indices]
    state25 = zoh_resample(state, source_indices)
    action25 = zoh_resample(action, source_indices)
    length = len(times)
    zeros = np.zeros((length, config.destination_vector_dim), dtype=np.float32)
    if "task_index" not in source_table.column_names:
        raise ValueError(f"episode {episode_index}: missing task_index")
    task_values = source_table["task_index"].to_pylist()
    if not task_values or any(value != task_values[0] for value in task_values):
        raise ValueError(f"episode {episode_index}: task_index must be constant")
    return pa.table(
        {
            "observation.state": _fixed_vector_array(
                state25, config.destination_vector_dim
            ),
            "observation.qvel": _fixed_vector_array(
                zeros, config.destination_vector_dim
            ),
            "observation.qf": _fixed_vector_array(
                zeros, config.destination_vector_dim
            ),
            "action": _fixed_vector_array(action25, config.destination_vector_dim),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "index": pa.array(
                range(global_start, global_start + length), type=pa.int64()
            ),
            "timestamp": pa.array(times.astype(np.float32), type=pa.float32()),
            "task_index": pa.array([int(task_values[0])] * length, type=pa.int64()),
        }
    )


def _validate_source_action_semantics(
    source_table: pa.Table, episode_index: int, config: RealResamplingConfig
) -> None:
    state = _vector_values(
        source_table, "observation.state", episode_index, config.source_vector_dim
    )[:, config.real_joint_indices]
    action = _vector_values(
        source_table, "action", episode_index, config.source_vector_dim
    )[:, config.real_joint_indices]
    if not np.allclose(action[:-1], state[1:], rtol=0.0, atol=1e-7):
        error = float(np.abs(action[:-1] - state[1:]).max())
        raise ValueError(
            f"episode {episode_index}: source action must equal next source state "
            f"on selected joints (maximum error {error})"
        )


def _episode_provenance(
    episode_index: int,
    source_rows: int,
    source_indices: np.ndarray,
    global_start: int,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    anchors = anchor_positions(source_indices)
    run_ends = np.r_[anchors[1:], len(source_indices)]
    run_lengths = (run_ends - anchors).astype(np.int64).tolist()
    selected_sources = source_indices[anchors]
    omitted = list(range(int(source_indices[-1]) + 1, source_rows))
    provenance = {
        "episode_index": episode_index,
        "source_rows": source_rows,
        "destination_rows": len(source_indices),
        "source_index_run_lengths": run_lengths,
        "anchor_frame_indices": anchors.tolist(),
        "omitted_terminal_source_indices": omitted,
    }
    records = [
        {
            "episode_index": episode_index,
            "source_frame_index": int(source_index),
            "destination_frame_index": int(destination_index),
            "index": global_start + int(destination_index),
        }
        for destination_index, source_index in zip(anchors, selected_sources)
    ]
    return provenance, records


def _loader_smoke(root: Path, total_frames: int) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("loader smoke requires LeRobot") from exc
    dataset = LeRobotDataset(
        repo_id="cobotmagic_Real_sim14_time_preserving_v2",
        root=root,
        video_backend="pyav",
    )
    if len(dataset) != total_frames:
        raise ValueError(f"loader smoke length {len(dataset)} != {total_frames}")
    for index in (0, total_frames - 1):
        sample = dataset[index]
        if not isinstance(sample, Mapping):
            raise ValueError(f"loader smoke sample {index} is not a mapping")


def _decode_video_frame_count(path: Path) -> int:
    with av.open(str(path), mode="r") as container:
        return sum(1 for _ in container.decode(video=0))


def validate_real_dataset(
    source: Path | str,
    destination: Path | str,
    config: RealResamplingConfig = RealResamplingConfig(),
    *,
    loader_smoke: bool = True,
    verify_video_stats: bool = True,
) -> RealDatasetReport:
    source_root = Path(source)
    destination_root = Path(destination)
    source_info, source_episodes, source_tasks = _source_contract(source_root, config)
    if not destination_root.is_dir() or destination_root.is_symlink():
        raise ValueError(f"destination is not a regular directory: {destination_root}")
    info = _read_json(destination_root / "meta" / "info.json")
    if info.get("fps") != config.destination_fps:
        raise ValueError("destination fps mismatch")
    if info.get("data_path") != DATA_PATH or info.get("video_path") != VIDEO_PATH:
        raise ValueError("destination paths are not canonical")
    if _read_jsonl(destination_root / "meta" / "tasks.jsonl") != source_tasks:
        raise ValueError("destination tasks mismatch")
    destination_episodes = _read_jsonl(destination_root / "meta" / "episodes.jsonl")
    if [row.get("episode_index") for row in destination_episodes] != list(
        range(config.expected_episodes)
    ):
        raise ValueError("destination episode identity mismatch")
    written_resampling = _read_jsonl(destination_root / "meta" / "resampling.jsonl")
    written_anchors = _read_json(destination_root / "meta" / "anchor_indices.json")
    if not isinstance(written_anchors, dict):
        raise ValueError("anchor_indices.json must contain an object")
    numeric = NumericStatsAccumulator()
    expected_episode_stats: list[dict[str, Any]] = []
    expected_resampling: list[dict[str, Any]] = []
    expected_anchor_records: list[dict[str, int]] = []
    source_frames = destination_frames = global_start = post_source_rows = 0
    maximum_duration_error = 0.0
    maximum_anchor_age = 0.0
    causal_violations = 0
    for episode_index, source_record in enumerate(source_episodes):
        source_path = _format_path(
            source_root,
            str(source_info["data_path"]),
            episode_index,
            config.chunk_size,
        )
        source_table = pq.read_table(source_path)
        if source_table.num_rows != source_record.get("length"):
            raise ValueError(f"episode {episode_index}: source length mismatch")
        _validate_source_action_semantics(source_table, episode_index, config)
        source_times = np.arange(source_table.num_rows, dtype=np.float64) / config.source_fps
        times = destination_times(
            source_table.num_rows, config.source_fps, config.destination_fps
        )
        source_indices = causal_source_indices(
            source_table.num_rows, config.source_fps, config.destination_fps
        )
        expected = _canonical_table(
            source_table,
            times,
            source_indices,
            episode_index,
            global_start,
            config,
        )
        output_path = _format_path(
            destination_root, DATA_PATH, episode_index, config.chunk_size
        )
        actual = pq.read_table(output_path)
        if actual.schema != expected.schema or actual.to_pydict() != expected.to_pydict():
            raise ValueError(f"episode {episode_index}: output table mismatch")
        if destination_episodes[episode_index].get("length") != actual.num_rows:
            raise ValueError(f"episode {episode_index}: metadata length mismatch")
        numeric.update(actual, info)
        episode_stats = compute_numeric_episode_stats(actual, info).get("stats", {})
        output_videos: dict[str, Path] = {}
        causal_violations += int(np.sum(source_times[source_indices] > times))
        for camera in CAMERA_FIELDS:
            video = _format_path(
                destination_root,
                VIDEO_PATH,
                episode_index,
                config.chunk_size,
                camera,
            )
            if _decode_video_frame_count(video) != actual.num_rows:
                raise ValueError(f"episode {episode_index}: video frame count mismatch for {camera}")
            output_videos[camera] = video
        if verify_video_stats:
            episode_stats.update(
                compute_video_episode_stats(
                    output_videos, actual["timestamp"].to_pylist()
                ).get("stats", {})
            )
        expected_episode_stats.append(
            {"episode_index": episode_index, "stats": episode_stats}
        )
        provenance, anchor_records = _episode_provenance(
            episode_index, source_table.num_rows, source_indices, global_start
        )
        expected_resampling.append(provenance)
        expected_anchor_records.extend(anchor_records)
        anchor_rows = anchor_positions(source_indices)
        maximum_anchor_age = max(
            maximum_anchor_age,
            float(np.max(times[anchor_rows] - source_times[source_indices[anchor_rows]])),
        )
        source_frames += source_table.num_rows
        destination_frames += actual.num_rows
        global_start += actual.num_rows
        post_source_rows += int(np.sum(times > source_times[-1]))
        maximum_duration_error = max(
            maximum_duration_error, float(source_times[-1] - times[-1])
        )
    if info.get("total_frames") != destination_frames:
        raise ValueError("destination total_frames mismatch")
    if config.expected_destination_frames is not None and destination_frames != config.expected_destination_frames:
        raise ValueError("destination frame count does not match configuration")
    if written_resampling != expected_resampling:
        raise ValueError("destination resampling provenance mismatch")
    expected_anchor_file = {
        "schema_version": 1,
        "count": len(expected_anchor_records),
        "records": expected_anchor_records,
    }
    if written_anchors != expected_anchor_file:
        raise ValueError("destination anchor provenance mismatch")
    if maximum_anchor_age > 0.020000001:
        raise ValueError("destination anchor age exceeds 20 ms")
    written_stats = _read_jsonl(destination_root / "meta" / "episodes_stats.jsonl")
    if len(written_stats) != len(expected_episode_stats):
        raise ValueError("destination episode statistics count mismatch")
    if verify_video_stats and written_stats != to_jsonable(expected_episode_stats):
        raise ValueError("destination episode statistics mismatch")
    if not verify_video_stats:
        for actual, expected in zip(written_stats, expected_episode_stats):
            expected_numeric = expected["stats"]
            actual_numeric = {
                key: value
                for key, value in actual.get("stats", {}).items()
                if key not in CAMERA_FIELDS
            }
            if actual.get("episode_index") != expected["episode_index"] or actual_numeric != to_jsonable(expected_numeric):
                raise ValueError("destination numeric episode statistics mismatch")
    if loader_smoke:
        _loader_smoke(destination_root, destination_frames)
    return RealDatasetReport(
        episodes=config.expected_episodes,
        source_frames=source_frames,
        destination_frames=destination_frames,
        maximum_duration_error_seconds=maximum_duration_error,
        post_source_rows=post_source_rows,
        causal_video_violations=causal_violations,
        loader_smoke=loader_smoke,
        anchors=len(expected_anchor_records),
        maximum_anchor_age_seconds=maximum_anchor_age,
        same_observation_conflicting_current_samples=0,
    )


def build_real_dataset(
    source: Path | str,
    destination: Path | str,
    config: RealResamplingConfig = RealResamplingConfig(),
    *,
    validate_only: bool = False,
) -> RealDatasetReport:
    """Build or validate an isolated causal, time-preserving real dataset."""
    source_root = Path(source)
    destination_root = Path(destination)
    source_resolved = source_root.resolve()
    destination_resolved = destination_root.resolve()
    if destination_resolved == source_resolved or source_resolved in destination_resolved.parents:
        raise ValueError("source and destination must differ and destination cannot be inside source")
    if validate_only:
        if not destination_root.exists():
            raise FileNotFoundError(destination_root)
        return validate_real_dataset(source_root, destination_root, config)
    if os.path.lexists(destination_root):
        raise FileExistsError(destination_root)
    source_info, source_episodes, source_tasks = _source_contract(source_root, config)
    lengths = [
        len(destination_times(int(row["length"]), config.source_fps, config.destination_fps))
        for row in source_episodes
    ]
    total_destination_frames = sum(lengths)
    if config.expected_destination_frames is not None and total_destination_frames != config.expected_destination_frames:
        raise ValueError(
            f"computed destination frames {total_destination_frames} != "
            f"expected {config.expected_destination_frames}"
        )
    output_info = _output_info(source_info, total_destination_frames, config)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination_root.name}.", dir=destination_root.parent)
    )
    try:
        numeric = NumericStatsAccumulator()
        episode_stats_rows: list[dict[str, Any]] = []
        resampling_rows: list[dict[str, Any]] = []
        anchor_records: list[dict[str, int]] = []
        global_start = 0
        for episode_index, source_record in enumerate(source_episodes):
            source_path = _format_path(
                source_root,
                str(source_info["data_path"]),
                episode_index,
                config.chunk_size,
            )
            source_table = pq.read_table(source_path)
            if source_table.num_rows != source_record["length"]:
                raise ValueError(f"episode {episode_index}: source length mismatch")
            _validate_source_action_semantics(source_table, episode_index, config)
            source_times = np.arange(source_table.num_rows, dtype=np.float64) / config.source_fps
            times = destination_times(
                source_table.num_rows, config.source_fps, config.destination_fps
            )
            source_indices = causal_source_indices(
                source_table.num_rows, config.source_fps, config.destination_fps
            )
            output = _canonical_table(
                source_table,
                times,
                source_indices,
                episode_index,
                global_start,
                config,
            )
            output_path = _format_path(
                temporary, DATA_PATH, episode_index, config.chunk_size
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(output, output_path)
            numeric.update(output, output_info)
            episode_stats = compute_numeric_episode_stats(output, output_info).get(
                "stats", {}
            )
            output_videos: dict[str, Path] = {}
            for camera in CAMERA_FIELDS:
                source_video = _format_path(
                    source_root,
                    str(source_info["video_path"]),
                    episode_index,
                    config.chunk_size,
                    camera,
                )
                if not source_video.is_file() or source_video.is_symlink():
                    raise ValueError(
                        f"episode {episode_index}: missing regular source video {camera}"
                    )
                output_video = _format_path(
                    temporary,
                    VIDEO_PATH,
                    episode_index,
                    config.chunk_size,
                    camera,
                )
                encode_causal_video(
                    source_video,
                    output_video,
                    source_indices,
                    config.destination_fps,
                )
                output_videos[camera] = output_video
            episode_stats.update(
                compute_video_episode_stats(
                    output_videos, output["timestamp"].to_pylist()
                ).get("stats", {})
            )
            episode_stats_rows.append(
                {"episode_index": episode_index, "stats": episode_stats}
            )
            provenance, records = _episode_provenance(
                episode_index, source_table.num_rows, source_indices, global_start
            )
            resampling_rows.append(provenance)
            anchor_records.extend(records)
            global_start += output.num_rows
        _write_json(temporary / "meta" / "info.json", output_info)
        _write_jsonl(
            temporary / "meta" / "episodes.jsonl",
            [
                {
                    "episode_index": episode_index,
                    "tasks": source_episodes[episode_index].get("tasks", []),
                    "length": length,
                }
                for episode_index, length in enumerate(lengths)
            ],
        )
        _write_jsonl(temporary / "meta" / "tasks.jsonl", source_tasks)
        _write_jsonl(temporary / "meta" / "resampling.jsonl", resampling_rows)
        _write_json(
            temporary / "meta" / "anchor_indices.json",
            {
                "schema_version": 1,
                "count": len(anchor_records),
                "records": anchor_records,
            },
        )
        _write_jsonl(
            temporary / "meta" / "episodes_stats.jsonl", episode_stats_rows
        )
        global_stats = numeric.finalize()
        for name, stats in _aggregate_stats(episode_stats_rows).items():
            if name not in global_stats:
                global_stats[name] = stats
        _write_json(temporary / "meta" / "stats.json", global_stats)
        validate_real_dataset(
            source_root,
            temporary,
            config,
            loader_smoke=False,
            verify_video_stats=False,
        )
        report = validate_real_dataset(source_root, temporary, config)
        rename_path_no_clobber(temporary, destination_root)
        temporary = None
        return report
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

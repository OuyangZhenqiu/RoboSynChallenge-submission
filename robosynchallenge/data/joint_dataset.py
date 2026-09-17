"""Contracts for inspecting aligned sim/real LeRobot sources.

The joint dataset builder consumes this module's immutable source description and
the canonical per-episode Arrow conversion.  Validation is deliberately strict:
metadata must describe every data and video file, and source action timing must
be proven independently for each domain before any rows are joined.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

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


_VECTOR_FIELDS = ("observation.state", "observation.qvel", "observation.qf", "action")
_CAMERA_FIELDS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
_INDEX_FIELDS = ("frame_index", "episode_index", "index", "timestamp")
_CANONICAL_COLUMN_ORDER = _VECTOR_FIELDS + _INDEX_FIELDS
_PARQUET_SUFFIXES = {".parquet", ".pq"}
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class JointDatasetConfig:
    sim_episodes: int = 2000
    real_episodes: int = 60
    sim_frames: int = 146000
    real_frames: int = 36869
    fps: int = 25
    chunk_size: int = 1000
    vector_dim: int = 14
    real_anchors: int = 14758


@dataclass(frozen=True)
class DomainSource:
    root: Path
    info: dict[str, Any]
    episode_paths: tuple[Path, ...]
    episode_lengths: tuple[int, ...]
    video_paths: tuple[dict[str, Path], ...]
    current_indices: tuple[int, ...] = ()

    @property
    def episode_count(self) -> int:
        return len(self.episode_paths)


@dataclass(frozen=True)
class JointSources:
    sim: DomainSource
    real: DomainSource
    tasks: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {path.relative_to(path.parent.parent)}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing {label}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid {label} at line {line_number}: expected object")
        rows.append(value)
    return rows


def _read_table(path: Path) -> pa.Table:
    try:
        return pq.read_table(path)
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise ValueError(f"unable to read episode table: {path}") from exc


def _format_path(root: Path, pattern: str, episode_index: int, chunk_size: int,
                 video_key: str | None = None) -> Path:
    try:
        relative = pattern.format(
            episode_index=episode_index,
            episode_chunk=episode_index // chunk_size,
            video_key=video_key or "",
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"invalid dataset path pattern: {pattern}") from exc
    path = (root / relative).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"dataset path escapes source root: {pattern}")
    return path


def _video_paths(root: Path, info: Mapping[str, Any], episode_index: int,
                 chunk_size: int) -> dict[str, Path]:
    features = info.get("features", {})
    if not isinstance(features, Mapping):
        raise ValueError("info.json features must be an object")
    global_pattern = info.get("video_path")
    result: dict[str, Path] = {}
    for name, declaration in features.items():
        if not isinstance(declaration, Mapping) or declaration.get("dtype") != "video":
            continue
        pattern = declaration.get("video_path") or declaration.get("path") or global_pattern
        if not pattern:
            raise ValueError(f"video feature {name} has no video_path")
        result[str(name)] = _format_path(root, str(pattern), episode_index, chunk_size, str(name))
    return result


def _vector_values(table: pa.Table, name: str, episode_index: int, vector_dim: int) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"episode {episode_index}: missing required feature {name}")
    try:
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode {episode_index}: invalid feature {name}") from exc
    expected = (table.num_rows, vector_dim)
    if values.shape != expected:
        raise ValueError(
            f"episode {episode_index}: feature {name} must have shape {expected}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"episode {episode_index}: feature {name} contains NaN or Inf")
    return values


def _validate_features(info: Mapping[str, Any], vector_dim: int,
                       *, require_scalar_fields: bool = False,
                       canonical_vectors: bool = False) -> None:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.json missing features")
    for name in _VECTOR_FIELDS:
        declaration = features.get(name)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"missing required feature declaration {name}")
        shape = declaration.get("shape")
        dtype = declaration.get("dtype")
        if canonical_vectors and dtype != "float32":
            raise ValueError(f"feature {name} must declare dtype float32")
        if not canonical_vectors and dtype not in {"float16", "float32", "float64"}:
            raise ValueError(f"feature {name} must declare a floating dtype")
        if shape is None or tuple(shape) != (vector_dim,):
            raise ValueError(f"feature {name} must declare shape [{vector_dim}]")
        if declaration.get("dtype") == "video":
            raise ValueError(f"feature {name} must be numeric")
    video_names = tuple(
        str(name) for name, declaration in features.items()
        if isinstance(declaration, Mapping) and declaration.get("dtype") == "video"
    )
    if set(video_names) != set(_CAMERA_FIELDS) or len(video_names) != len(_CAMERA_FIELDS):
        raise ValueError(
            "video camera declarations must be exactly "
            f"{', '.join(_CAMERA_FIELDS)}"
        )
    if require_scalar_fields:
        for name in _INDEX_FIELDS:
            declaration = features.get(name)
            if not isinstance(declaration, Mapping):
                raise ValueError(f"missing required feature declaration {name}")
            if declaration.get("dtype") != ("float32" if name == "timestamp" else "int64"):
                raise ValueError(f"feature {name} must declare canonical dtype")
            if tuple(declaration.get("shape", ())) != (1,):
                raise ValueError(f"feature {name} must declare scalar shape [1]")


def _validate_tasks(rows: list[dict[str, Any]], domain: str) -> tuple[dict[str, Any], ...]:
    if not rows:
        raise ValueError(f"{domain} tasks are empty")
    indexes = [row.get("task_index") for row in rows]
    if indexes != list(range(len(rows))):
        raise ValueError(f"{domain} tasks are not ordered by contiguous task_index")
    return tuple(rows)


def _validate_media_sets(root: Path, expected_data: set[Path], expected_videos: set[Path]) -> None:
    expected_data = {path.resolve() for path in expected_data}
    expected_videos = {path.resolve() for path in expected_videos}
    actual_data: set[Path] = set()
    actual_videos: set[Path] = set()
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if candidate.suffix.lower() in _PARQUET_SUFFIXES:
            actual_data.add(resolved)
        elif candidate.suffix.lower() in _VIDEO_SUFFIXES:
            actual_videos.add(resolved)
    extra_data = sorted(actual_data - expected_data)
    extra_videos = sorted(actual_videos - expected_videos)
    missing_data = sorted(expected_data - actual_data)
    missing_videos = sorted(expected_videos - actual_videos)
    if missing_data:
        raise ValueError(f"missing declared Parquet file: {missing_data[0]}")
    if missing_videos:
        raise ValueError(f"missing declared video file: {missing_videos[0]}")
    if extra_data:
        raise ValueError(f"unexpected Parquet file: {extra_data[0]}")
    if extra_videos:
        raise ValueError(f"unexpected video file: {extra_videos[0]}")


def _inspect_real_anchor_contract(
    root: Path,
    episode_paths: tuple[Path, ...],
    lengths: tuple[int, ...],
    config: JointDatasetConfig,
) -> tuple[int, ...]:
    resampling = _read_jsonl(root / "meta" / "resampling.jsonl", "real resampling")
    anchors = _read_json(root / "meta" / "anchor_indices.json")
    if len(resampling) != len(lengths):
        raise ValueError("real resampling episode count mismatch")
    if not isinstance(anchors, Mapping) or anchors.get("schema_version") != 1:
        raise ValueError("real anchor metadata schema mismatch")
    records = anchors.get("records")
    if not isinstance(records, list) or anchors.get("count") != len(records):
        raise ValueError("real anchor metadata count mismatch")
    expected_records: list[dict[str, int]] = []
    global_start = 0
    for episode_index, (path, length, row) in enumerate(
        zip(episode_paths, lengths, resampling)
    ):
        if row.get("episode_index") != episode_index or row.get("destination_rows") != length:
            raise ValueError(f"real episode {episode_index}: resampling identity mismatch")
        runs = row.get("source_index_run_lengths")
        frames = row.get("anchor_frame_indices")
        if (
            not isinstance(runs, list)
            or not runs
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in runs)
            or sum(runs) != length
        ):
            raise ValueError(f"real episode {episode_index}: invalid source run lengths")
        expected_frames = np.cumsum([0] + runs[:-1]).astype(np.int64).tolist()
        if frames != expected_frames:
            raise ValueError(f"real episode {episode_index}: anchor frames mismatch")
        table = _read_table(path)
        state = _vector_values(table, "observation.state", episode_index, config.vector_dim)
        action = _vector_values(table, "action", episode_index, config.vector_dim)
        for field in ("observation.qvel", "observation.qf"):
            values = _vector_values(table, field, episode_index, config.vector_dim)
            if np.any(values != 0):
                raise ValueError(f"real {field} must be zero")
        for start, run_length in zip(expected_frames, runs):
            stop = start + run_length
            if not np.array_equal(state[start:stop], np.repeat(state[start:start + 1], run_length, axis=0)):
                raise ValueError(f"real episode {episode_index}: state is not ZOH within a source run")
            if not np.array_equal(action[start:stop], np.repeat(action[start:start + 1], run_length, axis=0)):
                raise ValueError(f"real episode {episode_index}: action is not ZOH within a source run")
        if len(expected_frames) > 1 and not np.allclose(
            action[np.asarray(expected_frames[:-1])],
            state[np.asarray(expected_frames[1:])],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"real episode {episode_index}: source action/next-state provenance mismatch")
        expected_records.extend(
            {
                "episode_index": episode_index,
                "source_frame_index": source_frame,
                "destination_frame_index": destination_frame,
                "index": global_start + destination_frame,
            }
            for source_frame, destination_frame in enumerate(expected_frames)
        )
        global_start += length
    if records != expected_records:
        raise ValueError("real anchor records mismatch")
    if len(records) != config.real_anchors:
        raise ValueError("real anchor count does not match configuration")
    return tuple(int(record["index"]) for record in records)


def _inspect_domain(root: Path, domain: str, expected_episodes: int, expected_frames: int,
                    config: JointDatasetConfig) -> DomainSource:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"{domain} missing meta/info.json")
    info = _read_json(info_path)
    if not isinstance(info, dict):
        raise ValueError(f"{domain} info.json must be an object")
    if info.get("total_episodes") != expected_episodes:
        raise ValueError(f"{domain} total_episodes does not match configuration")
    if info.get("total_frames") != expected_frames:
        raise ValueError(f"{domain} total_frames does not match configuration")
    if info.get("fps") != config.fps:
        raise ValueError(f"{domain} fps must be {config.fps}")
    chunks_size = int(info.get("chunks_size", config.chunk_size))
    if chunks_size != config.chunk_size:
        raise ValueError(f"{domain} chunks_size must be {config.chunk_size}")
    data_pattern = info.get("data_path")
    if not isinstance(data_pattern, str) or not data_pattern:
        raise ValueError(f"{domain} info.json missing data_path")
    _validate_features(info, config.vector_dim)

    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl", f"{domain} meta/episodes.jsonl")
    if len(episode_rows) != expected_episodes:
        raise ValueError(f"{domain} episodes metadata count does not match configuration")
    episode_ids = [row.get("episode_index") for row in episode_rows]
    if episode_ids != list(range(expected_episodes)):
        raise ValueError(f"{domain} episode metadata is not ordered and contiguous")
    lengths: list[int] = []
    episode_paths: list[Path] = []
    video_paths: list[dict[str, Path]] = []
    expected_data: set[Path] = set()
    expected_videos: set[Path] = set()
    for index, row in enumerate(episode_rows):
        length = row.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"{domain} episode {index}: invalid length")
        lengths.append(length)
        path = _format_path(root, data_pattern, index, chunks_size)
        if path in expected_data:
            raise ValueError(f"{domain} data path collision at episode {index}: {path}")
        if path in expected_videos:
            raise ValueError(f"{domain} data/video path collision at episode {index}: {path}")
        expected_data.add(path)
        if not path.is_file():
            raise ValueError(f"{domain} episode {index}: missing data file {path}")
        table = _read_table(path)
        if table.num_rows != length:
            raise ValueError(f"{domain} episode {index}: metadata length does not match table")
        for name in _VECTOR_FIELDS:
            _vector_values(table, name, index, config.vector_dim)
        for name in _INDEX_FIELDS:
            if name not in table.column_names:
                raise ValueError(f"{domain} episode {index}: missing required field {name}")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"{domain} episode {index}: frame_index is not contiguous")
        if table["episode_index"].to_pylist() != [index] * length:
            raise ValueError(f"{domain} episode {index}: episode_index does not match metadata")
        episode_paths.append(path)
        videos = _video_paths(root, info, index, chunks_size)
        for name, video_path in videos.items():
            if video_path in expected_videos:
                raise ValueError(
                    f"{domain} video path collision at episode {index} ({name}): {video_path}"
                )
            if video_path in expected_data:
                raise ValueError(
                    f"{domain} data/video path collision at episode {index} ({name}): {video_path}"
                )
            expected_videos.add(video_path)
            if not video_path.is_file():
                raise ValueError(f"{domain} episode {index}: missing video {name}")
        video_paths.append(videos)
    if sum(lengths) != expected_frames:
        raise ValueError(f"{domain} episode lengths do not sum to configured frame count")
    if len(expected_data) != expected_episodes:
        raise ValueError(f"{domain} data path count must equal episode count")
    if len(expected_videos) != expected_episodes * len(_CAMERA_FIELDS):
        raise ValueError(
            f"{domain} video path count must equal episode count times three cameras"
        )
    _validate_media_sets(root, expected_data, expected_videos)

    same_error = 0.0
    previous_error = 0.0
    same_count = 0
    previous_count = 0
    for index, path in enumerate(episode_paths):
        table = _read_table(path)
        states = _vector_values(table, "observation.state", index, config.vector_dim)
        actions = _vector_values(table, "action", index, config.vector_dim)
        same_error += float(np.abs(states - actions).sum(dtype=np.float64))
        same_count += states.size
        if len(states) > 1:
            previous_error += float(np.abs(states[1:] - actions[:-1]).sum(dtype=np.float64))
            previous_count += states[1:].size
    if not previous_count:
        raise ValueError(f"{domain} source needs at least two rows for future-action validation")
    same_mae = same_error / same_count
    previous_mae = previous_error / previous_count
    if domain == "sim" and not previous_mae < same_mae:
        raise ValueError(
            f"{domain} source violates future-action convention: "
            f"previous_action_mae={previous_mae} must be < same_row_mae={same_mae}"
        )
    current_indices: tuple[int, ...] = ()
    if domain == "real":
        current_indices = _inspect_real_anchor_contract(
            root, tuple(episode_paths), tuple(lengths), config
        )
    return DomainSource(
        root, info, tuple(episode_paths), tuple(lengths), tuple(video_paths), current_indices
    )


def inspect_joint_sources(sim_root: Path, real_root: Path,
                          config: JointDatasetConfig) -> JointSources:
    """Inspect both declared domains and return immutable source contracts."""
    if config.fps != 25:
        raise ValueError("JointDatasetConfig fps must be canonical 25")
    if config.vector_dim != 14:
        raise ValueError("JointDatasetConfig vector_dim must be canonical 14")
    sim = _inspect_domain(Path(sim_root), "sim", config.sim_episodes, config.sim_frames, config)
    real = _inspect_domain(Path(real_root), "real", config.real_episodes, config.real_frames, config)
    sim_tasks = _validate_tasks(_read_jsonl(Path(sim_root) / "meta" / "tasks.jsonl", "sim tasks"), "sim")
    real_tasks = _validate_tasks(_read_jsonl(Path(real_root) / "meta" / "tasks.jsonl", "real tasks"), "real")
    if sim_tasks != real_tasks:
        raise ValueError("sim and real tasks are incompatible")
    return JointSources(sim=sim, real=real, tasks=sim_tasks)


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"missing required field {name}")
    return table.set_column(index, name, values)


def _loader_safe_numeric_stats(stats: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scalar declarations canonical while serializing stats as 1-D arrays."""
    result = {str(name): dict(value) for name, value in stats.items()}
    features = info.get("features", {})
    for name, declaration in features.items():
        if not isinstance(declaration, Mapping) or declaration.get("dtype") == "video":
            continue
        if tuple(declaration.get("shape", ())) != () or name not in result:
            continue
        for key, value in result[name].items():
            if key != "count":
                result[name][key] = np.asarray(value).reshape(1)
    return result


def canonicalize_episode(table: pa.Table, episode_index: int, global_start: int,
                         fps: int) -> pa.Table:
    """Cast fields and reorder every episode to one canonical Arrow schema."""
    if fps != 25:
        raise ValueError("fps must be canonical 25")
    missing = [name for name in _CANONICAL_COLUMN_ORDER if name not in table.column_names]
    if missing:
        raise ValueError(f"missing required field {missing[0]}")
    output = table
    for name in _VECTOR_FIELDS:
        if name not in table.column_names:
            raise ValueError(f"missing required feature {name}")
        try:
            values = np.asarray(table[name].to_pylist(), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid feature {name}") from exc
        if values.shape != (table.num_rows, 14):
            raise ValueError(f"feature {name} must have shape ({table.num_rows}, 14)")
        if not np.isfinite(values).all():
            raise ValueError(f"feature {name} contains NaN or Inf")
        array = pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1), type=pa.float32()), 14
        )
        output = _replace_column(output, name, array)
    output = _replace_column(output, "frame_index", pa.array(range(table.num_rows), type=pa.int64()))
    output = _replace_column(
        output, "episode_index", pa.array([episode_index] * table.num_rows, type=pa.int64())
    )
    output = _replace_column(
        output, "index", pa.array(range(global_start, global_start + table.num_rows), type=pa.int64())
    )
    timestamps = np.arange(table.num_rows, dtype=np.float32) / np.float32(fps)
    output = _replace_column(output, "timestamp", pa.array(timestamps, type=pa.float32()))
    extras = sorted(name for name in output.column_names if name not in _CANONICAL_COLUMN_ORDER)
    return output.select(list(_CANONICAL_COLUMN_ORDER) + extras)


@dataclass(frozen=True)
class JointBuildState:
    """Immutable result of assembling the unpublished joint tree."""

    total_episodes: int
    total_frames: int
    sim_frames: int
    real_frames: int
    episode_lengths: tuple[int, ...]
    episode_stats: tuple[dict[str, Any], ...]
    numeric_stats: dict[str, dict[str, np.ndarray]]
    info: dict[str, Any]


@dataclass(frozen=True)
class JointBuildReport:
    """Evidence returned after structural and optional loader validation."""

    total_episodes: int
    total_frames: int
    sim_frames: int
    real_frames: int
    same_row_mae: float = 0.0
    previous_action_mae: float = 0.0
    loader_smoke: bool = False
    real_same_row_mae: float = 0.0
    real_previous_action_mae: float = 0.0

    @property
    def episodes(self) -> int:
        return self.total_episodes

    @property
    def frames(self) -> int:
        return self.total_frames

    @property
    def sim_same_row_mae(self) -> float:
        return self.same_row_mae

    @property
    def sim_previous_action_mae(self) -> float:
        return self.previous_action_mae


_DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mp4"
)


def _joint_info(sources: JointSources, config: JointDatasetConfig) -> dict[str, Any]:
    """Build canonical output info without inheriting source counters."""
    info = dict(sources.sim.info)
    source_features = {
        str(name): dict(value) if isinstance(value, Mapping) else value
        for name, value in dict(info.get("features", {})).items()
    }
    # These fields are present in every output row and must participate in the
    # rebuilt numeric statistics even when an older source omitted declarations.
    scalar_fields = {
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    features: dict[str, Any] = {}
    for name in _VECTOR_FIELDS:
        declaration = source_features.get(name)
        if not isinstance(declaration, Mapping):
            declaration = {}
        declaration = dict(declaration)
        declaration.update(dtype="float32", shape=[config.vector_dim])
        features[name] = declaration
    for name, declaration in scalar_fields.items():
        features[name] = declaration
    for name in sorted(source_features):
        if name not in features and name not in _CAMERA_FIELDS:
            features[name] = source_features[name]
    for camera in _CAMERA_FIELDS:
        declaration = source_features.get(camera)
        if not isinstance(declaration, Mapping):
            declaration = {"dtype": "video", "shape": [480, 640, 3]}
        else:
            declaration = dict(declaration)
            # The global canonical template is used for all three output cameras.
            declaration.pop("video_path", None)
            declaration.pop("path", None)
        features[camera] = declaration
    info["features"] = features
    info["codebase_version"] = "v2.1"
    info["data_path"] = _DEFAULT_DATA_PATH
    info["video_path"] = _DEFAULT_VIDEO_PATH
    info["chunks_size"] = config.chunk_size
    total_episodes = sources.sim.episode_count + sources.real.episode_count
    info["total_episodes"] = total_episodes
    info["total_frames"] = sum(sources.sim.episode_lengths) + sum(sources.real.episode_lengths)
    info["fps"] = config.fps
    info["total_chunks"] = (total_episodes + config.chunk_size - 1) // config.chunk_size
    info["total_videos"] = total_episodes * len(_CAMERA_FIELDS)
    return info


def _output_path(root: Path, pattern: str, episode_index: int, config: JointDatasetConfig,
                 video_key: str | None = None) -> Path:
    return _format_path(root, pattern, episode_index, config.chunk_size, video_key)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(to_jsonable(value), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(to_jsonable(row), allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_sim_camera_stats(source: DomainSource) -> tuple[dict[str, Any], ...]:
    """Load and validate all simulation camera statistics once per assembly."""
    stats_path = source.root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        raise ValueError(f"sim episodes_stats.jsonl is required: {stats_path}")
    rows = _read_jsonl(stats_path, f"{source.root.name} episodes_stats.jsonl")
    if len(rows) != source.episode_count:
        raise ValueError(
            f"sim episodes_stats.jsonl must contain {source.episode_count} records"
        )
    ids = [row.get("episode_index") for row in rows]
    if ids != list(range(source.episode_count)):
        raise ValueError("sim episodes_stats.jsonl episode IDs are not ordered")
    all_results: list[dict[str, Any]] = []
    for episode_index, (row, output_length) in enumerate(zip(rows, source.episode_lengths)):
        nested = row.get("stats")
        if not isinstance(nested, Mapping):
            raise ValueError(f"sim episode {episode_index}: missing camera statistics")
        extra_cameras = sorted(
            str(name) for name in nested
            if str(name).startswith("observation.images.") and str(name) not in _CAMERA_FIELDS
        )
        if extra_cameras:
            raise ValueError(
                f"sim episode {episode_index} camera {extra_cameras[0]}: unsupported camera statistics"
            )
        result: dict[str, Any] = {}
        for camera in _CAMERA_FIELDS:
            if camera not in nested:
                raise ValueError(f"sim episode {episode_index} camera {camera}: missing statistics")
            camera_stats = nested[camera]
            context = f"sim episode {episode_index} camera {camera}"
            if not isinstance(camera_stats, Mapping):
                raise ValueError(f"{context}: invalid statistics")
            missing = sorted({"min", "max", "mean", "std", "count"} - set(camera_stats))
            if missing:
                raise ValueError(f"{context}: missing statistics fields {', '.join(missing)}")
            raw_count = camera_stats["count"]
            if isinstance(raw_count, (bool, str)):
                raise ValueError(f"{context}: invalid count")
            try:
                count_values = np.asarray(raw_count)
                if count_values.size != 1 or not np.issubdtype(count_values.dtype, np.number):
                    raise ValueError("count must be one numeric scalar")
                count_value = count_values.reshape(-1)[0]
                count_float = float(count_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{context}: invalid count") from exc
            if not np.isfinite(count_float) or count_float < 0 or not count_float.is_integer():
                raise ValueError(f"{context}: invalid count")
            count = int(count_float)
            if count != output_length:
                raise ValueError(
                    f"{context}: statistics count {count} "
                    f"does not match output length {output_length}"
                )
            result[camera] = dict(camera_stats)
        all_results.append(result)
    return tuple(all_results)


def _link_video(source_path: Path, destination_path: Path, domain: str, episode_index: int,
                camera: str) -> None:
    context = f"{domain} episode {episode_index} camera {camera}"
    if not source_path.exists():
        raise ValueError(f"{context}: missing video {source_path}")
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"{context}: source video must be a regular non-symlink file")
    if os.path.lexists(destination_path):
        raise ValueError(f"{context}: destination video already exists {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, destination_path)
    except OSError as exc:
        raise ValueError(f"{context}: unable to hard-link video {source_path} -> {destination_path}") from exc


def assemble_joint_tree(temporary_root: Path, sources: JointSources,
                        config: JointDatasetConfig) -> JointBuildState:
    """Write canonical Parquet rows and hard-linked videos in joint order."""
    if config.fps != 25 or config.vector_dim != 14:
        raise ValueError("joint assembly requires canonical fps=25 and vector_dim=14")
    root = Path(temporary_root)
    root.mkdir(parents=True, exist_ok=True)
    info = _joint_info(sources, config)
    numeric_accumulator = NumericStatsAccumulator()
    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    sim_count = sources.sim.episode_count
    sim_camera_stats = _load_sim_camera_stats(sources.sim)
    global_start = 0
    for domain, source, offset in (("sim", sources.sim, 0), ("real", sources.real, sim_count)):
        for source_index, source_path in enumerate(source.episode_paths):
            episode_index = offset + source_index
            table = _read_table(source_path)
            output = canonicalize_episode(table, episode_index, global_start, config.fps)
            output_path = _output_path(root, info["data_path"], episode_index, config)
            if os.path.lexists(output_path):
                raise ValueError(f"{domain} episode {source_index}: destination data already exists")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(output, output_path)
            numeric_accumulator.update(output, info)
            episode_numeric = compute_numeric_episode_stats(output, info)
            episode_numeric["stats"] = _loader_safe_numeric_stats(
                episode_numeric.get("stats", {}), info
            )
            episode_record: dict[str, Any] = {
                "episode_index": episode_index,
                "stats": dict(episode_numeric.get("stats", {})),
            }
            output_videos: dict[str, Path] = {}
            for camera, source_video in source.video_paths[source_index].items():
                destination_video = _output_path(root, info["video_path"], episode_index, config, camera)
                _link_video(source_video, destination_video, domain, source_index, camera)
                output_videos[camera] = destination_video
            if domain == "sim":
                episode_record["stats"].update(sim_camera_stats[source_index])
            elif output_videos:
                video_result = compute_video_episode_stats(
                    output_videos, output["timestamp"].to_pylist()
                )
                episode_record["stats"].update(video_result.get("stats", {}))
            records.append(episode_record)
            lengths.append(output.num_rows)
            global_start += output.num_rows
    return JointBuildState(
        total_episodes=len(records),
        total_frames=global_start,
        sim_frames=sum(lengths[:sim_count]),
        real_frames=sum(lengths[sim_count:]),
        episode_lengths=tuple(lengths),
        episode_stats=tuple(records),
        numeric_stats=_loader_safe_numeric_stats(numeric_accumulator.finalize(), info),
        info=info,
    )


def write_joint_metadata(root: Path, state: JointBuildState, sources: JointSources,
                         config: JointDatasetConfig) -> None:
    """Atomically write all joint metadata and the fixed-ratio domain manifest."""
    root = Path(root)
    total_episodes = sources.sim.episode_count + sources.real.episode_count
    if state.total_episodes != total_episodes or len(state.episode_stats) != total_episodes:
        raise ValueError("JointBuildState episode count does not match sources")
    if tuple(record.get("episode_index") for record in state.episode_stats) != tuple(range(total_episodes)):
        raise ValueError("JointBuildState episode statistics are not ordered")
    info = dict(state.info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = state.total_frames
    info["chunks_size"] = config.chunk_size
    info["total_chunks"] = (total_episodes + config.chunk_size - 1) // config.chunk_size
    info["total_videos"] = total_episodes * len(_CAMERA_FIELDS)
    _atomic_write_json(root / "meta" / "info.json", info)
    _atomic_write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": index, "length": length} for index, length in enumerate(state.episode_lengths)],
    )
    _atomic_write_jsonl(
        root / "meta" / "tasks.jsonl",
        list(sources.tasks),
    )
    episode_rows = [
        {"episode_index": int(record["episode_index"]), "stats": record.get("stats", {})}
        for record in state.episode_stats
    ]
    _atomic_write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_rows)
    global_stats: dict[str, Any] = dict(state.numeric_stats)
    pooled_camera_stats = _aggregate_stats(list(state.episode_stats))
    for name, stats in pooled_camera_stats.items():
        if name not in global_stats:
            global_stats[name] = stats
    _atomic_write_json(root / "meta" / "stats.json", global_stats)
    local_anchor_payload = _read_json(sources.real.root / "meta" / "anchor_indices.json")
    joint_anchor_records = [
        {
            **dict(record),
            "episode_index": int(record["episode_index"]) + sources.sim.episode_count,
            "index": int(record["index"]) + state.sim_frames,
        }
        for record in local_anchor_payload["records"]
    ]
    anchor_path = root / "meta" / "real_anchor_indices.json"
    _atomic_write_json(anchor_path, {
        "schema_version": 1,
        "count": len(joint_anchor_records),
        "records": joint_anchor_records,
    })
    anchor_digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    _atomic_write_json(root / "meta" / "domain_sampling.json", {
        "version": 2,
        "sim": {
            "episode_range": [0, sources.sim.episode_count],
            "probability": 0.75,
            "frames": state.sim_frames,
            "current_samples": {
                "kind": "range", "start": 0, "stop": state.sim_frames,
                "count": state.sim_frames,
            },
        },
        "real": {
            "episode_range": [sources.sim.episode_count, total_episodes],
            "probability": 0.25,
            "frames": state.real_frames,
            "current_samples": {
                "kind": "file", "path": "real_anchor_indices.json",
                "count": len(joint_anchor_records), "sha256": anchor_digest,
            },
        },
    })


def _stats_equal(actual: Any, expected: Any, path: str) -> None:
    """Compare JSON statistics without trusting metadata supplied by a source."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"stats mismatch at {path}")
        for key, value in expected.items():
            _stats_equal(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, np.ndarray) and expected.ndim == 0:
        _stats_equal(actual, expected.item(), path)
        return
    if isinstance(expected, (list, tuple, np.ndarray)):
        if not isinstance(actual, (list, tuple, np.ndarray)):
            raise ValueError(f"stats mismatch at {path}")
        left = np.asarray(actual)
        right = np.asarray(expected)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"stats mismatch at {path}")
        return
    if isinstance(expected, (float, np.floating)):
        if not isinstance(actual, (float, int)) or float(actual) != float(expected):
            raise ValueError(f"stats mismatch at {path}")
        return
    if actual != expected:
        raise ValueError(f"stats mismatch at {path}")


def _alignment_mae(source: DomainSource, config: JointDatasetConfig) -> tuple[float, float]:
    same_error = previous_error = 0.0
    same_count = previous_count = 0
    for episode_index, path in enumerate(source.episode_paths):
        table = _read_table(path)
        state = _vector_values(table, "observation.state", episode_index, config.vector_dim)
        action = _vector_values(table, "action", episode_index, config.vector_dim)
        same_error += float(np.abs(state - action).sum(dtype=np.float64))
        same_count += state.size
        if len(state) > 1:
            previous_error += float(np.abs(state[1:] - action[:-1]).sum(dtype=np.float64))
            previous_count += state[1:].size
    if not same_count or not previous_count:
        raise ValueError("cannot measure source alignment without at least two rows")
    return same_error / same_count, previous_error / previous_count


def _declared_output_files(root: Path, info: Mapping[str, Any], total_episodes: int,
                           config: JointDatasetConfig) -> tuple[set[Path], set[Path]]:
    data_pattern = info.get("data_path")
    if not isinstance(data_pattern, str) or not data_pattern:
        raise ValueError("joint info.json missing data_path")
    expected_data: set[Path] = set()
    expected_videos: set[Path] = set()
    for episode_index in range(total_episodes):
        data_path = _output_path(root, data_pattern, episode_index, config)
        if data_path in expected_data:
            raise ValueError(f"joint data path collision at episode {episode_index}")
        expected_data.add(data_path)
        videos = _video_paths(root, info, episode_index, config.chunk_size)
        if set(videos) != set(_CAMERA_FIELDS):
            raise ValueError("joint video declarations must contain exactly three cameras")
        expected_videos.update(videos.values())
    if len(expected_data) != total_episodes:
        raise ValueError("joint data declaration count mismatch")
    if len(expected_videos) != total_episodes * len(_CAMERA_FIELDS):
        raise ValueError("joint video declaration count mismatch")
    return expected_data, expected_videos


def _validate_declared_media(root: Path, expected_data: set[Path], expected_videos: set[Path]) -> None:
    expected_data = set(expected_data)
    expected_videos = set(expected_videos)
    actual_data: set[Path] = set()
    actual_videos: set[Path] = set()
    for candidate in root.rglob("*"):
        if os.path.lexists(os.fspath(candidate)) and candidate.suffix.lower() in _PARQUET_SUFFIXES:
            actual_data.add(candidate.resolve())
        if os.path.lexists(os.fspath(candidate)) and candidate.suffix.lower() in _VIDEO_SUFFIXES:
            actual_videos.add(candidate.resolve())
    missing_data = sorted(expected_data - actual_data)
    missing_videos = sorted(expected_videos - actual_videos)
    extra_data = sorted(actual_data - expected_data)
    extra_videos = sorted(actual_videos - expected_videos)
    if missing_data:
        raise ValueError(f"missing declared Parquet file: {missing_data[0]}")
    if missing_videos:
        raise ValueError(f"missing declared video file: {missing_videos[0]}")
    if extra_data:
        raise ValueError(f"unexpected Parquet file: {extra_data[0]}")
    if extra_videos:
        raise ValueError(f"unexpected video file: {extra_videos[0]}")


def _validate_output_videos(sources: JointSources, root: Path, info: Mapping[str, Any],
                            config: JointDatasetConfig) -> tuple[dict[str, Path], ...]:
    output_videos: list[dict[str, Path]] = []
    for domain, source, offset in (("sim", sources.sim, 0),
                                   ("real", sources.real, sources.sim.episode_count)):
        for source_index, source_paths in enumerate(source.video_paths):
            episode_index = offset + source_index
            destination_paths = _video_paths(root, info, episode_index, config.chunk_size)
            for camera in _CAMERA_FIELDS:
                destination = destination_paths[camera]
                source_path = source_paths[camera]
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"{domain} episode {source_index} camera {camera}: output video is not regular")
                if source_path.is_symlink() or not source_path.is_file():
                    raise ValueError(f"{domain} episode {source_index} camera {camera}: source video is not regular")
                destination_stat = destination.stat()
                source_stat = source_path.stat()
                if (destination_stat.st_dev, destination_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino):
                    raise ValueError(f"{domain} episode {source_index} camera {camera}: output video is not a hardlink")
            output_videos.append(destination_paths)
    return tuple(output_videos)


def _loader_smoke(root: Path, total_frames: int, sim_frames: int) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id="cobotmagic_joint_aligned_sr3to1",
                                 root=Path(root), video_backend="pyav")
        if len(dataset) != total_frames:
            raise ValueError(f"loader smoke length {len(dataset)} != {total_frames}")
        indices = (0, sim_frames - 1, sim_frames, total_frames - 1)
        for index in indices:
            sample = dataset[index]
            if not isinstance(sample, Mapping):
                raise ValueError(f"loader smoke sample at index {index} is not a mapping")
            sample_index = sample.get("index")
            if sample_index is not None:
                if hasattr(sample_index, "item"):
                    sample_index = sample_index.item()
                if int(sample_index) != index:
                    raise ValueError(f"loader smoke index mismatch at {index}")
    except ImportError as exc:
        raise RuntimeError("loader smoke requires LeRobot") from exc


def validate_joint_dataset(sim_root: Path, real_root: Path, joint_root: Path,
                           config: JointDatasetConfig, *, loader_smoke: bool = True,
                           verify_video_stats: bool = True) -> JointBuildReport:
    """Validate every row, metadata record, statistic, and hard-linked video."""
    root = Path(joint_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"joint destination is not a regular directory: {root}")
    sources = inspect_joint_sources(Path(sim_root), Path(real_root), config)
    total_episodes = sources.sim.episode_count + sources.real.episode_count
    total_frames = sum(sources.sim.episode_lengths) + sum(sources.real.episode_lengths)
    expected_sim_frames = sum(sources.sim.episode_lengths)
    info = _read_json(root / "meta" / "info.json")
    if not isinstance(info, Mapping):
        raise ValueError("joint info.json must be an object")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("joint metadata codebase_version must be v2.1")
    for key, expected in (("total_episodes", total_episodes), ("total_frames", total_frames),
                          ("fps", config.fps), ("chunks_size", config.chunk_size),
                          ("total_chunks", (total_episodes + config.chunk_size - 1) // config.chunk_size),
                          ("total_videos", total_episodes * len(_CAMERA_FIELDS))):
        if info.get(key) != expected:
            raise ValueError(f"joint metadata {key} mismatch")
    _validate_features(info, config.vector_dim, require_scalar_fields=True,
                       canonical_vectors=True)
    expected_data, expected_videos = _declared_output_files(root, info, total_episodes, config)
    _validate_declared_media(root, expected_data, expected_videos)
    output_videos = _validate_output_videos(sources, root, info, config)

    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl", "joint meta/episodes.jsonl")
    expected_lengths = sources.sim.episode_lengths + sources.real.episode_lengths
    if len(episode_rows) != total_episodes:
        raise ValueError("joint episodes metadata count mismatch")
    if [row.get("episode_index") for row in episode_rows] != list(range(total_episodes)):
        raise ValueError("joint episode metadata indexes are not ordered")
    if tuple(row.get("length") for row in episode_rows) != expected_lengths:
        raise ValueError("joint episode metadata lengths mismatch")
    tasks = tuple(_read_jsonl(root / "meta" / "tasks.jsonl", "joint tasks"))
    if tasks != sources.tasks:
        raise ValueError("joint tasks mismatch")

    numeric_accumulator = NumericStatsAccumulator()
    expected_episode_stats: list[dict[str, Any]] = []
    expected_schema = None
    global_start = 0
    for domain, source, offset in (("sim", sources.sim, 0),
                                   ("real", sources.real, sources.sim.episode_count)):
        sim_camera_stats = _load_sim_camera_stats(source) if domain == "sim" else ()
        for source_index, source_path in enumerate(source.episode_paths):
            episode_index = offset + source_index
            output_path = _output_path(root, str(info["data_path"]), episode_index, config)
            table = _read_table(output_path)
            expected = canonicalize_episode(_read_table(source_path), episode_index, global_start, config.fps)
            if table.column_names != expected.column_names:
                raise ValueError(f"episode {episode_index}: schema columns mismatch")
            schema = tuple((field.name, field.type, field.nullable) for field in table.schema)
            expected_sig = tuple((field.name, field.type, field.nullable) for field in expected.schema)
            if schema != expected_sig:
                raise ValueError(f"episode {episode_index}: Arrow schema mismatch ({schema} != {expected_sig})")
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(f"episode {episode_index}: Arrow schema differs from canonical schema")
            if table.num_rows != expected.num_rows:
                raise ValueError(f"episode {episode_index}: row count mismatch")
            for name in table.column_names:
                if table[name].to_pylist() != expected[name].to_pylist():
                    label = name if name in _VECTOR_FIELDS else f"{name} field"
                    raise ValueError(f"episode {episode_index}: {label} mismatch")
            numeric_accumulator.update(table, info)
            numeric_stats = _loader_safe_numeric_stats(
                compute_numeric_episode_stats(table, info).get("stats", {}), info
            )
            cameras: dict[str, Any]
            if domain == "sim":
                cameras = dict(sim_camera_stats[source_index])
            elif verify_video_stats:
                cameras = compute_video_episode_stats(output_videos[episode_index],
                                                       table["timestamp"].to_pylist()).get("stats", {})
            else:
                cameras = {}
            merged = dict(numeric_stats)
            if verify_video_stats:
                merged.update(cameras)
            expected_episode_stats.append({"episode_index": episode_index, "stats": merged})
            global_start += table.num_rows
    written_episode_stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl", "joint episodes_stats")
    if len(written_episode_stats) != total_episodes:
        raise ValueError("joint episodes_stats count mismatch")
    for index, (actual, expected) in enumerate(zip(written_episode_stats, expected_episode_stats)):
        if actual.get("episode_index") != index:
            raise ValueError(f"episode stats index mismatch at episode {index}")
        actual_stats = actual.get("stats")
        expected_stats = expected["stats"]
        if not verify_video_stats and isinstance(actual_stats, Mapping):
            actual_stats = {
                key: value for key, value in actual_stats.items()
                if not (isinstance(info.get("features", {}).get(key), Mapping)
                        and info["features"][key].get("dtype") == "video")
            }
        _stats_equal(actual_stats, expected_stats, f"episode {index} stats")
    expected_global: dict[str, Any] = to_jsonable(
        _loader_safe_numeric_stats(numeric_accumulator.finalize(), info)
    )
    if verify_video_stats:
        pooled = to_jsonable(_aggregate_stats(expected_episode_stats))
        for camera in _CAMERA_FIELDS:
            if camera in pooled:
                expected_global[camera] = pooled[camera]
    written_global = _read_json(root / "meta" / "stats.json")
    if not verify_video_stats and isinstance(written_global, Mapping):
        written_global = {
            key: value for key, value in written_global.items()
            if not (isinstance(info.get("features", {}).get(key), Mapping)
                    and info["features"][key].get("dtype") == "video")
        }
    _stats_equal(written_global, expected_global, "global stats")
    sampling = _read_json(root / "meta" / "domain_sampling.json")
    anchor_path = root / "meta" / "real_anchor_indices.json"
    anchor_payload = _read_json(anchor_path)
    expected_anchor_records = [
        {
            **dict(record),
            "episode_index": int(record["episode_index"]) + sources.sim.episode_count,
            "index": int(record["index"]) + expected_sim_frames,
        }
        for record in _read_json(sources.real.root / "meta" / "anchor_indices.json")["records"]
    ]
    expected_anchor_payload = {
        "schema_version": 1, "count": len(expected_anchor_records),
        "records": expected_anchor_records,
    }
    if anchor_payload != expected_anchor_payload:
        raise ValueError("joint real anchor metadata mismatch")
    expected_sampling = {
        "version": 2,
        "sim": {"episode_range": [0, sources.sim.episode_count], "probability": 0.75,
                 "frames": expected_sim_frames,
                 "current_samples": {"kind": "range", "start": 0,
                                     "stop": expected_sim_frames,
                                     "count": expected_sim_frames}},
        "real": {"episode_range": [sources.sim.episode_count, total_episodes], "probability": 0.25,
                  "frames": sum(sources.real.episode_lengths),
                  "current_samples": {"kind": "file",
                                      "path": "real_anchor_indices.json",
                                      "count": len(expected_anchor_records),
                                      "sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest()}},
    }
    if sampling != expected_sampling:
        raise ValueError("domain_sampling metadata mismatch")
    same_mae, previous_mae = _alignment_mae(sources.sim, config)
    real_same, real_previous = _alignment_mae(sources.real, config)
    if loader_smoke:
        _loader_smoke(root, total_frames, expected_sim_frames)
    return JointBuildReport(
        total_episodes, total_frames, expected_sim_frames,
        sum(sources.real.episode_lengths), same_mae, previous_mae,
        loader_smoke, real_same, real_previous,
    )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_destination_safe(sim_root: Path, real_root: Path, destination: Path) -> None:
    """Reject equal/descendant destinations using both lexical and resolved paths."""
    destination_variants = (destination.absolute(), destination.resolve())
    for source in (sim_root, real_root):
        source_variants = (source.absolute(), source.resolve())
        for source_path in source_variants:
            for destination_path in destination_variants:
                if destination_path == source_path or source_path in destination_path.parents:
                    raise ValueError("source and destination must differ and destination cannot be inside source")


def build_joint_dataset(sim_root: Path, real_root: Path, destination: Path,
                        config: JointDatasetConfig = JointDatasetConfig(), *,
                        validate_only: bool = False, skip_loader_smoke: bool = False) -> JointBuildReport:
    """Assemble and publish a joint dataset with hidden-temp/no-clobber semantics."""
    sim_root, real_root, destination = Path(sim_root), Path(real_root), Path(destination)
    _assert_destination_safe(sim_root, real_root, destination)
    if skip_loader_smoke and not validate_only:
        raise ValueError("--skip-loader-smoke is only legal with --validate-only")
    if validate_only:
        if not _path_exists(destination):
            raise FileNotFoundError(destination)
        return validate_joint_dataset(sim_root, real_root, destination, config,
                                      loader_smoke=not skip_loader_smoke)
    if _path_exists(destination):
        raise FileExistsError(destination)
    sources = inspect_joint_sources(sim_root, real_root, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        state = assemble_joint_tree(temporary, sources, config)
        write_joint_metadata(temporary, state, sources, config)
        validate_joint_dataset(sim_root, real_root, temporary, config,
                               loader_smoke=False, verify_video_stats=False)
        report = validate_joint_dataset(sim_root, real_root, temporary, config,
                                        loader_smoke=True)
        rename_path_no_clobber(temporary, destination)
        temporary = None  # type: ignore[assignment]
        return report
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

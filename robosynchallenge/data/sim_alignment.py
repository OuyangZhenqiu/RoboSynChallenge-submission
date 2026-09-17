"""Validation and exact per-episode action alignment for LeRobot tables."""

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import shutil
import ctypes
import errno
import os
import stat
import tempfile
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/Arrow values into JSON-native values."""
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        return to_jsonable(converted)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def array_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.ndim == 0 or values64.shape[0] == 0:
        raise ValueError("cannot compute statistics for empty values")
    result = {"min": values64.min(axis=0), "max": values64.max(axis=0),
              "mean": values64.mean(axis=0), "std": values64.std(axis=0),
              "count": np.array([values64.shape[0]], dtype=np.int64)}
    result.update({name: np.quantile(values64, q, axis=0) for name, q in QUANTILES.items()})
    return result


def compute_numeric_episode_stats(table: pa.Table, info: Mapping[str, Any]) -> dict[str, Any]:
    """Compute per-feature statistics over the rows in one aligned episode."""
    result: dict[str, Any] = {}
    features = info.get("features", info) if isinstance(info, Mapping) else info
    for name, declaration in features.items():
        if name not in table.column_names or not isinstance(declaration, Mapping):
            continue
        if declaration.get("dtype") in {"video", "string"}:
            continue
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if values.ndim < 1:
            continue
        shape = tuple(declaration.get("shape", values.shape[1:]))
        flattened = values.reshape(values.shape[0], -1)
        result[name] = {key: val.reshape(shape) if key != "count" else val
                        for key, val in array_stats(flattened).items()}
    episode_index = int(table["episode_index"][0].as_py()) if "episode_index" in table.column_names and table.num_rows else 0
    result["episode_index"] = episode_index
    result["stats"] = {key: value for key, value in result.items() if key != "episode_index"}
    return result


class NumericStatsAccumulator:
    """Accumulate retained numeric rows and compute exact global quantiles."""

    def __init__(self) -> None:
        self._rows: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

    def update(self, table: pa.Table, info: Mapping[str, Any]) -> None:
        features = info.get("features", info) if isinstance(info, Mapping) else info
        for name, declaration in features.items():
            if name not in table.column_names or not isinstance(declaration, Mapping):
                continue
            if declaration.get("dtype") in {"video", "string"}:
                continue
            values = np.asarray(table[name].to_pylist(), dtype=np.float64)
            if values.ndim < 1:
                continue
            shape = tuple(declaration.get("shape", values.shape[1:]))
            self._shapes[name] = shape
            self._rows.setdefault(name, []).append(values.reshape(values.shape[0], -1))

    def finalize(self) -> dict[str, dict[str, np.ndarray]]:
        result = {}
        for name, chunks in self._rows.items():
            stats = array_stats(np.concatenate(chunks, axis=0))
            shape = self._shapes[name]
            result[name] = {key: value.reshape(shape) if key != "count" else value
                            for key, value in stats.items()}
        return result


class _StreamingRGBStats:
    """Accumulate exact uint8 RGB distribution statistics in constant memory."""

    def __init__(self) -> None:
        self._histogram = np.zeros((3, 256), dtype=np.uint64)

    def update(self, frame: np.ndarray, multiplicity: int = 1) -> None:
        values = np.asarray(frame)
        if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("decoded video frame must be uint8 RGB")
        if isinstance(multiplicity, bool) or multiplicity <= 0:
            raise ValueError("frame multiplicity must be a positive integer")
        height, width = values.shape[:2]
        factor = max(1, max(height, width) // 150) if max(height, width) > 300 else 1
        sampled = values[::factor, ::factor]
        weight = np.uint64(multiplicity)
        for channel in range(3):
            counts = np.bincount(sampled[..., channel].reshape(-1), minlength=256)
            self._histogram[channel] += counts.astype(np.uint64) * weight

    @staticmethod
    def _value_at(cumulative: np.ndarray, index: int) -> int:
        return int(np.searchsorted(cumulative, index, side="right"))

    def finalize(self, selected_frames: int) -> dict[str, np.ndarray]:
        pixel_counts = self._histogram.sum(axis=1)
        if selected_frames <= 0 or np.any(pixel_counts == 0):
            raise ValueError("cannot compute statistics for empty video selection")
        values = np.arange(256, dtype=np.float64)
        counts64 = self._histogram.astype(np.float64)
        means = (counts64 @ values) / pixel_counts.astype(np.float64) / 255.0
        second_moments = (
            (counts64 @ np.square(values))
            / pixel_counts.astype(np.float64)
            / (255.0 * 255.0)
        )
        variances = np.maximum(second_moments - np.square(means), 0.0)
        nonzero = self._histogram > 0
        minimum = np.argmax(nonzero, axis=1).astype(np.float64) / 255.0
        maximum = (255 - np.argmax(nonzero[:, ::-1], axis=1)).astype(np.float64) / 255.0
        result: dict[str, np.ndarray] = {
            "min": minimum,
            "max": maximum,
            "mean": means,
            "std": np.sqrt(variances),
            "count": np.array([selected_frames], dtype=np.int64),
        }
        cumulative = np.cumsum(self._histogram, axis=1)
        for name, quantile in QUANTILES.items():
            channel_values = []
            for channel in range(3):
                position = quantile * (int(pixel_counts[channel]) - 1)
                lower_index = int(np.floor(position))
                upper_index = int(np.ceil(position))
                lower = self._value_at(cumulative[channel], lower_index)
                upper = self._value_at(cumulative[channel], upper_index)
                fraction = position - lower_index
                channel_values.append(((1.0 - fraction) * lower + fraction * upper) / 255.0)
            result[name] = np.asarray(channel_values, dtype=np.float64)
        return {
            key: value if key == "count" else value.reshape(3, 1, 1)
            for key, value in result.items()
        }


def _fallback_video_episode_stats(path: Path, timestamps: Sequence[float]) -> dict[str, np.ndarray]:
    """Decode bounded timestamp batches when PyAV cannot be imported directly."""
    from lerobot.datasets.video_utils import decode_video_frames

    accumulator = _StreamingRGBStats()
    selected_frames = 0
    batch_size = 32
    for start in range(0, len(timestamps), batch_size):
        batch_timestamps = timestamps[start:start + batch_size]
        frames = decode_video_frames(
            str(path), batch_timestamps, tolerance_s=1e-4, backend="pyav"
        )
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        if frames.ndim == 4 and frames.shape[1] == 3:
            frames = np.moveaxis(frames, 1, -1)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"invalid decoded frames for video {path}")
        for frame in frames:
            if frame.dtype != np.uint8:
                if not np.issubdtype(frame.dtype, np.floating) or not np.isfinite(frame).all():
                    raise ValueError(f"invalid decoded frames for video {path}")
                scale = 255.0 if float(np.max(frame)) <= 1.0 else 1.0
                frame = np.rint(np.clip(frame * scale, 0.0, 255.0)).astype(np.uint8)
            accumulator.update(frame)
            selected_frames += 1
        del frames
    return accumulator.finalize(selected_frames)


def _stream_video_episode_stats(path: Path, timestamps: Sequence[float]) -> dict[str, np.ndarray]:
    """Decode one PyAV frame at a time and retain only the last frame for EOF clamping."""
    if len(timestamps) == 0:
        raise ValueError(f"video selection contains no timestamps: {path}")
    try:
        import av
    except ImportError:
        return _fallback_video_episode_stats(path, timestamps)

    accumulator = _StreamingRGBStats()
    selected_frames = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video contains no video stream: {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        try:
            requested = Counter(max(int(round(float(ts) * fps)), 0) for ts in timestamps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"video contains invalid timestamps: {path}") from exc
        last_frame: np.ndarray | None = None
        decoded_count = 0
        for frame_index, decoded_frame in enumerate(container.decode(stream)):
            frame = decoded_frame.to_ndarray(format="rgb24")
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"invalid decoded frame for video {path}")
            last_frame = frame
            multiplicity = requested.pop(frame_index, 0)
            if multiplicity:
                accumulator.update(frame, multiplicity)
                selected_frames += multiplicity
            decoded_count = frame_index + 1
        if last_frame is None:
            raise ValueError(f"video contains no frames: {path}")
        beyond_eof = sum(requested.values())
        if beyond_eof:
            accumulator.update(last_frame, beyond_eof)
            selected_frames += beyond_eof
        del last_frame
    return accumulator.finalize(selected_frames)


def compute_video_episode_stats(video_paths: Mapping[str, Path], timestamps: Sequence[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera, path_value in video_paths.items():
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing video for {camera} episode: {path}")
        try:
            result[camera] = _stream_video_episode_stats(path, timestamps)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise ValueError(
                f"invalid video statistics for {camera} episode ({path}): {exc}"
            ) from exc
        except Exception as exc:
            raise ValueError(f"unable to compute video statistics for {camera} episode: {path}") from exc
    result["episode_index"] = 0
    result["stats"] = {key: value for key, value in result.items() if key != "episode_index"}
    return result


@dataclass(frozen=True)
class AlignmentConfig:
    expected_episodes: int = 2000
    source_rows_per_episode: int = 74
    fps: int = 25
    vector_dim: int = 14


@dataclass(frozen=True)
class AlignmentMetrics:
    same_row_mae: float
    previous_action_mae: float


@dataclass(frozen=True)
class AlignmentReport:
    episodes: int
    frames: int
    metrics: AlignmentMetrics
    loader_smoke: bool = False

    @property
    def same_row_mae(self) -> float:
        return self.metrics.same_row_mae

    @property
    def previous_action_mae(self) -> float:
        return self.metrics.previous_action_mae


@dataclass(frozen=True)
class SourceDataset:
    info: dict[str, Any]
    state_key: str
    episode_paths: tuple[Path, ...]


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, values)


def _vector_array(table: pa.Table, name: str, episode_index: int, config: AlignmentConfig) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"episode {episode_index}: missing required field {name}")
    try:
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode {episode_index}: invalid field {name}") from exc
    expected = (config.source_rows_per_episode, config.vector_dim)
    if values.shape != expected:
        raise ValueError(f"episode {episode_index}: field {name} must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"episode {episode_index}: field {name} contains NaN or Inf")
    return values


def validate_episode_table(table: pa.Table, episode_index: int, config: AlignmentConfig,
                           state_key: str = "observation.state") -> None:
    if table.num_rows != config.source_rows_per_episode:
        raise ValueError(f"episode {episode_index}: wrong source row count {table.num_rows}; expected {config.source_rows_per_episode}")
    for name in (state_key, "observation.qvel", "observation.qf", "action"):
        _vector_array(table, name, episode_index, config)
    for name in ("frame_index", "episode_index", "index", "timestamp"):
        if name not in table.column_names:
            raise ValueError(f"episode {episode_index}: missing required field {name}")
    frames = table["frame_index"].to_pylist()
    if frames != list(range(config.source_rows_per_episode)):
        raise ValueError(f"episode {episode_index}: frame_index must be contiguous from zero")
    episodes = table["episode_index"].to_pylist()
    if episodes != [episode_index] * config.source_rows_per_episode:
        raise ValueError(f"episode {episode_index}: episode_index column does not match episode")


def transform_episode(table: pa.Table, episode_index: int, global_start: int, config: AlignmentConfig,
                      state_key: str = "observation.state") -> pa.Table:
    validate_episode_table(table, episode_index, config, state_key=state_key)
    output = table.slice(0, table.num_rows - 1)
    output = _replace_column(output, "action", table["action"].slice(1, output.num_rows))
    output = _replace_column(output, "frame_index", pa.array(range(output.num_rows), type=pa.int64()))
    output = _replace_column(output, "episode_index", pa.array([episode_index] * output.num_rows, type=pa.int64()))
    output = _replace_column(output, "index", pa.array(range(global_start, global_start + output.num_rows), type=pa.int64()))
    timestamps = np.arange(output.num_rows, dtype=np.float32) / config.fps
    return _replace_column(output, "timestamp", pa.array(timestamps, type=pa.float32()))


def _read_table(path: Path) -> pa.Table:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pq.read_table(path)
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except (pa.ArrowInvalid, ValueError):
            source.seek(0)
            return ipc.open_stream(source).read_all()


_PARQUET_SUFFIXES = {".parquet", ".pq"}
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _reject_unexpected_media_files(root: Path, expected_data: set[Path],
                                   expected_videos: set[Path]) -> None:
    """Reject undeclared Parquet/video files that could be silently ignored."""
    expected_data = {Path(path) for path in expected_data}
    expected_videos = {Path(path) for path in expected_videos}
    for candidate in Path(root).rglob("*"):
        if not (candidate.is_file() or os.path.lexists(os.fspath(candidate))):
            continue
        suffix = candidate.suffix.lower()
        if suffix in _PARQUET_SUFFIXES and candidate not in expected_data:
            raise ValueError(f"unexpected Parquet file: {candidate}")
        if suffix in _VIDEO_SUFFIXES and candidate not in expected_videos:
            raise ValueError(f"unexpected video file: {candidate}")


def measure_alignment(episode_paths: Sequence[Path], state_key: str) -> AlignmentMetrics:
    same_error = previous_error = 0.0
    same_count = previous_count = 0
    for path in episode_paths:
        table = _read_table(Path(path))
        if state_key not in table.column_names or "action" not in table.column_names:
            raise ValueError(f"missing {state_key} or action in {path}")
        state = np.asarray(table[state_key].to_pylist(), dtype=np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if state.shape != action.shape or state.ndim != 2:
            raise ValueError(f"invalid state/action shape in {path}")
        same_error += np.abs(state - action).sum(dtype=np.float64)
        same_count += state.size
        if len(state) > 1:
            previous_error += np.abs(state[1:] - action[:-1]).sum(dtype=np.float64)
            previous_count += state[1:].size
    if not same_count or not previous_count:
        raise ValueError("cannot measure alignment without at least two rows")
    return AlignmentMetrics(same_error / same_count, previous_error / previous_count)


def inspect_source(root: Path, config: AlignmentConfig) -> SourceDataset:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.exists():
        raise ValueError("missing meta/info.json")
    if not episodes_path.exists():
        raise ValueError("missing meta/episodes.jsonl")
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise ValueError("missing meta/tasks.jsonl")
    info = json.loads(info_path.read_text())
    if info.get("total_episodes") != config.expected_episodes:
        raise ValueError("info.json total_episodes does not match expected_episodes")
    entries = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    ids = [int(entry["episode_index"]) for entry in entries]
    if len(entries) != config.expected_episodes or sorted(ids) != list(range(config.expected_episodes)):
        raise ValueError("episode IDs are not contiguous")
    if any(entry.get("length") != config.source_rows_per_episode for entry in entries):
        raise ValueError("episodes.jsonl contains an incorrect episode length")
    pattern = info.get("data_path")
    if not pattern:
        raise ValueError("info.json missing data_path")
    paths = []
    for index in sorted(ids):
        path = root / pattern.format(episode_index=index, episode_chunk=index // 1000)
        if not path.exists():
            raise ValueError(f"missing episode {index} data file: {path}")
        table = _read_table(path)
        if table.num_rows != config.source_rows_per_episode:
            raise ValueError(f"episode {index}: wrong source row count")
        paths.append(path)
    first = _read_table(paths[0])
    state_key = "observation.state" if "observation.state" in first.column_names else "observation.qpos" if "observation.qpos" in first.column_names else ""
    if not state_key:
        raise ValueError("missing observation.state and observation.qpos")
    metrics = measure_alignment(paths, state_key)
    if not metrics.same_row_mae < metrics.previous_action_mae:
        raise ValueError("source alignment convention is not post-step current-action")
    _reject_unexpected_media_files(root, set(paths), {
        video for index in range(config.expected_episodes)
        for video in _video_paths(root, info, index).values()
    })
    return SourceDataset(info=info, state_key=state_key, episode_paths=tuple(paths))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(value), allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _aggregate_stats(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    aggregate: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        nested = record.get("stats", record)
        for name, stats in nested.items():
            if name in {"episode_index", "stats"} or not isinstance(stats, Mapping) or "mean" not in stats:
                continue
            count = float(np.asarray(stats["count"]).reshape(-1)[0])
            mean = np.asarray(stats["mean"], dtype=np.float64)
            variance = np.asarray(stats["std"], dtype=np.float64) ** 2
            current = aggregate.get(name)
            if current is None:
                aggregate[name] = {"count": np.array([count]), "mean": mean.copy(),
                                   "std": np.sqrt(variance), "min": np.asarray(stats["min"]),
                                   "max": np.asarray(stats["max"])}
                continue
            old_count = float(current["count"][0]); total = old_count + count
            delta = mean - current["mean"]
            pooled_var = (old_count * current["std"] ** 2 + count * variance +
                          old_count * count / total * delta ** 2) / total
            current["mean"] = (old_count * current["mean"] + count * mean) / total
            current["std"] = np.sqrt(pooled_var)
            current["count"] = np.array([total], dtype=np.int64)
            current["min"] = np.minimum(current["min"], stats["min"])
            current["max"] = np.maximum(current["max"], stats["max"])
    return aggregate


def write_rebuilt_metadata(root: Path, source: SourceDataset, episode_stats: Sequence[dict],
                           global_numeric_stats: Mapping[str, dict[str, np.ndarray]],
                           config: AlignmentConfig) -> None:
    """Write v2.1 JSONL metadata for an already-created temporary dataset."""
    root = Path(root)
    expected_ids = list(range(config.expected_episodes))
    actual_ids = [record.get("episode_index") for record in episode_stats]
    if len(episode_stats) != config.expected_episodes or actual_ids != expected_ids:
        raise ValueError("episode_stats must contain exactly expected ordered episode records")
    if any("stats" not in record or not isinstance(record["stats"], Mapping) for record in episode_stats):
        raise ValueError("episode_stats records must contain nested stats")
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    source_root = next((candidate for candidate in source.episode_paths[0].parents
                        if (candidate / "meta" / "info.json").exists()), source.episode_paths[0].parents[1])
    source_meta = source_root / "meta"
    tasks = source_meta / "tasks.jsonl"
    if tasks.exists():
        shutil.copy2(tasks, meta / "tasks.jsonl")

    episodes = [{"episode_index": index, "length": config.source_rows_per_episode - 1}
                for index in range(config.expected_episodes)]
    _atomic_jsonl = lambda path, rows: _atomic_text(path, "\n".join(json.dumps(to_jsonable(row), allow_nan=False) for row in rows) + "\n")
    _atomic_jsonl(meta / "episodes.jsonl", episodes)
    stats_rows = [{"episode_index": int(record.get("episode_index", index)),
                   "stats": to_jsonable(record.get("stats", {k: v for k, v in record.items() if k != "episode_index"}))}
                  for index, record in enumerate(episode_stats)]
    _atomic_jsonl(meta / "episodes_stats.jsonl", stats_rows)

    info = dict(source.info)
    info["total_frames"] = config.expected_episodes * (config.source_rows_per_episode - 1)
    _atomic_json(meta / "info.json", info)
    global_stats = {name: dict(stats) for name, stats in global_numeric_stats.items()}
    video_stats = _aggregate_stats(episode_stats)
    for name, stats in video_stats.items():
        if name not in global_stats:
            global_stats[name] = stats
    _atomic_json(meta / "stats.json", global_stats)


def _path_exists(path: Path) -> bool:
    """Like exists(), but true for dangling symlinks too."""
    return os.path.lexists(os.fspath(path))


def rename_path_no_clobber(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing a competing destination."""
    source = Path(source); destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Linux renameat2(RENAME_NOREPLACE), available on the production platform.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unavailable; refusing publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(destination)
    if error in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
        raise OSError(error, "atomic no-replace rename unavailable; refusing publication", destination)
    raise OSError(error, os.strerror(error), destination)


def _format_path(root: Path, pattern: str, episode_index: int, video_key: str | None = None) -> Path:
    return root / pattern.format(episode_index=episode_index, episode_chunk=episode_index // 1000,
                                 video_key=video_key or "")


def _video_paths(root: Path, info: Mapping[str, Any], episode_index: int) -> dict[str, Path]:
    pattern = info.get("video_path")
    features = info.get("features", {})
    result: dict[str, Path] = {}
    if pattern:
        for name, declaration in features.items():
            if isinstance(declaration, Mapping) and declaration.get("dtype") == "video":
                result[name] = _format_path(root, str(pattern), episode_index, name)
    for name, declaration in features.items():
        if isinstance(declaration, Mapping) and declaration.get("dtype") == "video":
            path = declaration.get("video_path") or declaration.get("path")
            if path:
                result[name] = _format_path(root, str(path), episode_index)
    return result


def _json_stats_equal(actual: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"stats mismatch at {path}")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"stats missing {path}.{key}")
            _json_stats_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"stats mismatch at {path}")
        for index, value in enumerate(expected):
            _json_stats_equal(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, (int, float)):
        if not np.isclose(float(actual), float(expected), rtol=1e-6, atol=1e-7):
            raise ValueError(f"stats mismatch at {path}")
    elif actual != expected:
        raise ValueError(f"stats mismatch at {path}")


def validate_aligned_dataset(source_root: Path, aligned_root: Path, config: AlignmentConfig,
                             loader_smoke: bool) -> AlignmentReport:
    source = inspect_source(Path(source_root), config)
    aligned_root = Path(aligned_root)
    info_path = aligned_root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError("aligned dataset missing meta/info.json")
    info = json.loads(info_path.read_text())
    expected_frames = config.expected_episodes * (config.source_rows_per_episode - 1)
    if info.get("total_frames") != expected_frames:
        raise ValueError(f"metadata total_frames must be {expected_frames}")
    tasks_path = aligned_root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise ValueError("aligned dataset missing meta/tasks.jsonl")
    acc = NumericStatsAccumulator(); episodes_stats = []
    pattern = info.get("data_path")
    if not pattern:
        raise ValueError("aligned metadata missing data_path")
    expected_output_data = {
        _format_path(aligned_root, pattern, index)
        for index in range(config.expected_episodes)
    }
    expected_output_videos = {
        video for index in range(config.expected_episodes)
        for video in _video_paths(aligned_root, info, index).values()
    }
    _reject_unexpected_media_files(aligned_root, expected_output_data, expected_output_videos)
    for episode_index, source_path in enumerate(source.episode_paths):
        output_path = _format_path(aligned_root, pattern, episode_index)
        if not output_path.exists():
            raise ValueError(f"episode {episode_index}: missing output data file")
        output = _read_table(output_path); original = _read_table(source_path)
        if output.num_rows != config.source_rows_per_episode - 1:
            raise ValueError(f"episode {episode_index}: wrong output row count {output.num_rows}")
        rebuilt_types = {
            "frame_index": pa.int64(),
            "episode_index": pa.int64(),
            "index": pa.int64(),
            "timestamp": pa.float32(),
        }
        for name in original.column_names:
            if name == "action":
                continue
            if name not in output.column_names:
                raise ValueError(f"episode {episode_index}: missing field {name}")
            expected_type = rebuilt_types.get(name, original.schema.field(name).type)
            actual_type = output.schema.field(name).type
            if actual_type != expected_type:
                raise ValueError(
                    f"episode {episode_index}: field {name} Arrow schema type mismatch; "
                    f"expected {expected_type}, got {actual_type}"
                )
            if name in rebuilt_types:
                continue
            np.testing.assert_array_equal(np.asarray(output[name].to_pylist()), np.asarray(original[name].to_pylist())[:-1], err_msg=f"episode {episode_index} field {name}")
        if "action" not in output.column_names:
            raise ValueError(f"episode {episode_index}: missing field action")
        if output.schema.field("action").type != original.schema.field("action").type:
            raise ValueError(
                f"episode {episode_index}: field action Arrow schema type mismatch; "
                f"expected {original.schema.field('action').type}, got {output.schema.field('action').type}"
            )
        np.testing.assert_array_equal(np.asarray(output["action"].to_pylist()), np.asarray(original["action"].to_pylist())[1:], err_msg=f"episode {episode_index} field action")
        if output["frame_index"].to_pylist() != list(range(output.num_rows)):
            raise ValueError(f"episode {episode_index}: frame_index is not contiguous")
        if output["episode_index"].to_pylist() != [episode_index] * output.num_rows:
            raise ValueError(f"episode {episode_index}: episode_index mismatch")
        expected_start = episode_index * (config.source_rows_per_episode - 1)
        if output["index"].to_pylist() != list(range(expected_start, expected_start + output.num_rows)):
            raise ValueError(f"episode {episode_index}: index is not contiguous")
        expected_ts = (np.arange(output.num_rows, dtype=np.float32) / config.fps).tolist()
        np.testing.assert_allclose(output["timestamp"].to_pylist(), expected_ts, rtol=1e-6, atol=1e-7, err_msg=f"episode {episode_index} timestamp")
        acc.update(output, info); episodes_stats.append(compute_numeric_episode_stats(output, info))
        source_videos = _video_paths(source_root, source.info, episode_index)
        videos = _video_paths(aligned_root, info, episode_index)
        if set(videos) != set(source_videos):
            raise ValueError(f"episode {episode_index}: declared video fields do not match source")
        for name, source_video in source_videos.items():
            path = videos[name]
            try:
                source_stat = source_video.lstat()
                output_stat = path.lstat()
            except FileNotFoundError as exc:
                raise ValueError(f"episode {episode_index}: missing video {name}") from exc
            if (source_video.is_symlink() or path.is_symlink() or
                    not stat.S_ISREG(source_stat.st_mode) or not stat.S_ISREG(output_stat.st_mode)):
                raise ValueError(f"episode {episode_index}: video {name} must be a regular non-symlink hard link")
            if (source_stat.st_dev, source_stat.st_ino) != (output_stat.st_dev, output_stat.st_ino):
                raise ValueError(f"episode {episode_index}: video {name} must be a hard link to source")
        if videos:
            video_stats = compute_video_episode_stats(videos, output["timestamp"].to_pylist())
            episodes_stats[-1].setdefault("stats", {}).update(video_stats.get("stats", {}))
    stats_path = aligned_root / "meta" / "stats.json"
    if not stats_path.exists():
        raise ValueError("aligned dataset missing meta/stats.json")
    written_stats = json.loads(stats_path.read_text())
    expected_global = dict(acc.finalize())
    for name, stats in _aggregate_stats(episodes_stats).items():
        expected_global.setdefault(name, stats)
    if set(written_stats) != set(expected_global):
        raise ValueError(f"stats keys mismatch: expected {sorted(expected_global)}, got {sorted(written_stats)}")
    for name, stats in expected_global.items():
        if name not in written_stats:
            raise ValueError(f"stats missing {name}")
        _json_stats_equal(written_stats[name], to_jsonable(stats), name)
    episodes_stats_path = aligned_root / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        raise ValueError("aligned dataset missing meta/episodes_stats.jsonl")
    rows = [json.loads(line) for line in episodes_stats_path.read_text().splitlines() if line.strip()]
    if [row.get("episode_index") for row in rows] != list(range(config.expected_episodes)):
        raise ValueError("episodes_stats.jsonl episode IDs are not contiguous")
    for index, row in enumerate(rows):
        if set(row.get("stats", {})) != set(episodes_stats[index].get("stats", {})):
            raise ValueError(f"episode {index}: episodes_stats keys mismatch")
        _json_stats_equal(row.get("stats", {}), to_jsonable(episodes_stats[index].get("stats", {})), f"episode {index} stats")
    episodes_path = aligned_root / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()] if episodes_path.exists() else []
    expected_episodes = [{"episode_index": i, "length": config.source_rows_per_episode - 1} for i in range(config.expected_episodes)]
    if records != expected_episodes:
        raise ValueError("episodes.jsonl records do not match aligned episode IDs/lengths")
    if loader_smoke:
        try:
            with warnings.catch_warnings():
                # LeRobot currently imports a pinned torchvision compatibility
                # shim that emits this exact upstream deprecation warning.
                warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities of torchvision are deprecated", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                # Temporary publication roots intentionally begin with a dot;
                # use a valid local repo id so LeRobot avoids Hub validation.
                dataset = LeRobotDataset(repo_id="aligned_dataset", root=aligned_root, video_backend="pyav")
                for index in {0, 72, 73, len(dataset) - 1}:
                    if index < 0 or index >= len(dataset):
                        continue
                    sample = dataset[index]
                    for key in ("action", source.state_key):
                        if tuple(sample[key].shape) != (config.vector_dim,):
                            raise ValueError(f"loader smoke {key} shape at index {index}")
                    for key in info.get("features", {}):
                        if info["features"][key].get("dtype") == "video":
                            if tuple(sample[key].shape) != (3, 480, 640):
                                raise ValueError(f"loader smoke {key} shape at index {index}")
        except ImportError as exc:
            raise RuntimeError("loader smoke requires LeRobot") from exc
    return AlignmentReport(config.expected_episodes, expected_frames,
                           measure_alignment(tuple(_format_path(aligned_root, pattern, i) for i in range(config.expected_episodes)), source.state_key), loader_smoke)


def convert_dataset(source_root: Path, destination_root: Path, config: AlignmentConfig,
                    validate_only: bool = False) -> AlignmentReport:
    source_root = Path(source_root); destination_root = Path(destination_root)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("source and destination must differ")
    if validate_only:
        if not _path_exists(destination_root):
            raise FileNotFoundError(destination_root)
        return validate_aligned_dataset(source_root, destination_root, config, loader_smoke=True)
    if _path_exists(destination_root):
        raise FileExistsError(destination_root)
    source = inspect_source(source_root, config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination_root.name}.", dir=destination_root.parent))
    try:
        info = dict(source.info)
        data_pattern = info["data_path"]
        global_acc = NumericStatsAccumulator(); records = []
        for episode_index, source_path in enumerate(source.episode_paths):
            output = transform_episode(
                _read_table(source_path), episode_index,
                episode_index * (config.source_rows_per_episode - 1), config,
                state_key=source.state_key,
            )
            output_path = _format_path(temporary, data_pattern, episode_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() in {".parquet", ".pq"}:
                pq.write_table(output, output_path)
            else:
                with pa.OSFile(str(output_path), "wb") as sink:
                    with ipc.new_file(sink, output.schema) as writer: writer.write_table(output)
            global_acc.update(output, info)
            records.append(compute_numeric_episode_stats(output, info))
            for name, source_video in _video_paths(source_root, info, episode_index).items():
                destination_video = _video_paths(temporary, info, episode_index).get(name)
                if not source_video.exists():
                    raise FileNotFoundError(f"missing video {name} episode {episode_index}: {source_video}")
                if destination_video is not None:
                    destination_video.parent.mkdir(parents=True, exist_ok=True); os.link(source_video, destination_video)
            output_videos = _video_paths(temporary, info, episode_index)
            if output_videos:
                records[-1].setdefault("stats", {}).update(
                    compute_video_episode_stats(output_videos, output["timestamp"].to_pylist()).get("stats", {})
                )
        write_rebuilt_metadata(temporary, source, records, global_acc.finalize(), config)
        validate_aligned_dataset(source_root, temporary, config, loader_smoke=False)
        report = validate_aligned_dataset(source_root, temporary, config, loader_smoke=True)
        rename_path_no_clobber(temporary, destination_root)
        temporary = None
        return report
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

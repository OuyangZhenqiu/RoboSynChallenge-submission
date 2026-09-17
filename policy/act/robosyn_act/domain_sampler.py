"""Opt-in fixed-ratio sampling for the aligned simulation/real dataset."""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DomainSamplingSpec:
    """Validated contents of ``meta/domain_sampling.json``."""

    sim_episode_range: tuple[int, int]
    real_episode_range: tuple[int, int]
    sim_probability: float
    real_probability: float
    sim_frames: int
    real_frames: int
    sim_current_indices: tuple[int, ...] | None = None
    real_current_indices: tuple[int, ...] | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DomainSamplingSpec":
        manifest_path = Path(path)
        try:
            with manifest_path.open() as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid domain sampling manifest: {manifest_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError("domain sampling manifest must be a JSON object")
        version = payload.get("version")
        if isinstance(version, bool) or version not in (1, 2):
            raise ValueError("domain sampling manifest version must be 1 or 2")

        domains = {}
        for name in ("sim", "real"):
            domain = payload.get(name)
            if not isinstance(domain, dict):
                raise ValueError(f"domain sampling manifest missing {name} domain")
            required = {"episode_range", "probability", "frames"}
            if version == 2:
                required.add("current_samples")
            if set(domain) != required:
                raise ValueError(f"invalid {name} domain fields")
            domains[name] = domain

        sim_probability = domains["sim"]["probability"]
        real_probability = domains["real"]["probability"]
        if sim_probability != 0.75 or real_probability != 0.25:
            raise ValueError("domain sampling probability values must be exactly 0.75 and 0.25")

        sim_range = _validate_episode_range(domains["sim"]["episode_range"], "sim")
        real_range = _validate_episode_range(domains["real"]["episode_range"], "real")
        sim_frames = _validate_frame_total(domains["sim"]["frames"], "sim")
        real_frames = _validate_frame_total(domains["real"]["frames"], "real")
        if sim_range[1] > real_range[0] or real_range[1] <= sim_range[0]:
            raise ValueError("domain episode ranges must be ordered and non-overlapping")

        sim_current_indices = None
        real_current_indices = None
        if version == 2:
            sim_current_indices = _load_range_indices(
                domains["sim"]["current_samples"], sim_frames
            )
            real_current_indices = _load_anchor_indices(
                manifest_path, domains["real"]["current_samples"],
                lower_bound=sim_frames, upper_bound=sim_frames + real_frames,
            )

        return cls(
            sim_episode_range=sim_range,
            real_episode_range=real_range,
            sim_probability=float(sim_probability),
            real_probability=float(real_probability),
            sim_frames=sim_frames,
            real_frames=real_frames,
            sim_current_indices=sim_current_indices,
            real_current_indices=real_current_indices,
        )


def _load_range_indices(value: object, upper_bound: int) -> tuple[int, ...]:
    if not isinstance(value, dict) or set(value) != {"kind", "start", "stop", "count"}:
        raise ValueError("invalid sim current_samples range")
    start, stop, count = value.get("start"), value.get("stop"), value.get("count")
    if (
        value.get("kind") != "range"
        or any(isinstance(item, bool) or not isinstance(item, int)
               for item in (start, stop, count))
        or start < 0
        or stop <= start
        or stop > upper_bound
        or count != stop - start
    ):
        raise ValueError("invalid sim current_samples range")
    return tuple(range(start, stop))


def _load_anchor_indices(
    manifest_path: Path, value: object, *, lower_bound: int, upper_bound: int
) -> tuple[int, ...]:
    if not isinstance(value, dict) or set(value) != {"kind", "path", "count", "sha256"}:
        raise ValueError("invalid real current_samples file")
    relative = value.get("path")
    count = value.get("count")
    digest = value.get("sha256")
    if (
        value.get("kind") != "file"
        or not isinstance(relative, str)
        or not relative
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("invalid real current_samples file")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("real current_samples path escapes manifest directory")
    base = manifest_path.parent.resolve()
    anchor_path = manifest_path.parent / relative_path
    cursor = manifest_path.parent
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("real current_samples path must not contain symlinks")
    try:
        resolved = anchor_path.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as exc:
        raise ValueError("real current_samples path is invalid or escapes manifest directory") from exc
    if not resolved.is_file():
        raise ValueError("real current_samples path must be a regular file")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError("real current_samples file cannot be read") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("real current_samples SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid real current_samples JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("count") != count
        or not isinstance(payload.get("records"), list)
        or len(payload["records"]) != count
    ):
        raise ValueError("real current_samples count or schema mismatch")
    indices: list[int] = []
    for record in payload["records"]:
        required_fields = {
            "episode_index", "source_frame_index", "destination_frame_index", "index"
        }
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError("invalid real current_samples anchor record")
        if any(
            isinstance(record[field], bool)
            or not isinstance(record[field], int)
            or record[field] < 0
            for field in required_fields
        ):
            raise ValueError("invalid real current_samples anchor record")
        index = record["index"]
        if (
            index < lower_bound
            or index >= upper_bound
        ):
            raise ValueError("real current_samples index is outside the real frame range")
        indices.append(index)
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("real current_samples indices must be unique and strictly increasing")
    return tuple(indices)


def _validate_episode_range(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"invalid {name} episode_range")
    start, end = value
    if start < 0 or end <= start:
        raise ValueError(f"invalid {name} episode_range: expected a non-empty half-open range")
    return int(start), int(end)


def _validate_frame_total(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {name} frames total")
    return int(value)


class FixedRatioDomainSampler(Sampler[int]):
    """Yield exactly three simulation and one real index in every four samples.

    Indices are sampled with replacement.  A sampler instance advances its local
    epoch only after its iterator has been consumed completely, which makes
    interrupted epochs repeatable and keeps the audit a completed-epoch audit.
    """

    def __init__(
        self,
        sim_indices: Iterable[int],
        real_indices: Iterable[int],
        epoch_size: int = 4096,
        seed: int = 1000,
        audit_path: str | os.PathLike[str] | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        offset: int = 0,
        dataset_size: int | None = None,
    ) -> None:
        if dataset_size is not None and (
            isinstance(dataset_size, bool)
            or not isinstance(dataset_size, numbers.Integral)
            or dataset_size <= 0
        ):
            raise ValueError("dataset_size must be a positive integer")
        self.sim_indices = _materialize_indices(
            sim_indices, "sim_indices", upper_bound=dataset_size
        )
        self.real_indices = _materialize_indices(
            real_indices, "real_indices", upper_bound=dataset_size
        )
        if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size <= 0:
            raise ValueError("epoch_size must be a positive integer")
        if epoch_size % 4:
            raise ValueError("epoch_size must be divisible by four")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        local_length = (epoch_size - rank + world_size - 1) // world_size
        if isinstance(offset, bool) or not isinstance(offset, int) or not (
            0 <= offset < local_length if local_length else offset == 0
        ):
            raise ValueError("offset must be in [0, local sampler length)")
        self.epoch_size = epoch_size
        self.seed = seed
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.offset = offset
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        local_length = (self.epoch_size - self.rank + self.world_size - 1) // self.world_size
        return local_length - self.offset

    def set_epoch(self, epoch: int) -> "FixedRatioDomainSampler":
        """Set the deterministic epoch and reset its local sequence offset."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch
        self.offset = 0
        return self

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        global_values: list[int] = []
        blocks = self.epoch_size // 4
        for _ in range(blocks):
            sim_draws = torch.randint(
                len(self.sim_indices), (3,), generator=generator, device="cpu"
            ).tolist()
            real_draw = int(torch.randint(
                len(self.real_indices), (1,), generator=generator, device="cpu"
            ).item())
            block = [self.sim_indices[index] for index in sim_draws]
            block.append(self.real_indices[real_draw])
            order = torch.randperm(4, generator=generator, device="cpu").tolist()
            global_values.extend(block[index] for index in order)

        values = global_values[self.rank :: self.world_size][self.offset :]
        try:
            start_offset = self.offset
            for index, value in enumerate(values):
                # Update before yielding so an interrupted consumer can persist
                # the exact next position even if it closes the generator.
                self.offset = start_offset + index + 1
                if index == len(values) - 1:
                    self.epoch += 1
                    self.offset = 0
                    if self.rank == 0 and self.audit_path is not None:
                        self._update_audit(sim_samples=blocks * 3, real_samples=blocks)
                yield value
        except GeneratorExit:
            return

    def _update_audit(self, *, sim_samples: int, real_samples: int) -> None:
        assert self.audit_path is not None
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        cumulative = {"sim_samples": 0, "real_samples": 0, "epochs": 0}
        if self.audit_path.exists():
            try:
                with self.audit_path.open() as stream:
                    previous = json.load(stream)
                if not isinstance(previous, dict):
                    raise ValueError
                for key in cumulative:
                    value = previous.get(key, 0)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError
                    cumulative[key] = value
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid sampling audit: {self.audit_path}") from exc
        cumulative["sim_samples"] += sim_samples
        cumulative["real_samples"] += real_samples
        cumulative["epochs"] += 1

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.audit_path.name}.",
            suffix=".tmp",
            dir=self.audit_path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(cumulative, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.audit_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def _materialize_indices(
    indices: Iterable[int], name: str, upper_bound: int | None = None
) -> list[int]:
    try:
        result = []
        for index in indices:
            if isinstance(index, torch.Tensor):
                if index.ndim != 0:
                    raise ValueError
                index = index.item()
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                raise ValueError
            index = int(index)
            if index < 0 or (upper_bound is not None and index >= upper_bound):
                raise ValueError
            result.append(index)
    except (TypeError, ValueError) as exc:
        qualifier = " in range" if upper_bound is not None else ""
        raise ValueError(f"{name} must contain integer indices{qualifier}") from exc
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


__all__ = ["DomainSamplingSpec", "FixedRatioDomainSampler"]

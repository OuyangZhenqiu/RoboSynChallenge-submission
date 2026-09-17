"""Auditable, atomic telemetry for AMP optimizer steps."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

GRAD_NORM_CLASSES = ("finite", "nan", "posinf", "neginf")


def classify_grad_norm(value: float) -> str:
    """Return the JSON-safe class of a gradient norm."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "posinf"
    if value == -math.inf:
        return "neginf"
    return "finite"


def inprogress_path_for(destination: Path) -> Path:
    destination = Path(destination)
    return destination.with_name(f".{destination.name}.inprogress")


def summary_path_for(destination: Path) -> Path:
    destination = Path(destination)
    return destination.with_name(f"{destination.name}.summary.json")


class AmpTelemetryRecorder:
    """Append AMP step evidence and atomically publish it on success."""

    def __init__(self, destination: Path, *, candidate: str, log_freq: int):
        self.destination = Path(destination)
        self.inprogress_path = inprogress_path_for(self.destination)
        self.summary_path = summary_path_for(self.destination)
        self.validate_destination(self.destination)
        if log_freq <= 0:
            raise ValueError("log_freq must be positive")

        self.candidate = candidate
        self.log_freq = log_freq
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.inprogress_path.open("x", encoding="utf-8")
        self._state = "open"
        self._stream_closed = False
        self._attempted_steps = 0
        self._optimizer_step_called = 0
        self._amp_steps_skipped = 0
        self._skipped_step_numbers: list[int] = []
        self._current_consecutive_skips = 0
        self._maximum_consecutive_skips = 0
        self._gradient_class_counts = {name: 0 for name in GRAD_NORM_CLASSES}
        self._nonfinite_norm_without_skip = 0
        self._skip_without_found_inf = 0
        self._amp_scale_min = math.inf
        self._amp_scale_max = -math.inf
        self._last_attempted_step: int | None = None

    @staticmethod
    def validate_destination(destination: Path) -> None:
        destination = Path(destination)
        for path in (
            destination,
            inprogress_path_for(destination),
            summary_path_for(destination),
        ):
            if path.exists():
                raise FileExistsError(f"telemetry output is occupied: {path}")

    def append(self, record: dict[str, Any]) -> None:
        self._require_open("append")

        attempted_step = record["attempted_step"]
        if type(attempted_step) is not int or attempted_step <= 0:
            raise ValueError("attempted_step must be a positive integer")
        if (
            self._last_attempted_step is not None
            and attempted_step != self._last_attempted_step + 1
        ):
            raise ValueError(
                "attempted_step must be consecutive; "
                f"expected {self._last_attempted_step + 1}, got {attempted_step}"
            )

        serialized_record = dict(record)
        grad_norm = float(serialized_record["grad_norm_preclip"])
        grad_class = serialized_record["grad_norm_class"]
        if grad_class not in self._gradient_class_counts:
            raise ValueError(f"unknown grad_norm_class: {grad_class}")
        if not math.isfinite(grad_norm):
            serialized_record["grad_norm_preclip"] = None

        optimizer_called = bool(serialized_record["optimizer_step_called"])
        skipped = bool(serialized_record["amp_step_skipped"])
        found_inf_total = float(serialized_record["found_inf_total"])
        scale_before = float(serialized_record["amp_scale_before"])
        scale_after = float(serialized_record["amp_scale_after"])

        line = json.dumps(
            serialized_record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._stream.write(f"{line}\n")
        self._stream.flush()
        if attempted_step % self.log_freq == 0:
            os.fsync(self._stream.fileno())

        self._last_attempted_step = attempted_step
        self._attempted_steps += 1
        self._optimizer_step_called += int(optimizer_called)
        self._amp_steps_skipped += int(skipped)
        self._gradient_class_counts[grad_class] += 1
        if skipped:
            self._skipped_step_numbers.append(attempted_step)
            self._current_consecutive_skips += 1
            self._maximum_consecutive_skips = max(
                self._maximum_consecutive_skips,
                self._current_consecutive_skips,
            )
        else:
            self._current_consecutive_skips = 0
        if grad_class != "finite" and not skipped:
            self._nonfinite_norm_without_skip += 1
        if skipped and found_inf_total == 0:
            self._skip_without_found_inf += 1
        self._amp_scale_min = min(self._amp_scale_min, scale_before, scale_after)
        self._amp_scale_max = max(self._amp_scale_max, scale_before, scale_after)

    def abort(self) -> None:
        if self._state != "open":
            return
        self._close_stream()
        self._state = "aborted"

    def finalize(self) -> dict[str, Any]:
        self._require_open("finalize")
        if self._attempted_steps == 0:
            raise ValueError("telemetry requires at least one record before finalize")

        self._sync_stream()
        digest = hashlib.sha256(self.inprogress_path.read_bytes()).hexdigest()
        summary = self._build_summary(digest)
        serialized_summary = json.dumps(
            summary,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        summary_temp_path = self._write_summary_temp(serialized_summary)

        try:
            self._close_stream()
            try:
                os.replace(self.inprogress_path, self.destination)
                try:
                    os.replace(summary_temp_path, self.summary_path)
                except Exception as publication_error:
                    try:
                        os.replace(self.destination, self.inprogress_path)
                    except Exception as rollback_error:
                        publication_error.add_note(
                            "failed to roll finalized JSONL back to .inprogress: "
                            f"{rollback_error}"
                        )
                    raise
            except Exception:
                self._state = "aborted"
                raise
        finally:
            if summary_temp_path.exists():
                summary_temp_path.unlink()

        self._state = "finalized"
        return summary

    def _build_summary(self, digest: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate": self.candidate,
            "attempted_steps": self._attempted_steps,
            "optimizer_step_called": self._optimizer_step_called,
            "amp_steps_skipped": self._amp_steps_skipped,
            "skipped_step_numbers": self._skipped_step_numbers,
            "maximum_consecutive_skips": self._maximum_consecutive_skips,
            "gradient_class_counts": self._gradient_class_counts,
            "nonfinite_norm_without_skip": self._nonfinite_norm_without_skip,
            "skip_without_found_inf": self._skip_without_found_inf,
            "amp_scale_min": self._amp_scale_min,
            "amp_scale_max": self._amp_scale_max,
            "jsonl_sha256": digest,
        }

    def _close_stream(self) -> None:
        if self._stream_closed:
            return
        self._sync_stream()
        self._stream.close()
        self._stream_closed = True

    def _sync_stream(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def _require_open(self, operation: str) -> None:
        if self._state != "open":
            raise RuntimeError(
                f"cannot {operation} telemetry recorder in {self._state} state"
            )

    def _write_summary_temp(self, serialized: str) -> Path:
        temp_path = None
        completed = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="x",
                encoding="utf-8",
                dir=self.summary_path.parent,
                prefix=f".{self.summary_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                stream.write(f"{serialized}\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed = True
            return temp_path
        finally:
            if not completed and temp_path is not None and temp_path.exists():
                temp_path.unlink()


def _restore_instance_attribute(instance: Any, name: str, previous: Any) -> None:
    if previous is _MISSING:
        delattr(instance, name)
    else:
        setattr(instance, name, previous)


_MISSING = object()


def observe_update_policy(original_update_policy, recorder):
    """Decorate LeRobot's update function with non-mutating AMP observation."""

    signature = inspect.signature(original_update_policy)

    @functools.wraps(original_update_policy)
    def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        train_metrics = bound.arguments["train_metrics"]
        optimizer = bound.arguments["optimizer"]
        grad_scaler = bound.arguments["grad_scaler"]
        attempted_step = int(train_metrics.steps) + 1
        amp_scale_before = float(grad_scaler.get_scale())

        scaler_step_calls = 0
        optimizer_step_calls = 0
        found_inf_snapshots: list[dict[str, float]] = []

        scaler_instance_step = vars(grad_scaler).get("step", _MISSING)
        original_scaler_step = grad_scaler.step

        def observed_scaler_step(*step_args, **step_kwargs):
            nonlocal scaler_step_calls, optimizer_step_calls
            scaler_step_calls += 1
            step_optimizer = step_args[0] if step_args else step_kwargs.get("optimizer")
            found_inf = grad_scaler._found_inf_per_device(step_optimizer)
            found_inf_snapshots.append(
                {
                    str(device): float(value.item())
                    for device, value in found_inf.items()
                }
            )

            optimizer_instance_step = vars(step_optimizer).get("step", _MISSING)
            original_optimizer_step = step_optimizer.step

            def observed_optimizer_step(*optimizer_args, **optimizer_kwargs):
                nonlocal optimizer_step_calls
                optimizer_step_calls += 1
                return original_optimizer_step(*optimizer_args, **optimizer_kwargs)

            setattr(step_optimizer, "step", observed_optimizer_step)
            try:
                return original_scaler_step(*step_args, **step_kwargs)
            finally:
                _restore_instance_attribute(
                    step_optimizer,
                    "step",
                    optimizer_instance_step,
                )

        setattr(grad_scaler, "step", observed_scaler_step)
        try:
            result = original_update_policy(*args, **kwargs)
        finally:
            _restore_instance_attribute(grad_scaler, "step", scaler_instance_step)

        if scaler_step_calls != 1:
            raise RuntimeError(
                "expected exactly one scaler.step call; "
                f"observed {scaler_step_calls}"
            )
        found_inf_by_device = found_inf_snapshots[0]
        if not found_inf_by_device:
            raise RuntimeError("missing found_inf evidence")
        if any(
            not math.isfinite(value) or value < 0
            for value in found_inf_by_device.values()
        ):
            raise RuntimeError("found_inf must be finite and non-negative")
        found_inf_total = sum(found_inf_by_device.values())
        if found_inf_total > 0:
            if optimizer_step_calls != 0:
                raise RuntimeError("optimizer.step called despite found_inf")
        elif optimizer_step_calls != 1:
            raise RuntimeError(
                "expected exactly one optimizer.step call when found_inf is zero; "
                f"observed {optimizer_step_calls}"
            )

        grad_norm = float(train_metrics.grad_norm.val)
        record = {
            "attempted_step": attempted_step,
            "loss": float(train_metrics.loss.val),
            "grad_norm_preclip": grad_norm,
            "grad_norm_class": classify_grad_norm(grad_norm),
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": float(grad_scaler.get_scale()),
            "found_inf_by_device": found_inf_by_device,
            "found_inf_total": found_inf_total,
            "optimizer_step_called": optimizer_step_calls == 1,
            "amp_step_skipped": optimizer_step_calls == 0,
            "lr": float(train_metrics.lr.val),
        }
        recorder.append(record)
        return result

    return observed

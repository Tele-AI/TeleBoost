#!/usr/bin/env python3
# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared safety helpers for the standalone real-checkpoint smoke tools."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


class SmokePreflightError(RuntimeError):
    """A prerequisite or safety contract failed before model loading."""


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GPUStatus:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    utilization_percent: int
    processes: tuple[GPUProcess, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise SmokePreflightError(f"required file is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokePreflightError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SmokePreflightError(f"expected a JSON object in {path}")
    return value


def require_files(root: Path, relative_paths: Sequence[str]) -> None:
    missing = [str(root / relative) for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise SmokePreflightError("required model files are missing: " + ", ".join(missing))


def require_weight_index(root: Path, relative_index: str) -> tuple[str, ...]:
    index_path = root / relative_index
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SmokePreflightError(f"weight_map is empty or invalid: {index_path}")
    shards = tuple(sorted({str(name) for name in weight_map.values()}))
    shard_root = index_path.parent
    missing = [str(shard_root / shard) for shard in shards if not (shard_root / shard).is_file()]
    if missing:
        raise SmokePreflightError("model index references missing shards: " + ", ".join(missing))
    return shards


def require_versions(
    expected: Mapping[str, str],
    *,
    version_getter: Callable[[str], str] = version,
) -> dict[str, str]:
    installed: dict[str, str] = {}
    failures: list[str] = []
    for distribution, wanted in expected.items():
        try:
            found = version_getter(distribution)
        except PackageNotFoundError:
            found = "<missing>"
        except Exception as exc:
            raise SmokePreflightError(f"cannot inspect {distribution}: {exc}") from exc
        installed[distribution] = found
        if found != wanted:
            failures.append(f"{distribution}={found} (expected {wanted})")
    if failures:
        raise SmokePreflightError("version contract failed: " + ", ".join(failures))
    return installed


def _numeric_field(value: str, *, field: str) -> int:
    match = re.search(r"-?\d+", value)
    if match is None:
        raise SmokePreflightError(f"nvidia-smi returned invalid {field}: {value!r}")
    return int(match.group(0))


def parse_gpu_rows(gpu_output: str, process_output: str, *, gpu_index: int) -> GPUStatus:
    rows = list(csv.reader(line for line in gpu_output.splitlines() if line.strip()))
    selected = None
    for row in rows:
        if len(row) < 5:
            raise SmokePreflightError(f"invalid nvidia-smi GPU row: {row!r}")
        if _numeric_field(row[0], field="GPU index") == gpu_index:
            selected = row
            break
    if selected is None:
        available = ", ".join(row[0].strip() for row in rows) or "none"
        raise SmokePreflightError(f"GPU {gpu_index} does not exist; available indices: {available}")

    gpu_uuid = selected[1].strip()
    processes: list[GPUProcess] = []
    for row in csv.reader(line for line in process_output.splitlines() if line.strip()):
        if len(row) < 4:
            raise SmokePreflightError(f"invalid nvidia-smi process row: {row!r}")
        if row[0].strip() != gpu_uuid:
            continue
        processes.append(
            GPUProcess(
                pid=_numeric_field(row[1], field="process PID"),
                process_name=row[2].strip(),
                used_memory_mib=_numeric_field(row[3], field="process memory"),
            )
        )

    return GPUStatus(
        index=gpu_index,
        uuid=gpu_uuid,
        name=selected[2].strip(),
        memory_used_mib=_numeric_field(selected[3], field="GPU memory"),
        utilization_percent=_numeric_field(selected[4], field="GPU utilization"),
        processes=tuple(processes),
    )


def query_gpu_status(
    gpu_index: int,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GPUStatus:
    commands = (
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    outputs: list[str] = []
    for command in commands:
        try:
            completed = command_runner(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise SmokePreflightError("nvidia-smi is required for the GPU safety check") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise SmokePreflightError(f"nvidia-smi safety check failed: {detail}") from exc
        outputs.append(completed.stdout)
    return parse_gpu_rows(outputs[0], outputs[1], gpu_index=gpu_index)


def assert_gpu_idle(
    status: GPUStatus,
    *,
    max_existing_memory_mib: int = 256,
    max_existing_utilization_percent: int = 5,
) -> None:
    reasons: list[str] = []
    if status.processes:
        process_text = ", ".join(f"pid={process.pid} name={process.process_name} memory={process.used_memory_mib}MiB" for process in status.processes)
        reasons.append(f"compute processes: {process_text}")
    if status.memory_used_mib > max_existing_memory_mib:
        reasons.append(f"memory.used={status.memory_used_mib}MiB > {max_existing_memory_mib}MiB")
    if status.utilization_percent > max_existing_utilization_percent:
        reasons.append(f"utilization.gpu={status.utilization_percent}% > {max_existing_utilization_percent}%")
    if reasons:
        raise SmokePreflightError(f"refusing to use busy GPU {status.index}: " + "; ".join(reasons))


@contextmanager
def isolated_cache_environment(
    *,
    prefix: str,
    gpu_index: int,
    extra_environment: Mapping[str, str] | None = None,
) -> Iterator[Path]:
    """Route mutable runtime caches to a unique /tmp tree and remove it."""

    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as temporary:
        root = Path(temporary)
        directories = {
            "VLLM_CACHE_ROOT": root / "vllm",
            "HF_HOME": root / "huggingface",
            "HF_HUB_CACHE": root / "huggingface" / "hub",
            "XDG_CACHE_HOME": root / "xdg",
            "TORCH_HOME": root / "torch",
            "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
            "TRITON_CACHE_DIR": root / "triton",
            "CUDA_CACHE_PATH": root / "cuda",
            "TMPDIR": root / "tmp",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        updates = {key: str(path) for key, path in directories.items()}
        updates.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "DIFFUSERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        if extra_environment:
            updates.update({key: str(value) for key, value in extra_environment.items()})

        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)
        try:
            yield root
        finally:
            for key, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value


def print_result(sentinel: str, payload: Mapping[str, Any]) -> None:
    print(f"{sentinel} {json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}", flush=True)

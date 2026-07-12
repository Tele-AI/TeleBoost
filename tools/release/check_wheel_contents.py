#!/usr/bin/env python3
"""Validate the public Wan-only TeleBoost sdist/wheel boundary.

The root wheel is intentionally checked together with the sdist it was built
from.  A wheel built directly from a working tree is not a releasable artifact:
setuptools can otherwise reuse a stale ``build/lib`` directory.

Usage:
    python tools/release/check_wheel_contents.py ROOT_SDIST ROOT_WHEEL
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import re
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


class ArtifactContractError(RuntimeError):
    """Raised when a release artifact violates the checked contract."""


# Private path/host markers as (length, sha256) so the checker ships in the
# sdist without carrying — or letting anyone reconstruct — the plaintexts.
# Regenerate an entry with: sha256(marker.encode()).hexdigest().
_PRIVATE_MARKER_HASHES = (
    (5, "7a53dc9a64fc7e6e19e3f89357153910d728c854f4a4b7a24f9a05f66966a2bd"),
    (7, "46dc943e3424a2254a109373bb5e9e7d016a71c4ad9f38d63991107e091bee56"),
    (15, "5283d94e51ef9fb2237cbfc0940198d8152d01653d14814ff83016b958a03da6"),
    (11, "d40e7c468762e3a3a9f838114a5230395cea51b0995ae4a70fc7b52a2bc80eea"),
    (10, "12075d2f576ed00e61871c2bdd253944ce14e262b4937afb0fcdda90a1c76671"),
    (6, "6293bd8b5131c952eb455e68db769f3db1d52510563c061fbf4a4d9d334447a7"),
    (15, "a4269d64d2a9055b8937a728524e36f454aaef7b2fd46630e0c0045470d23166"),
    (16, "77828ae5c2a07838ba433c06afbac3bc2e6b22079c7b8a3aa43e806f80700236"),
    (24, "1d340dcdcae6b6ad9fa4be0ff0d338412100a64caa4ce0cb44ef5aec4633407f"),
)
# Markers are path/host tokens, so only path/host-shaped runs need scanning.
_MARKER_CANDIDATE_RUN = re.compile(r"[/A-Za-z0-9_.\-]{4,}")


def _find_private_markers(text: str) -> list[str]:
    lengths = sorted({length for length, _ in _PRIVATE_MARKER_HASHES})
    digests = {digest for _, digest in _PRIVATE_MARKER_HASHES}
    found: list[str] = []
    for run_match in _MARKER_CANDIDATE_RUN.finditer(text):
        run = run_match.group(0)
        for length in lengths:
            for start in range(0, len(run) - length + 1):
                window = run[start : start + length]
                if hashlib.sha256(window.encode("utf-8")).hexdigest() in digests:
                    found.append(window)
    return found


_SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
)
_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "outputs",
        "tests",
        "third_party",
        "videos",
    }
)
_FORBIDDEN_TOOL_SUBTREES = frozenset({"diagnostics", "smoke"})
_PUBLIC_RECIPE_DIRS = frozenset(
    {
        "wan_bgpo_fsdp",
        "wan_dpo_teletron",
        "wan_grpo_fsdp",
        "wan_tempflow_fsdp",
        "wan_vipo_fsdp",
    }
)
_PUBLIC_ALGORITHM_ENTRIES = frozenset(
    {
        "README.md",
        "__init__.py",
        "bgpo.py",
        "grpo",
        "grpo_guard.py",
        "rollout_contract.py",
        "solver_contract.py",
        "tempflow",
        "vipo.py",
        "wan_transition.py",
    }
)
_PUBLIC_OPTIONAL_EXTRAS = frozenset({"train", "wan", "dpo", "flash", "reward", "vllm", "sglang", "test", "release"})
_PUBLIC_PACKAGE_SUBDIRECTORIES = (
    ("teleboost/models/", frozenset({"wan"})),
    ("teleboost/programs/", frozenset({"wan"})),
    ("teleboost/training/families/", frozenset({"wan"})),
    ("teleboost/config/presets/programs/", frozenset({"wan_grpo_fsdp"})),
)
_PUBLIC_FILE_COMPONENTS = (
    ("teleboost/cli/", frozenset({"__init__.py", "convert_checkpoint.py"})),
    ("teleboost/artifacts/", frozenset({"__init__.py", "wan_conversion.py"})),
)
_ROOT_NOTICE_MARKERS = (
    "NVIDIA Megatron-LM",
    "OpenCLIP",
    "ModelScope DiffSynth-Studio",
    "Tele-AI TeleTron",
    "Alibaba Wan",
)
_ROOT_PROVENANCE_MARKERS = (
    "55ac7082517c3878ae653c07c09c534b8aed49f6",
    "ea7718f927b84e1b46ce057d3eae5ca4c9c41434",
    "451aab01161496fd68510e7682306eaf54ff97f2",
    "5be5c32fe4b240547a288afa4c29e3f81b6ef881",
    "204f899b6436fe2e1705a0b67c464b30b8137799",
    "OpenCLIP",
    "DiffSynth-Studio",
)
_CANONICAL_CONVERTER_COMMAND = "teleboost-convert-wan-to-teletron"
_CANONICAL_CONVERTER_TARGET = "teleboost.cli.convert_checkpoint:main"


def _normalized_member(raw_name: str) -> tuple[str, ...]:
    if not raw_name or "\\" in raw_name:
        raise ArtifactContractError(f"invalid archive member path: {raw_name!r}")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactContractError(f"unsafe archive member path: {raw_name!r}")
    return path.parts


def _zip_members(path: Path) -> tuple[set[str], dict[str, bytes]]:
    if not path.is_file():
        raise ArtifactContractError(f"wheel does not exist: {path}")
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactContractError(f"wheel contains duplicate member names: {path}")
            for info in infos:
                _normalized_member(info.filename.rstrip("/"))
                if info.flag_bits & 0x1:
                    raise ArtifactContractError(f"wheel contains encrypted member: {info.filename}")
                # Symlinks are neither needed nor portable in these wheels.
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ArtifactContractError(f"wheel contains symlink: {info.filename}")
            payloads = {info.filename: archive.read(info) for info in infos if not info.is_dir()}
    except BadZipFile as exc:
        raise ArtifactContractError(f"not a valid wheel/zip archive: {path}") from exc
    return set(names), payloads


def _sdist_members(path: Path) -> tuple[set[str], dict[str, bytes], str]:
    if not path.is_file():
        raise ArtifactContractError(f"sdist does not exist: {path}")
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            raw_names = [member.name.rstrip("/") for member in members]
            if len(raw_names) != len(set(raw_names)):
                raise ArtifactContractError(f"sdist contains duplicate member names: {path}")
            for member in members:
                _normalized_member(member.name.rstrip("/"))
                if not (member.isfile() or member.isdir()):
                    raise ArtifactContractError(f"sdist contains a link or special member: {member.name}")

            roots = {_normalized_member(name)[0] for name in raw_names}
            if len(roots) != 1:
                raise ArtifactContractError(f"sdist must have exactly one top-level directory, found {sorted(roots)}")
            root = roots.pop()
            prefix = f"{root}/"
            names: set[str] = set()
            payloads: dict[str, bytes] = {}
            for member in members:
                if member.name.rstrip("/") == root:
                    continue
                if not member.name.startswith(prefix):
                    raise ArtifactContractError(f"sdist member escaped root: {member.name}")
                relative = member.name[len(prefix) :]
                names.add(relative)
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ArtifactContractError(f"cannot read sdist member: {member.name}")
                    payloads[relative] = extracted.read()
    except (tarfile.ReadError, EOFError) as exc:
        raise ArtifactContractError(f"not a valid gzip sdist: {path}") from exc
    return names, payloads, root


def _one_metadata(names: set[str], payloads: dict[str, bytes]) -> tuple[str, str]:
    matches = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
    if len(matches) != 1:
        raise ArtifactContractError(f"expected exactly one .dist-info/METADATA, found {matches}")
    name = matches[0]
    return name, payloads[name].decode("utf-8", errors="strict")


def _require_members(names: set[str], required: set[str], *, artifact_name: str) -> None:
    missing = sorted(required - names)
    if missing:
        raise ArtifactContractError(f"{artifact_name} is missing required members: {missing}")


def _reject_forbidden_paths(names: set[str], *, artifact_name: str, allow_release_tools: bool = False) -> None:
    rejected: list[str] = []
    for name in names:
        parts = _normalized_member(name.rstrip("/"))
        if any(part in _FORBIDDEN_COMPONENTS or part.startswith("outputs.") for part in parts):
            rejected.append(name)
            continue
        if name.endswith((".pyc", ".pyo")):
            rejected.append(name)
            continue
        for index, part in enumerate(parts[:-1]):
            if part == "tools" and parts[index + 1] in _FORBIDDEN_TOOL_SUBTREES:
                rejected.append(name)
                break
            if not allow_release_tools and part == "tools" and parts[index + 1] == "release":
                rejected.append(name)
                break
        if len(parts) >= 2 and parts[0] == "tools" and parts[-1].endswith("_smoke.py"):
            rejected.append(name)

    if rejected:
        rejected = sorted(set(rejected))
        preview = rejected[:20]
        suffix = f" (+{len(rejected) - len(preview)} more)" if len(rejected) > len(preview) else ""
        raise ArtifactContractError(f"{artifact_name} contains forbidden paths: {preview}{suffix}")


def _reject_private_markers(payloads: dict[str, bytes], *, artifact_name: str) -> None:
    hits: list[tuple[str, str]] = []
    secret_hits: list[tuple[str, str]] = []
    for name, payload in payloads.items():
        text = payload.decode("utf-8", errors="ignore")
        for marker in _find_private_markers(text):
            hits.append((name, marker))
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                secret_hits.append((name, pattern.pattern))
    if hits:
        raise ArtifactContractError(f"{artifact_name} contains private/internal path markers: {hits[:20]}")
    if secret_hits:
        raise ArtifactContractError(f"{artifact_name} contains credential-shaped strings: {secret_hits[:20]}")


def _direct_components(names: set[str], prefix: str) -> set[str]:
    components = set()
    for name in names:
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :].rstrip("/")
        if remainder:
            components.add(remainder.split("/", 1)[0])
    return components


def _reject_out_of_scope_paths(names: set[str], *, artifact_name: str) -> None:
    violations = []
    for prefix, allowed_directories in _PUBLIC_PACKAGE_SUBDIRECTORIES:
        components = _direct_components(names, prefix)
        directories = {component for component in components if "." not in component}
        unexpected = sorted(directories - allowed_directories)
        if unexpected:
            violations.append(f"{prefix}: {unexpected}")

    recipe_components = _direct_components(names, "recipes/")
    recipe_directories = {component for component in recipe_components if "." not in component}
    unexpected_recipes = sorted(recipe_directories - _PUBLIC_RECIPE_DIRS)
    if unexpected_recipes:
        violations.append(f"recipes/: {unexpected_recipes}")

    algorithm_components = _direct_components(names, "teleboost/algorithms/")
    unexpected_algorithms = sorted(algorithm_components - _PUBLIC_ALGORITHM_ENTRIES)
    if unexpected_algorithms:
        violations.append(f"teleboost/algorithms/: {unexpected_algorithms}")

    for prefix, allowed in _PUBLIC_FILE_COMPONENTS:
        components = _direct_components(names, prefix)
        unexpected = sorted(components - allowed)
        if unexpected:
            violations.append(f"{prefix}: {unexpected}")

    if violations:
        raise ArtifactContractError(f"{artifact_name} exceeds the public source allowlist: {violations}")


def _require_text_markers(
    payload: bytes,
    markers: tuple[str, ...],
    *,
    member_name: str,
    artifact_name: str,
) -> None:
    text = payload.decode("utf-8", errors="strict")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise ArtifactContractError(f"{artifact_name} {member_name} is missing required attribution markers: {missing}")


def _validate_root_attributions(
    payloads: dict[str, bytes],
    *,
    notice_name: str,
    provenance_name: str,
    megatron_license_name: str,
    openclip_license_name: str,
    artifact_name: str,
) -> None:
    _require_text_markers(
        payloads[notice_name],
        _ROOT_NOTICE_MARKERS,
        member_name=notice_name,
        artifact_name=artifact_name,
    )
    _require_text_markers(
        payloads[provenance_name],
        _ROOT_PROVENANCE_MARKERS,
        member_name=provenance_name,
        artifact_name=artifact_name,
    )
    _require_text_markers(
        payloads[megatron_license_name],
        ("Redistribution and use in source and binary forms", "NVIDIA CORPORATION"),
        member_name=megatron_license_name,
        artifact_name=artifact_name,
    )
    _require_text_markers(
        payloads[openclip_license_name],
        ("Permission is hereby granted", "Gabriel Ilharco"),
        member_name=openclip_license_name,
        artifact_name=artifact_name,
    )


def _entry_points(names: set[str], payloads: dict[str, bytes]) -> dict[str, str]:
    matches = sorted(name for name in names if name.endswith(".dist-info/entry_points.txt"))
    if len(matches) != 1:
        raise ArtifactContractError(f"expected exactly one .dist-info/entry_points.txt, found {matches}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(payloads[matches[0]].decode("utf-8", errors="strict"))
    return dict(parser["console_scripts"]) if parser.has_section("console_scripts") else {}


def _validate_converter_script(scripts: dict[str, str], *, artifact_name: str) -> None:
    actual = scripts.get(_CANONICAL_CONVERTER_COMMAND)
    if actual != _CANONICAL_CONVERTER_TARGET:
        raise ArtifactContractError(f"{artifact_name} must map {_CANONICAL_CONVERTER_COMMAND!r} to {_CANONICAL_CONVERTER_TARGET!r}, got {actual!r}")


def validate_root_wheel(path: Path) -> int:
    names, payloads = _zip_members(path)
    metadata_name, metadata = _one_metadata(names, payloads)
    if "Name: teleboost\n" not in metadata:
        raise ArtifactContractError(f"root wheel has unexpected metadata: {metadata_name}")
    if "License-Expression: Apache-2.0\n" not in metadata:
        raise ArtifactContractError("root wheel must declare License-Expression: Apache-2.0")
    required_license_metadata = (
        "License-File: LICENSES/Megatron-LM-BSD-3-Clause.txt\n",
        "License-File: LICENSES/OpenCLIP-MIT.txt\n",
    )
    missing_license_metadata = [line.rstrip() for line in required_license_metadata if line not in metadata]
    if missing_license_metadata:
        raise ArtifactContractError(f"root wheel metadata is missing third-party license files: {missing_license_metadata}")

    _reject_forbidden_paths(names, artifact_name="root wheel")
    _reject_out_of_scope_paths(names, artifact_name="root wheel")
    _require_members(
        names,
        {
            "teleboost/datasets/preprocessing/wan.py",
            "teleboost/__init__.py",
            "teleboost/artifacts/wan_conversion.py",
        },
        artifact_name="root wheel",
    )
    scripts = _entry_points(names, payloads)
    _validate_converter_script(scripts, artifact_name="root wheel")

    license_root = metadata_name.removesuffix("METADATA") + "licenses/"
    _require_members(
        names,
        {
            f"{license_root}LICENSE",
            f"{license_root}LICENSES/Megatron-LM-BSD-3-Clause.txt",
            f"{license_root}LICENSES/OpenCLIP-MIT.txt",
            f"{license_root}NOTICE",
            f"{license_root}THIRD_PARTY_PROVENANCE.md",
        },
        artifact_name="root wheel",
    )
    _validate_root_attributions(
        payloads,
        notice_name=f"{license_root}NOTICE",
        provenance_name=f"{license_root}THIRD_PARTY_PROVENANCE.md",
        megatron_license_name=f"{license_root}LICENSES/Megatron-LM-BSD-3-Clause.txt",
        openclip_license_name=f"{license_root}LICENSES/OpenCLIP-MIT.txt",
        artifact_name="root wheel",
    )

    unexpected_top_levels = sorted({name.split("/", 1)[0] for name in names if "/" in name and not name.split("/", 1)[0].endswith(".dist-info") and name.split("/", 1)[0] not in {"teleboost"}})
    if unexpected_top_levels:
        raise ArtifactContractError(f"root wheel contains unexpected top-level packages: {unexpected_top_levels}")
    _reject_private_markers(payloads, artifact_name="root wheel")
    return len(names)


def validate_root_sdist(path: Path) -> int:
    names, payloads, archive_root = _sdist_members(path)
    if not archive_root.startswith("teleboost-"):
        raise ArtifactContractError(f"root sdist has unexpected archive root: {archive_root}")
    _reject_forbidden_paths(names, artifact_name="root sdist", allow_release_tools=True)
    _reject_out_of_scope_paths(names, artifact_name="root sdist")
    required = {
        "INSTALL.md",
        "LICENSE",
        "LICENSES/Megatron-LM-BSD-3-Clause.txt",
        "LICENSES/OpenCLIP-MIT.txt",
        "MANIFEST.in",
        "MODEL_AND_DATA_LICENSES.md",
        "NOTICE",
        "PKG-INFO",
        "README.md",
        "constraints/upstreams/flash-attn-3.txt",
        "constraints/upstreams/megatron-lm.txt",
        "constraints/upstreams/verl.txt",
        "SECURITY.md",
        "SUPPORT_MATRIX.md",
        "THIRD_PARTY_PROVENANCE.md",
        "constraints/release.txt",
        "pyproject.toml",
        "requirements.txt",
        "teleboost/artifacts/wan_conversion.py",
        "tools/release/README.md",
        "tools/release/build_artifacts.py",
        "tools/release/check_wheel_contents.py",
        "tools/install_flash_attn_3.sh",
        "tools/install_verl.sh",
    }
    required.update(
        {
            "recipes/wan_bgpo_fsdp/run.sh",
            "recipes/wan_dpo_teletron/run.sh",
            "recipes/wan_grpo_fsdp/run.sh",
            "recipes/wan_tempflow_fsdp/run.sh",
            "recipes/wan_vipo_fsdp/run.sh",
        }
    )
    _require_members(names, required, artifact_name="root sdist")
    _validate_root_attributions(
        payloads,
        notice_name="NOTICE",
        provenance_name="THIRD_PARTY_PROVENANCE.md",
        megatron_license_name="LICENSES/Megatron-LM-BSD-3-Clause.txt",
        openclip_license_name="LICENSES/OpenCLIP-MIT.txt",
        artifact_name="root sdist",
    )

    metadata = payloads["PKG-INFO"].decode("utf-8", errors="strict")
    if "Name: teleboost\n" not in metadata or "License-Expression: Apache-2.0\n" not in metadata:
        raise ArtifactContractError("root sdist PKG-INFO has unexpected name or license")
    for line in (
        "License-File: LICENSES/Megatron-LM-BSD-3-Clause.txt\n",
        "License-File: LICENSES/OpenCLIP-MIT.txt\n",
    ):
        if line not in metadata:
            raise ArtifactContractError(f"root sdist PKG-INFO is missing third-party license metadata: {line.rstrip()}")

    pyproject = tomllib.loads(payloads["pyproject.toml"].decode("utf-8", errors="strict"))
    scripts = pyproject.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        raise ArtifactContractError("root sdist pyproject project.scripts must be a table")
    project = pyproject.get("project", {})
    extras = project.get("optional-dependencies", {})
    if set(extras) != _PUBLIC_OPTIONAL_EXTRAS:
        raise ArtifactContractError(f"root sdist dependency extras do not match the public allowlist: {sorted(extras)}")
    _validate_converter_script(scripts, artifact_name="root sdist")
    support_matrix = payloads["SUPPORT_MATRIX.md"].decode("utf-8", errors="strict")
    if _CANONICAL_CONVERTER_COMMAND not in support_matrix:
        raise ArtifactContractError("SUPPORT_MATRIX.md must document the canonical converter command")

    readme = payloads["README.md"].decode("utf-8", errors="strict")
    if "SUPPORT_MATRIX.md" not in readme:
        raise ArtifactContractError("README.md must link to SUPPORT_MATRIX.md")
    install = payloads["INSTALL.md"].decode("utf-8", errors="strict")
    release_readme = payloads["tools/release/README.md"].decode("utf-8", errors="strict")
    if "tools/release/build_artifacts.py" not in install:
        raise ArtifactContractError("INSTALL.md must document the hermetic release builder")
    for command in (
        "tools/release/build_artifacts.py",
        "tools/release/check_wheel_contents.py",
    ):
        if command not in release_readme:
            raise ArtifactContractError(f"tools/release/README.md must document {command}")
    _reject_private_markers(payloads, artifact_name="root sdist")
    return len(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_sdist", type=Path)
    parser.add_argument("root_wheel", type=Path)
    args = parser.parse_args()

    sdist_count = validate_root_sdist(args.root_sdist)
    root_count = validate_root_wheel(args.root_wheel)
    print(f"release artifact contract passed: sdist={sdist_count} members, root={root_count} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

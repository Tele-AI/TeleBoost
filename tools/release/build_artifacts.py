#!/usr/bin/env python3
"""Build and validate the public Wan-only TeleBoost release artifacts.

The public branch, source distribution, and wheel share one source boundary.
This builder never removes or rewrites family code while staging: a checkout
that is not already Wan-only fails before packaging. The wheel is always built
from a freshly extracted sdist so stale checkout build products cannot leak in.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath


_ROOT_FILES = (
    "INSTALL.md",
    "LICENSE",
    "MANIFEST.in",
    "MODEL_AND_DATA_LICENSES.md",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "SUPPORT_MATRIX.md",
    "THIRD_PARTY_PROVENANCE.md",
    "pyproject.toml",
    "requirements.txt",
)
_ROOT_DIRS = ("constraints", "LICENSES", "recipes", "teleboost", "tools")
_PUBLIC_PROGRAMS = frozenset(
    {
        "wan.grpo.fsdp",
        "wan.bgpo.fsdp",
        "wan.vipo.fsdp",
        "wan.tempflow.fsdp",
        "wan.dpo.teletron",
    }
)
_PUBLIC_RECIPE_DIRS = frozenset(name.replace(".", "_") for name in _PUBLIC_PROGRAMS)
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
_PUBLIC_SUBDIRECTORIES = (
    ("teleboost/models", frozenset({"wan"})),
    ("teleboost/programs", frozenset({"wan"})),
    ("teleboost/training/families", frozenset({"wan"})),
    ("teleboost/config/presets/programs", frozenset({"wan_grpo_fsdp"})),
    ("third_party", frozenset({"wan", "raft", "Videophy"})),
)
_PUBLIC_FILE_ENTRIES = (
    ("teleboost/cli", frozenset({"__init__.py", "convert_checkpoint.py"})),
    ("teleboost/artifacts", frozenset({"__init__.py", "wan_conversion.py"})),
)
_IGNORED_NAMES = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
    }
)


class ReleaseBuildError(RuntimeError):
    """Raised when the hermetic release build cannot satisfy its contract."""


def _clean_build_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.setdefault("SOURCE_DATE_EPOCH", "946684800")
    return env


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise ReleaseBuildError(f"release command failed with exit code {exc.returncode}: {command}") from exc


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_NAMES or name.endswith((".pyc", ".pyo", ".egg-info"))}


def _reject_source_symlinks(source: Path) -> None:
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        preview = [str(path) for path in symlinks[:20]]
        raise ReleaseBuildError(f"release source contains symlinks: {preview}")


def _visible_child_directories(root: Path) -> set[str]:
    return {child.name for child in root.iterdir() if child.is_dir() and child.name not in _IGNORED_NAMES}


def _builtin_program_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_program":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
    return names


def _validate_public_source_tree(repo_root: Path) -> None:
    """Require the checkout itself to satisfy the public Wan-only boundary."""

    missing = [name for name in (*_ROOT_FILES, *_ROOT_DIRS) if not (repo_root / name).exists()]
    if missing:
        raise ReleaseBuildError(f"required release sources are missing: {missing}")

    for relative, allowed in _PUBLIC_SUBDIRECTORIES:
        actual = _visible_child_directories(repo_root / relative)
        if actual != allowed:
            raise ReleaseBuildError(f"public source directory contract failed for {relative}: expected {sorted(allowed)}, found {sorted(actual)}")

    for relative, allowed in _PUBLIC_FILE_ENTRIES:
        actual = {entry.name for entry in (repo_root / relative).iterdir() if entry.name not in _IGNORED_NAMES}
        if actual != allowed:
            raise ReleaseBuildError(f"public source file contract failed for {relative}: expected {sorted(allowed)}, found {sorted(actual)}")

    recipe_dirs = _visible_child_directories(repo_root / "recipes")
    if recipe_dirs != _PUBLIC_RECIPE_DIRS:
        raise ReleaseBuildError(f"public recipe contract failed: expected {sorted(_PUBLIC_RECIPE_DIRS)}, found {sorted(recipe_dirs)}")

    algorithm_entries = {entry.name for entry in (repo_root / "teleboost/algorithms").iterdir() if entry.name not in _IGNORED_NAMES}
    if algorithm_entries != _PUBLIC_ALGORITHM_ENTRIES:
        raise ReleaseBuildError(f"public algorithm contract failed: expected {sorted(_PUBLIC_ALGORITHM_ENTRIES)}, found {sorted(algorithm_entries)}")

    program_names = _builtin_program_names(repo_root / "teleboost/programs/builtins.py")
    if program_names != _PUBLIC_PROGRAMS:
        raise ReleaseBuildError(f"public program contract failed: expected {sorted(_PUBLIC_PROGRAMS)}, found {sorted(program_names)}")

    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project.get("optional-dependencies", {})
    if set(extras) != _PUBLIC_OPTIONAL_EXTRAS:
        raise ReleaseBuildError(f"public dependency-extra contract failed: expected {sorted(_PUBLIC_OPTIONAL_EXTRAS)}, found {sorted(extras)}")

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if "This public branch is physically Wan-only." not in readme:
        raise ReleaseBuildError("README.md must state the physical public source boundary")


def _stage_root_source(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative in _ROOT_FILES:
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise ReleaseBuildError(f"required root release file is missing or unsafe: {source}")
        shutil.copy2(source, destination / relative)
    for relative in _ROOT_DIRS:
        source = repo_root / relative
        if not source.is_dir() or source.is_symlink():
            raise ReleaseBuildError(f"required root release directory is missing or unsafe: {source}")
        _reject_source_symlinks(source)
        shutil.copytree(source, destination / relative, ignore=_ignore_copy)


def _one_artifact(directory: Path, suffix: str, *, label: str) -> Path:
    matches = sorted(path for path in directory.iterdir() if path.name.endswith(suffix))
    if len(matches) != 1:
        raise ReleaseBuildError(f"expected exactly one {label} in {directory}, found {[path.name for path in matches]}")
    return matches[0]


def _safe_extract_sdist(sdist: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        roots: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
                raise ReleaseBuildError(f"unsafe sdist member: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ReleaseBuildError(f"sdist contains link or special member: {member.name}")
            roots.add(pure.parts[0])
        if len(roots) != 1:
            raise ReleaseBuildError(f"sdist must contain one top-level directory, found {sorted(roots)}")
        root = roots.pop()

        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.extractfile(member)
            if payload is None:
                raise ReleaseBuildError(f"cannot read sdist member: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(payload, output)
            target.chmod(member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))
    return destination / root


def _isolated_install_smoke(
    python: Path,
    root_wheel: Path,
    work_dir: Path,
    env: dict[str, str],
) -> None:
    venv_dir = work_dir / "install-smoke-venv"
    smoke_cwd = work_dir / "install-smoke-cwd"
    smoke_cwd.mkdir()
    _run([str(python), "-m", "venv", str(venv_dir)], cwd=smoke_cwd, env=env)
    venv_python = venv_dir / "bin/python"
    scripts_dir = venv_dir / "bin"
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            str(root_wheel),
        ],
        cwd=smoke_cwd,
        env=env,
    )
    metadata_smoke = """
from importlib import metadata, util

root = metadata.distribution("teleboost")
scripts = {ep.name: ep.value for ep in root.entry_points if ep.group == "console_scripts"}
assert scripts["teleboost-convert-wan-to-teletron"] == "teleboost.cli.convert_checkpoint:main"
assert util.find_spec("teleboost") is not None
assert root.metadata["License-Expression"] == "Apache-2.0"
import teleboost
from teleboost.programs.backend_metadata import BUILTIN_BACKENDS_BY_NAME
from teleboost.programs.builtins import builtin_program_names
assert callable(teleboost.apply_runtime_patches)
assert set(BUILTIN_BACKENDS_BY_NAME) == {"wan"}
assert set(builtin_program_names()) == {
    "wan.grpo.fsdp",
    "wan.bgpo.fsdp",
    "wan.vipo.fsdp",
    "wan.tempflow.fsdp",
    "wan.dpo.teletron",
}
"""
    _run([str(venv_python), "-c", metadata_smoke], cwd=smoke_cwd, env=env)
    _run([str(scripts_dir / "teleboost-convert-wan-to-teletron"), "--help"], cwd=smoke_cwd, env=env)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_artifacts(artifacts: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.iterdir())
    if existing:
        raise ReleaseBuildError(f"release output directory must be empty: {[str(path) for path in existing[:20]]}")
    destinations = [output_dir / source.name for source in artifacts]
    for source, destination in zip(artifacts, destinations, strict=True):
        shutil.copy2(source, destination)
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in destinations),
        encoding="utf-8",
    )
    return destinations


def _outside_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.relative_to(repo_root)
    except ValueError:
        return True
    return False


def build_release_artifacts(
    *,
    repo_root: Path,
    output_dir: Path,
    python: Path,
) -> list[Path]:
    """Build the only supported release profile: the public Wan-only tree."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    # Do NOT resolve() the interpreter: dereferencing a venv's python symlink
    # escapes the venv where the pinned build tooling lives.
    python = python.absolute()
    if not python.is_file():
        raise ReleaseBuildError(f"Python interpreter does not exist: {python}")
    if not _outside_repo(output_dir, repo_root):
        raise ReleaseBuildError(f"release output must be outside the repository (use /tmp or another directory): {output_dir}")

    _validate_public_source_tree(repo_root)
    env = _clean_build_env()
    with tempfile.TemporaryDirectory(prefix="teleboost-release-build-", dir="/tmp") as raw_work:
        work = Path(raw_work)
        root_stage = work / "root-source"
        root_sdist_dir = work / "root-sdist"
        root_wheel_dir = work / "root-wheel"
        root_sdist_dir.mkdir()
        root_wheel_dir.mkdir()

        _stage_root_source(repo_root, root_stage)
        _run(
            [str(python), "-m", "build", "--no-isolation", "--sdist", "--outdir", str(root_sdist_dir), str(root_stage)],
            cwd=work,
            env=env,
        )
        root_sdist = _one_artifact(root_sdist_dir, ".tar.gz", label="root sdist")
        extracted_root = _safe_extract_sdist(root_sdist, work / "root-extracted")
        _run(
            [str(python), "-m", "build", "--no-isolation", "--wheel", "--outdir", str(root_wheel_dir), str(extracted_root)],
            cwd=work,
            env=env,
        )
        root_wheel = _one_artifact(root_wheel_dir, ".whl", label="root wheel")

        checker = repo_root / "tools/release/check_wheel_contents.py"
        _run([str(python), str(checker), str(root_sdist), str(root_wheel)], cwd=work, env=env)
        _run([str(python), "-m", "twine", "check", "--strict", str(root_sdist), str(root_wheel)], cwd=work, env=env)
        _isolated_install_smoke(python, root_wheel, work, env)
        published = _publish_artifacts([root_sdist, root_wheel], output_dir)

    print(f"release artifacts published to {output_dir}")
    for path in published:
        print(f"{_sha256(path)}  {path.name}")
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="fresh output directory outside the repository (default: unique /tmp directory)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python with the pinned release build tools installed",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="teleboost-release-", dir="/tmp"))
    build_release_artifacts(
        repo_root=repo_root,
        output_dir=output_dir,
        python=args.python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

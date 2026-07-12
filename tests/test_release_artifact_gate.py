"""Dependency-light contracts for the hermetic release artifact gate."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

from tools.release.build_artifacts import (
    ReleaseBuildError,
    _publish_artifacts,
    build_release_artifacts,
)
from tools.release.check_wheel_contents import (
    ArtifactContractError,
    _reject_forbidden_paths,
    _reject_private_markers,
    _validate_converter_script,
    _validate_root_attributions,
    validate_root_sdist,
)


@pytest.mark.parametrize(
    "path",
    [
        "third_party/wan/model.py",
        "tests/test_runtime.py",
        "teleboost/__pycache__/runtime.pyc",
        "build/lib/teleboost/__init__.py",
        "outputs.attempt18/checkpoint.pt",
        "tools/diagnostics/probe.py",
        "tools/smoke/model_smoke.py",
        "tools/family_smoke.py",
    ],
)
def test_forbidden_checkout_paths_are_rejected(path: str) -> None:
    with pytest.raises(ArtifactContractError, match="forbidden paths"):
        _reject_forbidden_paths({path}, artifact_name="fixture")


def test_release_tools_are_sdist_only() -> None:
    member = "tools/release/build_artifacts.py"
    with pytest.raises(ArtifactContractError, match="forbidden paths"):
        _reject_forbidden_paths({member}, artifact_name="wheel")
    _reject_forbidden_paths({member}, artifact_name="sdist", allow_release_tools=True)


@pytest.mark.parametrize(
    "payload",
    [
        b"checkout=/" + b"Users/example/TeleBoost",
        b"checkout=/" + b"root/TeleBoost",
        b"model=/" + b"data/wuxn5/model",
        b"remote=code." + b"srdcloud.cn/project",
        b"token=" + b"hf_" + b"a" * 24,
        b"token=" + b"sk-" + b"a" * 24,
    ],
)
def test_private_paths_and_credential_shapes_are_rejected(payload: bytes) -> None:
    with pytest.raises(ArtifactContractError):
        _reject_private_markers({"fixture.txt": payload}, artifact_name="fixture")


def test_root_attribution_contract_matches_checked_in_licenses() -> None:
    root = Path(__file__).parents[1]
    payloads = {
        "NOTICE": (root / "NOTICE").read_bytes(),
        "THIRD_PARTY_PROVENANCE.md": (root / "THIRD_PARTY_PROVENANCE.md").read_bytes(),
        "LICENSES/Megatron-LM-BSD-3-Clause.txt": (root / "LICENSES/Megatron-LM-BSD-3-Clause.txt").read_bytes(),
        "LICENSES/OpenCLIP-MIT.txt": (root / "LICENSES/OpenCLIP-MIT.txt").read_bytes(),
    }
    _validate_root_attributions(
        payloads,
        notice_name="NOTICE",
        provenance_name="THIRD_PARTY_PROVENANCE.md",
        megatron_license_name="LICENSES/Megatron-LM-BSD-3-Clause.txt",
        openclip_license_name="LICENSES/OpenCLIP-MIT.txt",
        artifact_name="fixture",
    )

    payloads["NOTICE"] = b"TeleBoost only"
    with pytest.raises(ArtifactContractError, match="attribution markers"):
        _validate_root_attributions(
            payloads,
            notice_name="NOTICE",
            provenance_name="THIRD_PARTY_PROVENANCE.md",
            megatron_license_name="LICENSES/Megatron-LM-BSD-3-Clause.txt",
            openclip_license_name="LICENSES/OpenCLIP-MIT.txt",
            artifact_name="fixture",
        )


def test_only_canonical_converter_command_is_accepted() -> None:
    scripts = {
        "teleboost-convert-wan-to-teletron": "teleboost.cli.convert_checkpoint:main",
    }
    _validate_converter_script(scripts, artifact_name="fixture")


def test_sdist_rejects_links_before_extraction(tmp_path: Path) -> None:
    sdist = tmp_path / "teleboost-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        root = tarfile.TarInfo("teleboost-0.1.0")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        link = tarfile.TarInfo("teleboost-0.1.0/teleboost/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)

    with pytest.raises(ArtifactContractError, match="link or special member"):
        validate_root_sdist(sdist)


def test_sdist_rejects_path_traversal(tmp_path: Path) -> None:
    sdist = tmp_path / "teleboost-0.1.0.tar.gz"
    payload = b"outside"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("teleboost-0.1.0/../outside")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ArtifactContractError, match="unsafe archive member"):
        validate_root_sdist(sdist)


def test_builder_refuses_output_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ReleaseBuildError, match="outside the repository"):
        build_release_artifacts(
            repo_root=tmp_path,
            output_dir=tmp_path / "dist",
            python=Path(sys.executable),
        )


def test_publisher_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"wheel")
    output = tmp_path / "release"
    output.mkdir()
    (output / "stale.whl").write_bytes(b"stale")

    with pytest.raises(ReleaseBuildError, match="must be empty"):
        _publish_artifacts([artifact], output)

    assert not (output / artifact.name).exists()

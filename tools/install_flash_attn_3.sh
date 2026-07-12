#!/usr/bin/env bash
# Build and install the exact optional FlashAttention-3 Hopper source pin.
# Only the required CUTLASS submodule is fetched; pip's recursive VCS clone
# would also fetch unrelated ROCm submodules.
#
#   tools/install_flash_attn_3.sh --show  # display the immutable inputs
#   tools/install_flash_attn_3.sh         # build a wheel, then install --no-deps
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
contract="$root/constraints/upstreams/flash-attn-3.txt"
[ -f "$contract" ] || { echo "missing $contract" >&2; exit 1; }

field() {
    awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$contract"
}

upstream_url="$(field UPSTREAM_URL)"
upstream_commit="$(field UPSTREAM_COMMIT)"
subdirectory="$(field UPSTREAM_SUBDIRECTORY)"
cutlass_commit="$(field CUTLASS_COMMIT)"
package="$(field PACKAGE)"
version="$(field VERSION)"
build_profile="$(field BUILD_PROFILE)"
python_bin="${PYTHON:-python}"

for value in "$upstream_url" "$upstream_commit" "$subdirectory" "$cutlass_commit" "$package" "$version" "$build_profile"; do
    [ -n "$value" ] || { echo "invalid $contract" >&2; exit 1; }
done

echo "FlashAttention-3: $upstream_url@$upstream_commit#$subdirectory"
echo "CUTLASS: $cutlass_commit"
echo "Distribution: $package==$version+$build_profile"
echo "Kernel profile: SM90, head-dim 128, FP16/BF16, forward/backward, varlen/local/GQA"
echo "Build: CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MAX_JOBS=${MAX_JOBS:-4} NVCC_THREADS=${NVCC_THREADS:-2}"
[ "${1:-}" = "--show" ] && exit 0
[ $# -eq 0 ] || { echo "usage: $0 [--show]" >&2; exit 2; }

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
[ -x "$cuda_home/bin/nvcc" ] || { echo "nvcc not found under CUDA_HOME=$cuda_home" >&2; exit 1; }

"$python_bin" - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("FlashAttention-3 requires a CUDA PyTorch build")
print(f"Using existing torch {torch.__version__} (CUDA {torch.version.cuda}); pip runs with --no-deps")
PY

work="$(mktemp -d "${TMPDIR:-/tmp}/teleboost-fa3.XXXXXX")"
trap 'rm -rf "$work"' EXIT
source_dir="$work/source"
wheel_dir="$work/wheel"
mkdir -p "$source_dir" "$wheel_dir"

git -C "$source_dir" init -q
git -C "$source_dir" remote add origin "$upstream_url"
git -C "$source_dir" fetch --depth 1 origin "$upstream_commit"
git -C "$source_dir" checkout -q --detach FETCH_HEAD
[ "$(git -C "$source_dir" rev-parse HEAD)" = "$upstream_commit" ] || {
    echo "FlashAttention source revision mismatch" >&2
    exit 1
}
git -C "$source_dir" submodule update --init --depth 1 csrc/cutlass
[ "$(git -C "$source_dir/csrc/cutlass" rev-parse HEAD)" = "$cutlass_commit" ] || {
    echo "CUTLASS source revision mismatch" >&2
    exit 1
}

CUDA_HOME="$cuda_home" \
PATH="$cuda_home/bin:$PATH" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_LOCAL_VERSION="$build_profile" \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE \
FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
MAX_JOBS="${MAX_JOBS:-4}" \
NVCC_THREADS="${NVCC_THREADS:-2}" \
    "$python_bin" -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir "$wheel_dir" "$source_dir/$subdirectory"

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
[ "${#wheels[@]}" -eq 1 ] || { echo "expected exactly one wheel in $wheel_dir" >&2; exit 1; }
"$python_bin" -m pip install --no-deps --force-reinstall "${wheels[0]}"

"$python_bin" - <<'PY'
from importlib.metadata import version
from flash_attn_3 import flash_attn_interface

installed = version("flash-attn-3")
if not installed.startswith("3.0.0+teleboost.wan.sm90"):
    raise SystemExit(f"unexpected flash_attn_3 build: {installed}")
print("Installed flash_attn_3", installed, flash_attn_interface.__file__)
PY

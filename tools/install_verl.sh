#!/usr/bin/env bash
# Install the pinned upstream verl for this checkout, as declared in
# constraints/upstreams/verl.txt.
#
#   tools/install_verl.sh          # install the pin
#   tools/install_verl.sh --show   # dry-run: print the command only
#
# The contract file is never sourced or eval'd wholesale — the single
# PIP_INSTALL line is extracted, echoed, then executed.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
req="$root/constraints/upstreams/verl.txt"
[ -f "$req" ] || { echo "missing $req" >&2; exit 1; }

cmd="$(awk '/^PIP_INSTALL=/{sub(/^PIP_INSTALL=/,""); print; exit}' "$req")"
[ -n "$cmd" ] || { echo "no PIP_INSTALL line in $req" >&2; exit 1; }

echo "+ $cmd"
[ "${1:-}" = "--show" ] && exit 0
exec bash -c "$cmd"

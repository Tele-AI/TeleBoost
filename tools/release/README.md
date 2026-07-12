# Wan-only release gate

The public branch, sdist, and wheel use one Wan-only source boundary. The
builder does not delete or rewrite family code while staging.

```bash
python -m pip install -c constraints/release.txt -e '.[release]'
python tools/release/build_artifacts.py \
  --out-dir /tmp/teleboost-release-wan
```

The builder:

1. rejects non-Wan family paths/references in public production source and
   documentation;
2. copies an allowlisted source tree to a fresh `/tmp` workspace;
3. builds an sdist, safely extracts it, and builds the wheel only from that
   extraction;
4. runs `tools/release/check_wheel_contents.py`, `twine check --strict`, and a
   fresh-venv `--no-deps` install/CLI smoke;
5. copies only validated artifacts and `SHA256SUMS` to a fresh output directory
   outside the checkout.

Re-run the archive checker directly with:

```bash
python tools/release/check_wheel_contents.py \
  /tmp/teleboost-release-wan/teleboost-0.1.0.tar.gz \
  /tmp/teleboost-release-wan/teleboost-0.1.0-py3-none-any.whl
```

The gate rejects vendored trees, tests, caches, build/output trees, diagnostics,
private path markers, credential-shaped strings, unexpected packages, missing
license texts, and stale CLI aliases. Passing it does not replace the human
maintainer/legal approval described in `THIRD_PARTY_PROVENANCE.md`.

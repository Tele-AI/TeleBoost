# Third-party source boundary

This directory contains development/source-checkout mirrors and is excluded
from the Apache-2.0 root sdist and wheel. Each component keeps its own license,
notices, provenance, and redistribution conditions. Production launchers never
add this directory to `PYTHONPATH`; users explicitly provide an importable
runtime.

| Directory | Role | Release handling |
|---|---|---|
| `wan/` | Alibaba Wan runtime mirror | Excluded; Apache-2.0 upstream license retained. |
| `raft/` | Optional optical-flow reward runtime | Excluded; BSD-3-Clause license retained. |
| `Videophy/` | Optional video-physics reward runtime | Excluded; directory and file-level notices retained. |

Repository tests may explicitly add `third_party/` to resolve the top-level
`wan` fixture. This is test setup, not a production fallback.

See `../THIRD_PARTY_PROVENANCE.md`, `../MODEL_AND_DATA_LICENSES.md`, and each
component's license before copying or redistributing source.

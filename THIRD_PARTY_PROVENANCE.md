# Third-party provenance and release boundary

This is an engineering redistribution inventory, not legal advice. File-level
copyright and license notices remain authoritative.

## Root distribution

The sdist and wheel contain the first-party `teleboost/` package, declarative
`recipes/`, and release tooling. They exclude all `third_party/` trees, tests,
smoke/diagnostic programs, weights, datasets, outputs, and checkpoints.

| Component | Distributed paths | Recorded baseline | License / handling |
|---|---|---|---|
| verl | External dependency; thin adapters under `teleboost/patches/` | 0.7.1, immutable revision in `constraints/upstreams/verl.txt` | Apache-2.0; not vendored in root artifacts. |
| NVIDIA Megatron-LM | `teleboost/engines/teletron/` and selected training/checkpoint files retaining NVIDIA headers | `55ac7082517c3878ae653c07c09c534b8aed49f6` | BSD-3-Clause upstream portions; full terms in `LICENSES/Megatron-LM-BSD-3-Clause.txt`. |
| OpenCLIP | `teleboost/models/offline_clip.py` | `ea7718f927b84e1b46ce057d3eae5ca4c9c41434` | MIT upstream portions; full terms in `LICENSES/OpenCLIP-MIT.txt`. |
| ModelScope DiffSynth-Studio | Flow scheduler and Wan VAE ports | scheduler `451aab01161496fd68510e7682306eaf54ff97f2`; VAE ancestor `5be5c32fe4b240547a288afa4c29e3f81b6ef881` | Apache-2.0; preserve attribution and modification notices. |
| Tele-AI TeleTron | `teleboost/engines/teletron/`, `teleboost/models/wan/teletron/` | `5fb0431bd9a42a14cf4ee5768d0a0482828e1ff5` | Apache-2.0 ports; NVIDIA-derived files retain BSD notices. |

The OpenCLIP and DiffSynth original import metadata was not retained. These
commits are reproducible reconciliation baselines: OpenCLIP was compared across
upstream release tags; the DiffSynth scheduler is a direct code match modulo
documented changes; the VAE revision is the closest structural ancestor. A
maintainer/legal reviewer must approve these reconciliations before the first
public source tag. Packaging checks cannot make that authorship decision.

## Source-checkout-only components

| Local path | Purpose and baseline | Release handling |
|---|---|---|
| `third_party/wan/` | Alibaba Wan runtime, baseline `204f899b6436fe2e1705a0b67c464b30b8137799` | Apache-2.0 license retained; excluded from root artifacts. |
| `third_party/Videophy/` | VideoPhy / mPLUG-Owl integration | MIT directory license with embedded Apache-2.0 file headers; excluded. |
| `third_party/raft/` | Princeton RAFT runtime | BSD-3-Clause license retained; excluded. |

Optional VideoCLIP source is user supplied and not present by default. Its
identified terms are non-commercial and must be reviewed independently.

## Boundary enforcement

```bash
python tools/release/build_artifacts.py \
  --out-dir /tmp/teleboost-release-wan
```

The builder rejects non-Wan family code in the checkout, stages an allowlisted
copy without rewriting product source, builds the wheel only from the extracted
sdist, and rejects vendored source, tests, diagnostics, outputs, private path
markers, missing attributions, or missing license texts.

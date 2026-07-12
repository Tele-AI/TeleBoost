# recipes/ layout

`recipes/` is declarative launch surface only. It is not a Python package and
must not be imported by `teleboost`.

Each directory is named by the public program taxonomy:

```
<family>_<algorithm>_<engine>[_<policy>]
```

Current first-class programs:

| Directory | Program | Notes |
|---|---|---|
| `wan_grpo_fsdp/` | `wan.grpo.fsdp` | Wan GRPO baseline over the FSDP engine. |
| `wan_bgpo_fsdp/` | `wan.bgpo.fsdp` | BGPO as a first-class self-developed algorithm; implemented by thin program wiring plus `teleboost.algorithms.bgpo`. |
| `wan_vipo_fsdp/` | `wan.vipo.fsdp` | VIPO as a first-class self-developed algorithm; implemented by thin program wiring plus `teleboost.algorithms.vipo`. |
| `wan_tempflow_fsdp/` | `wan.tempflow.fsdp` | TempFlow changes rollout topology, so it is a program. |
| `wan_dpo_teletron/` | `wan.dpo.teletron` | Wan DPO over the TeleTron/Megatron engine. |

Directory contract:

```
recipes/<program>/
  README.md       optional user-facing notes
  config.yaml     declarative program identity
  run.sh          launcher when that program has a supported public entrypoint
  prompts/        optional example prompts
```

No production Python belongs here. Program construction lives under
`teleboost/programs/`; algorithm math lives under `teleboost/algorithms/`;
runtime defaults live under `teleboost/config/`.

Preset-style features such as GRPO-Guard are not recipe directories. They
remain algorithm/config capabilities composed by a first-class program,
usually `wan.grpo.fsdp`.

## External backend plugins

Third-party model families extend TeleBoost through the
`teleboost.programs` Python entry-point group. Discovery is exact-name and lazy:
only `backend.name=<plugin-name>` loads that plugin entry point.

```toml
[project.entry-points."teleboost.programs"]
acme = "acme_teleboost.backend:registration"
```

The entry point returns a `BackendRegistration` or a zero-argument function that
returns one. Keep registration dependency-light; put framework/model imports
inside the backend factory.

Built-in backend runtime types and capabilities are listed in
`teleboost/programs/backend_metadata.py`; concrete in-tree factories are
installed by the `teleboost.programs` composition root.

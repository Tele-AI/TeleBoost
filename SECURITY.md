# Security policy

TeleBoost is a research training framework intended for trusted, isolated GPU
clusters. It is not a sandbox for untrusted models, configs, plugins, datasets,
or cluster users.

## Reporting a vulnerability

Use the repository host's private security-advisory feature to report a
vulnerability to the TeleBoost maintainers. Do not include exploit details,
credentials, private model/data URLs, or affected cluster addresses in a public
issue. The public repository must enable a private reporting channel before its
first release.

Include the affected TeleBoost version/commit, dependency and CUDA/PyTorch
versions, minimal reproduction, impact, and any suggested mitigation. The
maintainers will acknowledge the report, coordinate a fix, and publish an
advisory when appropriate.

Only the latest released `0.x` line and the current default branch receive
security fixes during the research-preview phase. There is no long-term support
promise for older snapshots.

## Trust boundaries

Treat all of the following as executable or otherwise privileged input:

- Python reward plugins configured by module or file path. Importing one
  executes its top-level code on a training worker.
- Pickle files (accepted only with the explicit `allow_unsafe_pickle=True`
  compatibility opt-in) and any external tool that loads PyTorch checkpoints
  with `weights_only=False`. Load them only from a trusted, access-controlled
  source. TeleBoost's built-in YAML readers use SafeLoader and its checkpoint
  readers use `weights_only=True`.
- Hugging Face repositories when any caller enables `trust_remote_code`.
- Local editable packages, CUDA extensions, launch scripts, and Ray runtime
  environments inherited by worker processes.

Prefer safetensors or `torch.load(..., weights_only=True)` for weights, JSON or
safe YAML for metadata, immutable artifact revisions, and cryptographic
checksums. Never run a downloaded config, checkpoint, or plugin merely because
its filename looks familiar.

## Cluster and network security

Ray control ports, vLLM/SGLang-compatible reward endpoints, NCCL rendezvous,
TensorBoard, and other launch-time services must run on a trusted private
network protected by host and network firewalls. The built-in reward clients
may use plain HTTP and do not provide authentication or transport encryption.
Do not expose these services to the public internet. Use a secured proxy or
service mesh when traffic crosses a trust boundary.

Run jobs under a dedicated unprivileged account and environment. Restrict model,
dataset, output, and checkpoint permissions; avoid shared writable package
directories; and do not install into a cluster base environment used by active
jobs.

## Secrets and release hygiene

- Pass tokens and credentials through the cluster's secret manager. Do not put
  them in Hydra overrides, shell history, logs, experiment names, or committed
  environment files.
- Assume resolved configs and Ray runtime environments can be logged. Redact
  secret-valued keys before printing or uploading diagnostics.
- Build releases from a clean Git worktree. Review the exact archive and wheel
  contents, scan for credentials/private paths, and never publish `outputs/`,
  checkpoints, generated videos, caches, or editable-install metadata.
- Run `tools/release/build_artifacts.py` before publishing. It verifies the
  public source boundary, builds the wheel from a fresh root sdist, checks both
  artifacts, runs strict metadata validation, and performs an isolated install
  smoke.

See `MODEL_AND_DATA_LICENSES.md` for model/data provenance and
`THIRD_PARTY_PROVENANCE.md` for the software redistribution boundary.

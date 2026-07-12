# Model and data licenses

TeleBoost distributes training software, not model weights, datasets, private
prompts, or generated checkpoints. Apache-2.0 code does not relicense an
artifact loaded at runtime.

Users must review the authoritative model card, dataset card, license,
acceptable-use policy, geographic restrictions, immutable revision, and
checksum for every artifact used by an experiment.

## Runtime artifact inventory

| Integration | User-supplied artifact | Required action |
|---|---|---|
| Wan 2.1 / 2.2 | DiT, VAE, encoders, and tokenizer files | Review the exact checkpoint's model-card terms; runtime source licensing does not automatically license weights. |
| Qwen3-VL judge | Model and processor directory | Review the selected checkpoint, tokenizer, processor, and acceptable-use terms. |
| HPSv2 / aesthetic CLIP | Reward-model checkpoints | Review source and weight licenses independently. |
| RAFT | Optical-flow checkpoint | Review checkpoint terms independently of the BSD-3-Clause runtime source. |
| VideoPhy | Physics/video-quality model and processor | Review model-card and dataset-derived restrictions. |
| VideoCLIP-XL | Optional source and weights | The identified integration terms are non-commercial; obtain and review it independently. |
| DINOv2 / VIPO | Feature model and optional PCA assets | Review checkpoint licensing and PCA/training-asset provenance. |
| CUDA extensions | Apex, FlashAttention, Transformer Engine | Follow each project's license and build against the active CUDA/PyTorch stack. |

## Training and evaluation data

- Use only prompts, media, captions, preferences, and annotations you are
  authorized to process and train on.
- Remove credentials, personal data, private URLs, and internal filesystem
  paths before sharing configs, logs, traces, checkpoints, or outputs.
- Record collection basis, consent, license, filtering, retention, and deletion
  requirements. A local path is not provenance.
- Review generated media before publication. A trained checkpoint may inherit
  restrictions from its base model, reward models, and data.

`.gitignore` is not a release boundary. Use the hermetic release builder and
audit its final archive list before publishing.

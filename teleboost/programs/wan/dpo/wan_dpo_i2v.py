# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Production-scale Wan I2V DPO config (14B + 8-shard preference CSV).

Paths are read from env vars at import time so OSS users only need to
export their data + checkpoint roots once:

    WAN_DPO_DATA_DIR   directory holding the preference-pair CSV shards
                       (training: ``prompt_video_pairs_enhanced_part{0..7}.csv``;
                        eval:     ``prompt_video_pairs_matched_image.csv``)
    WAN_T2V_1_3B_DIR   Wan2.1-T2V-1.3B upstream release dir (VAE + T5 weights)
    WAN_I2V_14B_DIR    Wan2.1-I2V-14B-480P upstream release dir
                       (T5 tokenizer + CLIP image encoder)
    WAN_DPO_CHOSEN_PATH_KEY   chosen-video CSV column (default: ``chosen``)
    WAN_DPO_REJECTED_PATH_KEY rejected-video CSV column (default: ``rejected``)
    WAN_DPO_PROMPT_KEY        prompt CSV column (default: ``prompt``)

When unset, paths fall back to ``/path/to/...`` placeholders so the
module still imports cleanly; ``build_train_valid_test_datasets``
raises on first attempted read, naming the unset env var.

The 1.3B T2V variant (smaller arch, single-CSV smoke / dev data) is in
the sibling ``wan_dpo_t2v.py`` — it imports this module and overrides
the model-arch + dataset fields.
"""

import os

_DATA = os.environ.get("WAN_DPO_DATA_DIR", "/path/to/dpo_csv")
_T2V_1_3B = os.environ.get("WAN_T2V_1_3B_DIR", "/path/to/Wan2.1-T2V-1.3B")
_I2V_14B = os.environ.get("WAN_I2V_14B_DIR", "/path/to/Wan2.1-I2V-14B-480P")
_CHOSEN_PATH_KEY = os.environ.get("WAN_DPO_CHOSEN_PATH_KEY", "chosen")
_REJECTED_PATH_KEY = os.environ.get("WAN_DPO_REJECTED_PATH_KEY", "rejected")
_PROMPT_KEY = os.environ.get("WAN_DPO_PROMPT_KEY", "prompt")


dst_size = (832, 480)
dst_fps = 16
dst_num_frames = 49

config = dict(
    dataset=dict(
        type="CSVPreferenceDPODataset",
        dataset_base_path="",
        dataset_metadata_path=f"{_DATA}/prompt_video_pairs_enhanced.csv",
        data_path_list=[f"{_DATA}/prompt_video_pairs_enhanced_part{i}.csv" for i in range(8)],
        dataset_repeat=2,
        # Keep the encoder/training branch schema canonical while allowing
        # manifests to use independently configurable CSV column names.
        chosen_video_key="chosen",
        rejected_video_key="rejected",
        chosen_path_key=_CHOSEN_PATH_KEY,
        rejected_path_key=_REJECTED_PATH_KEY,
        prompt_key=_PROMPT_KEY,
        height=480,
        width=832,
        num_frames=49,
        time_division_factor=4,
        time_division_remainder=1,
        height_division_factor=16,
        width_division_factor=16,
        max_pixels=1920 * 1080,
        transforms=[
            dict(
                type="InjectRawFirstImageFromVideo",
                video_key="video",
                output_key="raw_first_image",
            ),
            dict(
                type="PreprocessVideoToTensor",
                input_key="video",
                output_key="video",
                torch_dtype="bfloat16",
                pattern="B C T H W",
                min_value=-1,
                max_value=1,
                skip_if_tensor=True,
            ),
            dict(
                type="InjectImagesFromVideoTensor",
                video_key="video",
                output_key="images",
            ),
            dict(
                type="InjectPromptToTopLevel",
                prompt_key=_PROMPT_KEY,
            ),
            dict(
                type="PackInputsNoResize",
                normalize=False,
                image_keys=["images"],
                embedding_keys=["raw_first_image", "input_image"],
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
            f"{_DATA}/prompt_video_pairs_matched_image.csv",
        ],
        eval_time_steps=[200, 400, 600, 800, 1000],
    ),
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    ),
    model_config=dict(
        dit=dict(
            type="ParallelWanTeletronModel",
            # Architecture sizes — uncomment the row matching your target model.
            #   1.3B: dim=1536, ffn_dim=8960,  num_heads=12, num_layers=30
            #   10B:  dim=5120, ffn_dim=13824, num_heads=40, num_layers=30
            #   14B:  dim=5120, ffn_dim=13824, num_heads=40, num_layers=40
            # in_dim: t2v=16, i2v=36 (Wan2.1) | i2v Wan2.2=36 also
            config=dict(
                has_image_input=True,
                patch_size=[1, 2, 2],
                in_dim=36,
                dim=5120,
                ffn_dim=13824,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40,
                num_layers=40,
                eps=1e-6,
                has_image_pos_emb=False,
            ),
            train=dict(
                trainable_models="dit",
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=True,
                enable_fp8_training=False,
                lora=dict(
                    enable=False,
                    base_model=None,
                    target_modules="q,k,v,o,ffn.0,ffn.2",
                    rank=32,
                    checkpoint=None,
                ),
                dpo=dict(
                    enable=True,
                    beta=0.1,
                ),
                extra_inputs=["input_image"],
            ),
        ),
        encoder=dict(
            type="wan_teletron_encoder",
            encoder_schema=["context", "img_clip_feature", "img_emb_y", "latents"],
            vae=dict(
                type="DiffSynthWanVideoVAE",
                path=f"{_T2V_1_3B}/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
                torch_compile=False,
            ),
            text_encoder=dict(
                path=f"{_T2V_1_3B}/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path=f"{_I2V_14B}/google/umt5-xxl",
            ),
            image_encoder=dict(
                path=f"{_I2V_14B}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                torch_compile=False,
            ),
        ),
        training=dict(
            diffusion=dict(
                max_timestep_boundary=0.358,
                min_timestep_boundary=0.0,
            ),
            dpo_io=dict(
                chosen_key="chosen",
                rejected_key="rejected",
            ),
            scheduler=dict(
                num_train_timesteps=1000,
            ),
        ),
    ),
)

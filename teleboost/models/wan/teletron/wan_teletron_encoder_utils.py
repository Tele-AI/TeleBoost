# Copyright (c) 2025 TeleAI-infra Team (TeleTron)
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
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from teleboost.engines.teletron import get_args, set_config


def encode_prompt(prompter, prompt, positive=True):
    prompt_emb = prompter.encode_prompt(prompt, positive=positive, device=torch.cuda.current_device())
    return {"context": prompt_emb}


def encode_image(
    vae,
    image_encoder,
    image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    dtype=torch.bfloat16,
    compression=(4, 8, 8),
):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
    # print("msk create shape:", 1, num_frames, height // 8, width // 8 ) # 1, 81, 56, 98
    msk[:, 1:] = 0  # 1, 1:81, 56, 98
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)  # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
    # print("msk view shape:", 1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2])  # 1, 21, 4, 56, 98
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}


def encode_image_with_mask(vae, image_encoder, image, num_frames, height, width, msk, ref_images, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), dtype=torch.bfloat16):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    ref_images = rearrange(ref_images, "b t c h w -> b c t h w")
    y = encode_video(vae, ref_images.to(dtype=dtype, device=torch.cuda.current_device()), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
    y = y.unsqueeze(0)
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    msk = msk.transpose(1, 2).to(torch.cuda.current_device())
    y = torch.concat([msk, y], dim=1)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}


def encode_first_last_image(
    vae,
    image_encoder,
    pil_first_image,
    pil_last_image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    dtype=torch.bfloat16,
    compression=(4, 8, 8),
):
    first_image = preprocess_image(pil_first_image.resize((width, height))).to(torch.cuda.current_device())
    last_image = preprocess_image(pil_last_image.resize((width, height))).to(torch.cuda.current_device())
    # if self.dit.has_image_pos_emb:
    #     clip_context = torch.cat([self.image_encoder.encode_image([first_image]),
    #                             self.image_encoder.encode_image([last_image])], dim=1)
    # else:
    #     clip_context = self.image_encoder.encode_image([first_image])
    clip_context = torch.cat(
        [
            image_encoder.encode_image([first_image]),
            image_encoder.encode_image([last_image]),
        ],
        dim=1,
    )
    msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
    msk[:, 1:-1] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2])
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [
            first_image.transpose(0, 1),
            torch.zeros(3, num_frames - 2, height, width).to(first_image.device),
            last_image.transpose(0, 1),
        ],
        dim=1,
    )
    y = vae.encode(
        [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}


def encode_video(vae, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    latents = vae.encode(
        input_video,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return latents


def preprocess_image(image):
    image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
    return image


def _raw_image_to_pil(raw_image):
    if isinstance(raw_image, Image.Image):
        return raw_image
    if isinstance(raw_image, list | tuple) and len(raw_image) > 0:
        raw_image = raw_image[0]
        if isinstance(raw_image, Image.Image):
            return raw_image
    if torch.is_tensor(raw_image):
        tensor = raw_image
        if tensor.dim() >= 5:
            tensor = tensor[0][0]
        elif tensor.dim() == 4:
            tensor = tensor[0]
        tensor = tensor.detach().cpu().clamp(0, 255)
        return to_pil_image(tensor.permute(1, 2, 0).numpy().astype(np.uint8))
    return raw_image


@torch.no_grad
def get_context(batch, prompter, dtype=torch.bfloat16):
    prompt_emb = encode_prompt(prompter, batch["struct_prompt"][0])
    prompt_emb["context"] = prompt_emb["context"].to(dtype=dtype, device=torch.cuda.current_device())
    return prompt_emb["context"]


@torch.no_grad
def get_unprompt_emb(batch, prompter, dtype=torch.bfloat16):
    args = get_args()
    batch_size = args.micro_batch_size
    prompt_emb = encode_prompt(prompter, [args.negative_prompt] * batch_size)
    prompt_emb["context"] = prompt_emb["context"].to(dtype=dtype, device=torch.cuda.current_device())
    return prompt_emb["context"]


@torch.no_grad
def get_img_clip_feature(batch, image_encoder, dtype=torch.bfloat16):
    _, num_frames, _, height, width = batch["images"].shape
    if "raw_last_image" in batch:
        raise NotImplementedError("raw_last_image is not supported yet")
    elif "raw_first_image" in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = _raw_image_to_pil(raw_first_image)
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        clip_context = image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    elif "ref_images" in batch:
        raise NotImplementedError("ref_images is not supported yet")
    return clip_context


@torch.no_grad
def get_img_emb_y(batch, vae, dtype=torch.bfloat16, compression=(4, 8, 8), tiler_kwargs=None):
    tiler_kwargs = dict(tiler_kwargs) if tiler_kwargs is not None else {}
    _, num_frames, _, height, width = batch["images"].shape
    if "ref_images" in batch:
        # assert False, "ref_images is not supported yet"
        ref_images = rearrange(batch["ref_images"], "b t c h w -> b c t h w")
        y = vae.encode(ref_images.to(dtype=dtype, device=torch.cuda.current_device()), device=torch.cuda.current_device(), **tiler_kwargs)
        msk = batch["ref_mask"].transpose(1, 2).to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y], dim=1)

    elif "raw_first_image" in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = _raw_image_to_pil(raw_first_image)
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())

        msk[:, 1:] = 0  # 1, 1:81, 56, 98
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)  # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98

        msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2])  # 1, 21, 4, 56, 98
        msk = msk.transpose(1, 2)[0]
        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros((3, num_frames - 1, height, width), device=image.device)],
            dim=1,
        )
        y = vae.encode(
            [vae_input.to(dtype=dtype, device=torch.cuda.current_device())],
            device=torch.cuda.current_device(),
            tiled=False,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )[0]
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=dtype, device=torch.cuda.current_device())

    return y


@torch.no_grad
def get_latents(batch, vae, dtype=torch.bfloat16, tiler_kwargs=None):
    tiler_kwargs = dict(tiler_kwargs) if tiler_kwargs is not None else {}

    def _get_latents(images):
        # Rearrange the video tensor to the (b, c, t, h, w) layout expected by the VAE and encode it
        latents = encode_video(vae, rearrange(images, "b t c h w -> b c t h w").to(dtype=dtype, device=torch.cuda.current_device()), **tiler_kwargs)
        return latents.to(dtype=dtype, device=torch.cuda.current_device())

    return _get_latents(batch["images"])


@torch.no_grad
def get_noise(batch, dtype=torch.bfloat16, compression=(4, 8, 8)):
    if "latents" in batch:
        return torch.randn_like(batch["latents"]).to(dtype=dtype, device=torch.cuda.current_device())
    else:
        bsz, num_frames, _, height, width = batch["images"].shape
        return torch.randn(bsz, 16, (num_frames + 3) // compression[0], height // compression[1], width // compression[2]).to(dtype=dtype, device=torch.cuda.current_device())


@torch.no_grad
def get_fake_latents(batch, vae, dtype=torch.bfloat16, tiler_kwargs=None):
    tiler_kwargs = dict(tiler_kwargs) if tiler_kwargs is not None else {}
    bsz, num_frames, video_channels, height, width = batch["images"].shape

    low_res_video = torch.nn.functional.interpolate(rearrange(batch["images"], "b t c h w -> (b t) c h w"), size=(height // 2, width // 2), mode="bilinear").reshape(bsz, num_frames, video_channels, height // 2, width // 2)

    low_res_latent = encode_video(
        vae,
        rearrange(low_res_video, "b t c h w -> b c t h w").to(dtype=dtype, device=torch.cuda.current_device()),
        **tiler_kwargs,
    )  # b c t h w

    bsz, latent_channels, latent_frames, latent_height, latent_width = bsz, 16, (num_frames + 3) // 4, height // 8, width // 8
    fake_latents = torch.nn.functional.interpolate(rearrange(low_res_latent, "b c t h w -> (b t) c h w"), size=(latent_height, latent_width), mode="nearest").reshape(bsz, latent_frames, latent_channels, latent_height, latent_width)[0]  # t, c, h, w
    fake_latents = fake_latents.permute(1, 0, 2, 3)  # c, t, h, w

    fake_latents = fake_latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())

    return fake_latents  # b, c, t, h, w


@torch.no_grad
def get_depth_latents(batch, depth_model, vae, dtype=torch.bfloat16, tiler_kwargs=None):
    tiler_kwargs = dict(tiler_kwargs) if tiler_kwargs is not None else {}
    global_config = set_config()
    target_fps = global_config.dataset.filter_cfg.dst_fps
    frames = rearrange(batch["images"], "b t c h w -> b t h w c").squeeze(0).numpy()
    depths, fps = depth_model.infer_video_depth(frames, target_fps, device="cuda", fp32=dtype == torch.float32)
    depths = torch.from_numpy(depths).unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1).to(dtype=dtype, device=torch.cuda.current_device())

    depths_latents = encode_video(
        vae,
        rearrange(depths, "b t c h w -> b c t h w").to(dtype=dtype, device=torch.cuda.current_device()),
        **tiler_kwargs,
    )
    return depths_latents.to(dtype=dtype, device=torch.cuda.current_device())

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
from einops import repeat


class InjectRawFirstImageFromVideo:
    def __init__(self, video_key="video", output_key="raw_first_image"):
        self.video_key = video_key
        self.output_key = output_key

    def __call__(self, data_dict):
        if self.output_key in data_dict:
            return data_dict
        video = data_dict.get(self.video_key, None)
        if isinstance(video, list | tuple) and len(video) > 0:
            first = video[0]
            if isinstance(first, torch.Tensor):
                raw_first = first
                if raw_first.dim() == 3:
                    raw_first = raw_first.unsqueeze(0)
                data_dict[self.output_key] = raw_first.contiguous()
            else:
                # PIL ---> Tensor
                # The dataloader's collate_fn batches these later and breaks on non-tensors, so convert to a tensor first.
                raw_first = torch.from_numpy(np.array(first)).permute(2, 0, 1).contiguous()
                data_dict[self.output_key] = raw_first.unsqueeze(0)
        return data_dict


class PreprocessVideoToTensor:
    def __init__(
        self,
        input_key="video",
        output_key="video",
        torch_dtype=torch.bfloat16,
        device="cpu",
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
        skip_if_tensor=True,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.torch_dtype = torch_dtype
        self.device = device
        self.pattern = pattern
        self.min_value = min_value
        self.max_value = max_value
        self.skip_if_tensor = skip_if_tensor

        parts = pattern.split()
        self.t_dim = parts.index("T")
        self.image_pattern = " ".join([p for p in parts if p != "T"])

    def _image_to_tensor(self, image):
        tensor = torch.tensor(np.array(image, dtype=np.float32))
        tensor = tensor.to(dtype=self.torch_dtype, device=self.device)
        tensor = tensor * ((self.max_value - self.min_value) / 255) + self.min_value
        tensor = repeat(tensor, f"H W C -> {self.image_pattern}", **({"B": 1} if "B" in self.image_pattern else {}))
        return tensor

    def __call__(self, data_dict):
        video = data_dict.get(self.input_key, None)
        if video is None:
            return data_dict
        if torch.is_tensor(video):
            if self.skip_if_tensor:
                data_dict[self.output_key] = video
                return data_dict
            data_dict[self.output_key] = video.to(dtype=self.torch_dtype, device=self.device)
            return data_dict
        from PIL import Image

        if isinstance(video, Image.Image):
            video = [video]
        frames = [self._image_to_tensor(image) for image in video]
        data_dict[self.output_key] = torch.stack(frames, dim=self.t_dim)
        return data_dict


class InjectImagesFromVideoTensor:
    def __init__(self, video_key="video", output_key="images", take_batch_index=0):
        self.video_key = video_key
        self.output_key = output_key
        self.take_batch_index = take_batch_index

    def __call__(self, data_dict):
        if self.output_key in data_dict:
            return data_dict
        video = data_dict.get(self.video_key, None)
        if not torch.is_tensor(video):
            return data_dict
        if video.dim() == 5:
            images = video[self.take_batch_index].permute(1, 0, 2, 3).contiguous()
        elif video.dim() == 4:
            if video.shape[0] in (1, 3, 4):
                images = video.permute(1, 0, 2, 3).contiguous()
            else:
                images = video.contiguous()
        elif video.dim() == 3:
            images = video.unsqueeze(0).contiguous()
        else:
            return data_dict
        data_dict[self.output_key] = images
        return data_dict

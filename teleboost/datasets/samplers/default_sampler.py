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
import math

import torch


class DefaultSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, global_batch_size, seed=42, drop_last=True, shuffle=True, infinite=True):
        # Keep a copy of input params for later use.
        self.total_samples = len(dataset)

        if self.total_samples <= 0:
            raise ValueError("DefaultSampler requires a non-empty dataset")
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        if data_parallel_size <= 0:
            raise ValueError(f"data_parallel_size must be positive, got {data_parallel_size}")
        if not 0 <= data_parallel_rank < data_parallel_size:
            raise ValueError(f"data_parallel_rank must be in [0, {data_parallel_size}), got {data_parallel_rank}")
        if consumed_samples < 0:
            raise ValueError(f"consumed_samples must be non-negative, got {consumed_samples}")
        if global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")

        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.infinite = infinite

        if self.drop_last:
            self.num_samples = self.total_samples // self.micro_batch_size // self.data_parallel_size
        else:
            self.num_samples = math.ceil(math.ceil(self.total_samples / self.micro_batch_size) / self.data_parallel_size)

        self.total_size = self.num_samples * self.data_parallel_size * self.micro_batch_size
        if self.total_size == 0:
            required = self.micro_batch_size * self.data_parallel_size
            raise ValueError(f"drop_last=True would produce no samples: dataset has {self.total_samples}, but one data-parallel step requires {required}. Use a smaller micro batch / data-parallel size or set drop_last=False.")
        self.epoch = self.consumed_samples // self.total_size
        self.global_batch_size = global_batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        while True:
            indices = list(range(self.total_samples))
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                idx = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in idx]

            if not self.drop_last:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                indices = indices[: self.total_size]

            if self.consumed_samples % self.total_size != 0:
                indices = indices[self.consumed_samples % self.total_size :]

            for i in range(0, len(indices), self.micro_batch_size * self.data_parallel_size):
                start_idx = i + self.data_parallel_rank * self.micro_batch_size
                yield indices[start_idx : start_idx + self.micro_batch_size]

            self.epoch += 1
            self.consumed_samples += len(indices)

            if not self.infinite:
                break

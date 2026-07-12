# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025-2026 TeleAI and the TeleBoost contributors.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
import os


def ensure_directory_exists(filename, check_parent=True):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)

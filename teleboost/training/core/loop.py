# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

"""Small training-step scheduling helpers shared by trainer tests."""


def epoch_for_training_step(global_step: int, dataloader_len: int) -> int:
    """Return the zero-based epoch containing the one-based training step."""
    if dataloader_len <= 0:
        raise ValueError("dataloader_len must be > 0")
    return max(0, (int(global_step) - 1) // int(dataloader_len))


def should_continue_training(global_step: int, total_training_steps: int) -> bool:
    """Whether the one-based ``global_step`` still needs to run."""
    return int(global_step) <= int(total_training_steps)

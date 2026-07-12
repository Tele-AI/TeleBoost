# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025-2026 TeleAI and the TeleBoost contributors.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
from teleboost.engines.teletron.microbatches import build_num_microbatches_calculator

from .config import set_config
from .timers import Timers

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None


def set_global_args(args):
    assert args is not None
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_args(args):
    assert args is not None
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    _build_num_microbatches_calculator(args)
    _GLOBAL_ARGS = args
    _set_timers(args)
    _set_tensorboard_writer(args)


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    # _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)
    dit_model_config = set_config().get("model_config", None).get("dit", None)
    _GLOBAL_TIMERS = Timers(args, dit_model_config.config)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if hasattr(args, "tensorboard_dir") and args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter

            print("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print("WARNING: TensorBoard writing requested but is not available (are you using PyTorch 1.1.0 or later?), no TensorBoard logs will be written.", flush=True)


def _build_num_microbatches_calculator(args):
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, "num microbatches calculator")

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)

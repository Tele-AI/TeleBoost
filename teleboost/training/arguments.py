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
import argparse
import dataclasses
import os
from dataclasses import dataclass, field, fields
from typing import Optional, Union, get_args, get_origin, get_type_hints

from teleboost.training.utils import (
    _add_autoresume_args,
    _add_checkpointing_args,
    _add_data_args,
    _add_distributed_args,
    _add_experimental_args,
    _add_inference_args,
    _add_initialization_args,
    _add_learning_rate_args,
    _add_logging_args,
    _add_mixed_precision_args,
    _add_moe_args,
    _add_network_size_args,
    _add_regularization_args,
    _add_retro_args,
    _add_training_args,
    _add_transformer_engine_args,
    _add_validation_args,
)


@dataclass
class UnifiedArguments:
    show_args: Optional[bool] = field(default=False)

    model: str = field(
        default="",
    )

    timers: Optional[bool] = field(default=None)

    data_path: str = field(default="./checkpoint", metadata={"help": "Path to the dataset"})

    model_type: str = field(default="ModelType.encoder_or_decoder", metadata={"help": "Type of model to use"})

    pretrained_model_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained model"})
    micro_batch_size: int = field(default=1)
    weight_decay: float = field(default=1e-2)
    init_method_std: float = field(default=0.006)
    clip_grad: float = field(default=0.0)
    bf16: bool = field(default=False)
    lr: float = field(default=1e-5)
    lr_decay_style: str = "constant"
    lr_warmup_fraction: int = 0
    recompute_granularity: str = "full"
    recompute_method: str = "block"
    recompute_num_layers: int = 40
    no_rope_fusion: bool = field(default=False)
    distributed_timeout_minutes: int = 60
    use_distributed_optimizer: bool = field(default=False)

    log_interval: int = field(default=1)
    save_interval: int = field(default=100)
    eval_interval: int = field(default=10000)
    train_iters: int = field(default=10000)
    eval_iters: int = field(default=10000)

    tensorboard_queue_size: int = 10
    tensorboard_dir: Optional[str] = field(default="./logs")

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    # @dataclass
    # class ModelArguments:
    checkpoint_path: Optional[str] = field(default=None, metadata={"help": "Path to the model checkpoint"})
    # model config
    num_layers: int = field(
        default=1,
    )
    hidden_size: int = field(
        default=5120,
    )
    num_attention_heads: int = field(
        default=40,
    )
    seq_length: int = field(
        default=512,
    )
    max_position_embeddings: int = field(
        default=4096,
    )
    padded_vocab_size: int = field(
        default=0,
    )

    tensor_model_parallel_size: int = field(
        default=1,
    )
    context_parallel_size: int = field(
        default=1,
    )
    distributed_vae: bool = field(
        default=False,
    )
    distributed_vae_world_size: int = field(
        default=1,
    )
    producer_batch_size: int = field(
        default=1,
    )
    consumer_models_num: int = field(
        default=1,
    )
    encoder_dtype: str = field(default="bfloat16", metadata={"help": "Data type for the model. Options: 'float32', 'float16', 'bfloat16'."})
    # lora config
    lora: Optional[bool] = field(
        default=False,
    )
    lora_rank: int = field(
        default=8,
    )
    lora_alpha: int = field(
        default=32,
    )
    lora_dropout: float = field(
        default=0.05,
    )
    lora_target_modules: str = field(default="k,q,v,o", metadata={"help": "which module to apply lora, support q,k,v,o or single q currently"})
    lora_bias: str = field(default="none")
    lora_base_model_path: Optional[str] = field(default=None, metadata={"help": "If resume from checkpoint using lora, need to specify a base model path to load lora checkpoint correctly."})
    lora_task_type: str = field(default="FEATURE_EXTRACTION")
    flow_max_shift: float = field(
        default=1.15,
    )
    flow_shift: float = field(
        default=1.0,
    )
    flow_weighting_scheme: str = field(
        default="none",
    )
    flow_logit_mean: float = field(
        default=0.0,
    )
    flow_logit_std: float = field(
        default=1.0,
    )
    flow_mode_scale: float = field(
        default=1.29,
    )

    # @dataclass
    # class DataArguments:

    task_type: str = field(default="wan_i2v_prone", metadata={"help": "Type of task"})
    split: str = field(default="")
    num_workers: int = field(default=1)
    num_frames: int = field(default=9, metadata={"help": "numbers of frames to train, must be of 4n+1.Example:45"})
    video_resolution: tuple[int, int] = field(default=(1280, 720), metadata={"help": "video resolution to train.Example: 1280,720 (comma-separated)"})


def add_dataclass_arguments(parser, dataclass_type):
    """Add arguments from a dataclass to an ArgumentParser, handling existing arguments properly.

    Args:
        parser: ArgumentParser instance
        dataclass_type: A dataclass containing configuration fields
    """
    # Get type hints for the dataclass fields
    type_hints = get_type_hints(dataclass_type)

    # Create a dictionary of existing actions for quick lookup
    existing_actions = {}
    for action in parser._actions:
        if action.option_strings:
            # Store by the first option string (usually the long form like '--arg-name')
            existing_actions[action.option_strings[0]] = action

    # Process each field in the dataclass
    for field_obj in fields(dataclass_type):
        field_name = field_obj.name
        field_type = type_hints.get(field_name, field_obj.type)

        # Prepare argument name (convert to CLI style, e.g., 'input_file' -> '--input-file')
        arg_name = f"--{field_name.replace('_', '-')}"

        # Get the default value from the dataclass field
        default_value = field_obj.default if field_obj.default is not dataclasses.MISSING else None
        if default_value is None and field_obj.default_factory is not dataclasses.MISSING:
            default_value = field_obj.default_factory()

        # Handle existing arguments
        if arg_name in existing_actions:
            action = existing_actions[arg_name]

            # Update the existing argument with new default value and help text
            if default_value is not None:
                action.default = default_value

                # Update help text if metadata provides one
                if field_obj.metadata and "help" in field_obj.metadata:
                    action.help = field_obj.metadata["help"]
                elif action.help:
                    # Append default value to existing help
                    action.help = f"{action.help} (default: {default_value})"
                else:
                    action.help = f"(default: {default_value})"

            continue  # Skip adding new argument since it already exists

        # For new arguments, configure based on field type
        kwargs = {
            "dest": field_name,
            "default": default_value,
        }

        # Add help text from metadata if available
        if field_obj.metadata and "help" in field_obj.metadata:
            kwargs["help"] = field_obj.metadata["help"]

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            # Handle Optional[T] which is Union[T, None]
            args = get_args(field_type)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]

        # Handle different types
        if field_type is bool:
            # Boolean flags use action='store_true'/'store_false'
            kwargs["action"] = "store_true" if not default_value else "store_false"
        else:
            kwargs["type"] = field_type
            if default_value is not None:
                if "help" in kwargs:
                    kwargs["help"] = f"{kwargs['help']} (default: {default_value})"
                else:
                    kwargs["help"] = f"(default: {default_value})"

        # Special handling for video_resolution
        if field_name == "video_resolution":
            kwargs["type"] = str  # Keep as string, let the application parse it

        # Add the argument to the parser
        parser.add_argument(arg_name, **kwargs)


def parse_args(extra_args=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="TeleBoost arguments", allow_abbrev=False)

    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)
    parser = _add_experimental_args(parser)
    add_dataclass_arguments(parser, UnifiedArguments)
    if extra_args is not None:
        parser = extra_args(parser)

    args = parser.parse_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(f"rank {args.rank} world_size {args.world_size}")
    # args.world_size = 20
    if args.show_args is True:
        print_args(args)
        exit()

    return args


def print_args(args):
    from pprint import pprint

    pprint(vars(args))


if __name__ == "__main__":
    print_args()

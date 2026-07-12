# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025-2026 TeleAI and the TeleBoost contributors.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
import contextlib
import os

import torch
import torch.distributed as dist
from megatron.core import mpu

from teleboost.engines.teletron import get_args, get_attr_wrapped_model, get_model_config
from teleboost.engines.teletron.parallel_state import get_tensor_and_context_parallel_src_rank

NUM_BYTES_IN_MEGABYTE = 1024 * 1024
_TRANSFORMER_MODEL_GROUP = None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    loss_wo_w = output_tensor[1].mean()
    averaged_loss_wo_w = average_losses_across_data_parallel_group([loss_wo_w])
    loss_wo_w = loss_wo_w.unsqueeze(0)

    loss_f1 = output_tensor[2].mean()
    averaged_loss_f1 = average_losses_across_data_parallel_group([loss_f1])
    loss_f1 = loss_f1.unsqueeze(0)

    return loss, {"loss": averaged_loss[0], "loss_wo_w": averaged_loss_wo_w[0], "loss_f1": averaged_loss_f1[0]}


def deepspeed_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_only,
    zero_optimizer,
):
    if isinstance(model, list):
        model = model[0]
    config = get_model_config(model)
    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = deepspeed_forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                is_first_microbatch=(i == 0),
            )
            if not forward_only:
                deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = deepspeed_forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        is_first_microbatch=(num_microbatches == 1),
    )

    if not forward_only:
        deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config)

    if config.timers is not None:
        config.timers("forward-backward").stop()

    return forward_data_store


def deepspeed_forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    is_first_microbatch=False,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers("forward-compute", log_level=2).start()

    if is_first_microbatch and hasattr(model, "set_is_first_microbatch"):
        model.set_is_first_microbatch()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        output_tensor, loss_func = forward_step_func(data_iterator, model)

    output_tensor = loss_func(output_tensor)
    loss, loss_reduced = output_tensor
    # loss may be a Tensor or a [Tensor, Tensor] list
    if isinstance(loss, list | tuple):
        loss = [x / num_microbatches for x in loss]
    else:
        loss = loss / num_microbatches
    forward_data_store.append(loss_reduced)
    output_tensor = loss
    if config.timers is not None:
        config.timers("forward-compute").stop()

    if unwrap_output_tensor:
        return output_tensor

    if isinstance(output_tensor, list | tuple):
        return list(output_tensor)  # [loss_reject_scaled, loss_chosen_scaled]
    return [output_tensor]


def deepspeed_backward_step(zero_optimizer, input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    print(f"[Rank {torch.distributed.get_rank()}] enter deepspeed_backward")
    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        # output_tensor is guaranteed to be a list here (wrapped above, or already was one)
        output_tensor = [config.grad_scale_func(t) if torch.is_tensor(t) else t for t in output_tensor]
    loss_obj = output_tensor  # this is loss or [loss1, loss2]

    if isinstance(loss_obj, list | tuple):
        # Gradient-Decoupled DPO: each loss term gets its own backward + epilogue
        # so each backward's gradients are reduce-scattered to my-shard slices
        # immediately, freeing the full per-layer gradient tensors before the
        # next backward starts. Without the in-loop epilogue, full grads from
        # the first backward stay alive through the second backward's full
        # traversal — that's the ~50% peak-memory regression we are NOT willing
        # to take. Math equivalence to single backward(sum-of-losses) is empirically
        # verified within bf16 ULP (max|d| < 3e-4 on 5-iter Wan training).
        # NOTE: requires deepspeed <= 0.17.5; 0.17.6+ replaced the simple
        # multi-call epilogue with an all_grad_tensors state machine that
        # depends on DeepSpeedEngine driving is_gradient_accumulation_boundary.
        tensor_losses = [t for t in loss_obj if torch.is_tensor(t)]
        if len(tensor_losses) == 0:
            raise RuntimeError("loss_obj is list/tuple but contains no tensor loss.")

        use_cp_barrier = os.environ.get("TELEBOOST_DPO_SPLIT_BARRIER", "1") == "1"
        for idx, t in enumerate(tensor_losses):
            print(f"[DPO backward split {idx}] rank={torch.distributed.get_rank()} Before zero_optimizer.backward")
            zero_optimizer.backward(t, retain_graph=False)
            print(f"[DPO backward split {idx}] rank={torch.distributed.get_rank()} After zero_optimizer.backward")
            zero_optimizer.overlapping_partition_gradients_reduce_epilogue()
            if use_cp_barrier and mpu.get_context_parallel_world_size() > 1:
                torch.cuda.synchronize()
                torch.distributed.barrier(group=mpu.get_context_parallel_group())
    else:
        print(f"[DPO backward single] rank={torch.distributed.get_rank()} Before zero_optimizer.backward")
        zero_optimizer.backward(loss_obj, retain_graph=False)
        print(f"[DPO backward single] rank={torch.distributed.get_rank()} After zero_optimizer.backward")
        zero_optimizer.overlapping_partition_gradients_reduce_epilogue()

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers("backward-compute").stop()

    return input_tensor_grad


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def unpack_tensors(packed_tensor, intervals):
    features = tuple([packed_tensor[intervals[i - 1] : intervals[i]] for i in range(1, len(intervals))])
    return features


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def get_batch_on_this_tp_cp_rank_vast(data_iterator):
    def _broadcast(item):
        if item is not None:
            import torch.distributed as dist

            dist.get_rank()
            torch.distributed.broadcast(item, get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())

    if mpu.get_tensor_and_context_parallel_rank() == 0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        sizes_info = {}
        type_info = {}
        batch = dict(data)
        dtype = torch.bfloat16

        from teleboost.engines.teletron.parallel_state import get_comm_pair

        comm_pair = get_comm_pair()

        tensors_info = torch.empty((16), device=torch.cuda.current_device(), dtype=torch.int32)
        req = dist.irecv(tensors_info, comm_pair.producer, tag=0)
        req.wait()

        args = get_args()
        if args.distributed_vae:
            transformer_embedding_size = tensors_info[0] * tensors_info[1] * tensors_info[2]
            clip_embedding_size = tensors_info[3] * tensors_info[4] * tensors_info[5]
            first_img_embedding_size = tensors_info[6] * tensors_info[7] * tensors_info[8] * tensors_info[9] * tensors_info[10]
            video_embedding_size = tensors_info[11] * tensors_info[12] * tensors_info[13] * tensors_info[14] * tensors_info[15]

            recv_tensor = torch.empty((transformer_embedding_size + clip_embedding_size + first_img_embedding_size + video_embedding_size), device=torch.cuda.current_device(), dtype=torch.bfloat16)

            intervals = [0, transformer_embedding_size, transformer_embedding_size + clip_embedding_size, transformer_embedding_size + clip_embedding_size + first_img_embedding_size, transformer_embedding_size + clip_embedding_size + first_img_embedding_size + video_embedding_size]

            req = dist.irecv(recv_tensor, comm_pair.producer, tag=0)
            req.wait()

            context, clip_feature, img_y, latents = unpack_tensors(recv_tensor, intervals)
            context = context.view(tensors_info[0], tensors_info[1], tensors_info[2])
            clip_feature = clip_feature.view(tensors_info[3], tensors_info[4], tensors_info[5])
            img_y = img_y.view(tensors_info[6], tensors_info[7], tensors_info[8], tensors_info[9], tensors_info[10])
            latents = latents.view(tensors_info[11], tensors_info[12], tensors_info[13], tensors_info[14], tensors_info[15])
        else:
            pass

        batch["context"] = context
        batch["clip_feature"] = clip_feature
        batch["img_emb_y"] = img_y
        batch["latents"] = latents
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.to(torch.cuda.current_device())
        for key, tensor in batch.items():
            sizes_info[key] = tensor.size() if tensor is not None and isinstance(tensor, torch.Tensor) else len(tensor)
            type_info[key] = tensor.dtype if tensor is not None and isinstance(tensor, torch.Tensor) else type(tensor)

        # Step 2: broadcast the size info
        sizes_info = torch.distributed.broadcast_object_list([sizes_info], get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())
        type_info = torch.distributed.broadcast_object_list([type_info], get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())

        for key, tensor in batch.items():
            if isinstance(tensor, list):
                torch.distributed.broadcast_object_list(tensor, get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())
            elif isinstance(tensor, torch.Tensor):
                _broadcast(tensor)
            else:
                raise NotImplementedError(f"Unsupported data type: {type(tensor)}")

    else:
        sizes_info_list = [None]
        torch.distributed.broadcast_object_list(sizes_info_list, get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())
        type_info_list = [None]
        torch.distributed.broadcast_object_list(type_info_list, get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())

        batch = {}
        for key, value in sizes_info_list[0].items():
            dtype = type_info_list[0][key]

            if isinstance(dtype, torch.dtype):  # dtype is a torch dtype like torch.float32
                tensor = torch.empty(value, dtype=dtype, device=torch.cuda.current_device())
                _broadcast(tensor)
                batch[key] = tensor

            else:  # this is a list-typed object
                tensor = [None] * value
                torch.distributed.broadcast_object_list(tensor, src=get_tensor_and_context_parallel_src_rank(), group=mpu.get_tensor_and_context_parallel_group())
                batch[key] = tensor

    return batch


###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title="Transformer-Engine")

    group.add_argument("--fp8-format", default=None, choices=["e4m3", "hybrid"], help="Which fp8 format scheme to use for FP8 tensors in the forward and backward pass", dest="fp8")
    group.add_argument("--fp8-margin", type=int, default=0, help="Scaling margin for fp8", dest="fp8_margin")
    group.add_argument("--fp8-interval", type=int, default=1, help="Scaling update interval for fp8", dest="fp8_interval")
    group.add_argument("--fp8-amax-history-len", type=int, default=1, help="Number of steps for which amax history is recorded per tensor", dest="fp8_amax_history_len")
    group.add_argument("--fp8-amax-compute-algo", default="most_recent", choices=["most_recent", "max"], help="Algorithm for computing amax from history", dest="fp8_amax_compute_algo")
    group.add_argument("--no-fp8-wgrad", action="store_false", help="Execute wgrad in higher precision even for FP8 runs", dest="fp8_wgrad")
    group.add_argument("--transformer-impl", default="transformer_engine", choices=["local", "transformer_engine"], help="Which Transformer implementation to use.")
    group.add_argument("--use-fused-rmsnorm", action="store_true", help="Enable fused rmsnorm kernel")

    return parser


def _add_inference_args(parser):
    group = parser.add_argument_group(title="inference")

    group.add_argument("--inference-batch-times-seqlen-threshold", type=int, default=512, help="During inference, if batch-size times sequence-length is smaller than this threshold then we will not use pipelining, otherwise we will.")
    group.add_argument("--max-tokens-to-oom", type=int, default=12000, help="Maximum number of tokens during inferencetokens here is # in prompt + # to generateAllows us to throw an error before OOM crashes server")
    group.add_argument("--output-bert-embeddings", action="store_true", help="Output Bert embeddings (via mean pooling) from model, rather than its binary head output or entire hidden batch.")
    group.add_argument("--bert-embedder-type", default="megatron", choices=["megatron", "huggingface"], help="Select either Megatron or Huggingface as the Bert embedder.")

    return parser


def _add_retro_args(parser):
    group = parser.add_argument_group(title="retro")

    group.add_argument("--retro-project-dir", default=None, help="Retro project directory, which contains the preprocessed data for pretraining. This directory is built during preprocessing (see tools/retro/README.md), and contains subdirectories for the chunk database and pretraining neighbors.")
    group.add_argument("--retro-add-retriever", action="store_true", default=False, help="Add a retriever to the transformer, for use in pretraining a Retro model.")
    group.add_argument("--retro-cyclic-train-iters", type=int, default=None, help="Set number of training iterations for cyclic Retro training.")
    group.add_argument("--retro-encoder-layers", type=int, default=2, help="Number of layers to use for the retrieval encoder.")
    group.add_argument("--retro-encoder-hidden-dropout", type=float, default=0.1, help="Hidden dropout for retrieval encoder.")
    group.add_argument("--retro-encoder-attention-dropout", type=float, default=0.1, help="Attention dropout for retrieval encoder.")
    group.add_argument("--retro-num-neighbors", type=int, default=2, help="Number of neighbors to retrieve during pretraining.")
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2, help="Number of chunks to retrieve from the retrieval database.")
    group.add_argument("--retro-attention-gate", type=float, default=1, help="Gated cross attention.")
    group.add_argument("--retro-no-verify-neighbor-count", action="store_false", dest="retro_verify_neighbor_count", help="Skip verifying that len(GPT dataset) == len(saved neighbors).")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", "Retro args must be prefixed with '--retro-*', for consistent styling. Please fix '{}'.".format(", ".join(action.option_strings))

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title="network size")

    group.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers.")
    group.add_argument("--encoder-num-layers", type=int, default=None, help="Number of encoder transformer layers.")
    group.add_argument("--decoder-num-layers", type=int, default=None, help="Number of decoder transformer layers.")
    group.add_argument("--hidden-size", type=int, default=None, help="Tansformer hidden size.")
    group.add_argument("--ffn-hidden-size", type=int, default=None, help="Transformer Feed-Forward Network hidden size. This is set to 4*hidden-size if not provided")
    group.add_argument("--num-attention-heads", type=int, default=None, help="Number of transformer attention heads.")
    group.add_argument("--kv-channels", type=int, default=None, help="Projection weights dimension in multi-head attention. This is set to    args.hidden_size // args.num_attention_heads if not provided.")
    group.add_argument("--group-query-attention", action="store_true", help="Use group-query attention.")
    group.add_argument("--num-query-groups", type=int, default=1)

    group.add_argument("--max-position-embeddings", type=int, default=None, help="Maximum number of position embeddings to use. This is the size of position embedding.")
    group.add_argument("--position-embedding-type", type=str, default="learned_absolute", choices=["learned_absolute", "rope"], help="Position embedding type.")
    group.add_argument("--use-rotary-position-embeddings", action="store_true", help="Use rotary positional embeddings or not. Deprecated: use --position-embedding-type")
    group.add_argument("--rotary-percent", type=float, default=1.0, help="Percent of rotary dimension to use, default 100%%")
    group.add_argument("--rotary-interleaved", action="store_true", help="Use interleaved rotary embedding.")
    group.add_argument("--rotary-seq-len-interpolation-factor", type=int, default=None, help="Sequence length interpolation factor for rotary embeddings.")
    group.add_argument("--no-position-embedding", action="store_false", help="Disable position embedding. Deprecated: use --position-embedding-type", dest="add_position_embedding")
    group.add_argument("--make-vocab-size-divisible-by", type=int, default=128, help="Pad the vocab size to be divisible by this value.This is added for computational efficieny reasons.")
    group.add_argument("--normalization", default="LayerNorm", choices=["LayerNorm", "RMSNorm"], help="Which normalization technique to use.")
    group.add_argument("--norm-epsilon", type=float, default=1e-5, help="Epsilon for layer norm and RMS norm.")
    group.add_argument("--apply-layernorm-1p", action="store_true", help="Adjust LayerNorm weights such that they are centered around zero. This improves numerical stability.")
    group.add_argument("--apply-residual-connection-post-layernorm", action="store_true", help="If set, use original BERT residula connection ordering.")
    group.add_argument("--openai-gelu", action="store_true", help="Use OpenAIs GeLU implementation. This optionshould not be used unless for backward compatibilityreasons.")
    group.add_argument("--squared-relu", action="store_true", help="Use squared relu activation instead of default gelu")
    group.add_argument("--swiglu", action="store_true", help="Use gated linear units and SiLU activation instead of default gelu")
    group.add_argument("--onnx-safe", type=bool, required=False, help="Use workarounds for known problems with Torch ONNX exporter")
    group.add_argument("--bert-no-binary-head", action="store_false", help="Disable BERT binary head.", dest="bert_binary_head")
    (group.add_argument("--untie-embeddings-and-output-weights", action="store_true", help="Untie embeddings and output weights."),)
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title="logging")
    group.add_argument("--producer-log-level", type=int, default=2, choices=range(1, 3), help="logging level to producer detail.    1: DEBUG LEVEL    2: INFO LEVEL")
    group.add_argument("--log-params-norm", action="store_true", help="If set, calculate and log parameters norm.")
    group.add_argument("--log-num-zeros-in-grad", action="store_true", help="If set, calculate and log the number of zeros in gradient.")
    group.add_argument("--log-throughput", action="store_true", help="If set, calculate and log throughput per GPU.")
    group.add_argument("--log-progress", action="store_true", help="If set, log progress (in terms of number of processed tokens and number of floating-point operations) to progress.txt file in checkpoint directory.")
    group.add_argument(
        "--timing-log-level",
        type=int,
        default=0,
        choices=range(0, 3),
        help="Granularity level to measure and report timing. "
        "   0: report only iteration time and make sure timing "
        "      does not introduce extra overhead."
        "   1: report timing for operations that are executed "
        "      very limited times (basically once) during "
        "      each iteration (such as gradient all-reduce) "
        "   2: report timing for operations that migh be "
        "      executed numerous times during each iteration. "
        "Note that setting the level to 1 or 2 might "
        "cause increase in iteration time.",
    )
    group.add_argument(
        "--no-barrier-with-level-1-timing",
        action="store_false",
        help="If not set, use barrier with level 1 time measurements. Note that this is up to the user to make sure calling barrier with their timers will not result in hangs. This can happen if for example the user adds a level 1 timer that is not called by all ranks.",
        dest="barrier_with_L1_time",
    )
    group.add_argument("--timing-log-option", type=str, default="minmax", choices=["max", "minmax", "all"], help="Options for logging timing:  max: report the max timing across all ranks  minmax: report min and max timings across all ranks  all: report timings of all ranks.")
    group.add_argument("--tensorboard-log-interval", type=int, default=1, help="Report to tensorboard interval.")
    group.add_argument("--tensorboard-queue-size", type=int, default=1000, help="Size of the tensorboard queue for pending events and summaries before one of the ‘add’ calls forces a flush to disk.")
    group.add_argument("--log-timers-to-tensorboard", action="store_true", help="If set, write timers to tensorboard.")
    group.add_argument("--log-batch-size-to-tensorboard", action="store_true", help="If set, write batch-size to tensorboard.")
    group.add_argument("--no-log-gradient-norm-to-tensorboard", action="store_false", help="Disable gradient norm logging to tensorboard.", dest="log_gradient_norm_to_tensorboard")
    group.add_argument("--no-log-learnig-rate-to-tensorboard", action="store_false", help="Disable learning rate logging to tensorboard.", dest="log_learning_rate_to_tensorboard")
    group.add_argument("--no-log-loss-scale-to-tensorboard", action="store_false", help="Disable loss-scale logging to tensorboard.", dest="log_loss_scale_to_tensorboard")
    group.add_argument("--log-validation-ppl-to-tensorboard", action="store_true", help="If set, write validation perplexity to tensorboard.")
    group.add_argument("--log-memory-to-tensorboard", action="store_true", help="Enable memory logging to tensorboard.")
    group.add_argument("--log-world-size-to-tensorboard", action="store_true", help="Enable world size logging to tensorboard.")
    group.add_argument("--wandb-project", type=str, default="", help="The wandb project name. Ignore wandb by default.")
    group.add_argument("--wandb-exp-name", type=str, default="", help="The wandb experiment name.")
    group.add_argument("--wandb-save-dir", type=str, default="", help="Path to save the wandb results locally.")
    group.add_argument(
        "--enable-one-logger",
        action="store_true",
        help=("If set, use the optional NVIDIA one_logger integration to track E2E metrics. one_logger is not distributed or supported by TeleBoost; install it only from an artifact source that your organization is authorized to access."),
    )
    group.add_argument("--one-logger-project", type=str, default="e2e-tracking", help="The one-logger project name. Will ignore if --enable-one-logger is not set")
    group.add_argument("--one-logger-entity", type=str, default="hwinf_dcm", help="The one-logger username or team name. Will ignore if --enable-one-logger is not set")
    group.add_argument("--one-logger-run-name", type=str, default=None, help="The one-logger run name displayed. Will ignore if --enable-one-logger is not set")
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title="regularization")

    group.add_argument("--attention-dropout", type=float, default=0.1, help="Post attention dropout probability.")
    group.add_argument("--hidden-dropout", type=float, default=0.1, help="Dropout probability for hidden state transformer.")
    group.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay coefficient for L2 regularization.")
    group.add_argument("--start-weight-decay", type=float, help="Initial weight decay coefficient for L2 regularization.")
    group.add_argument("--end-weight-decay", type=float, help="End of run weight decay coefficient for L2 regularization.")
    group.add_argument("--weight-decay-incr-style", type=str, default="constant", choices=["constant", "linear", "cosine"], help="Weight decay increment function.")
    group.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping based on global L2 norm.")
    group.add_argument("--adam-beta1", type=float, default=0.9, help="First coefficient for computing running averages of gradient and its square")
    group.add_argument("--adam-beta2", type=float, default=0.999, help="Second coefficient for computing running averages of gradient and its square")
    group.add_argument("--adam-eps", type=float, default=1e-08, help="Term added to the denominator to improvenumerical stability")
    group.add_argument("--sgd-momentum", type=float, default=0.9, help="Momentum factor for sgd")
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument("--config-path", type=str, default=None)
    group.add_argument("--micro-batch-size", type=int, default=None, help="Batch size per model instance (local batch size). Global batch size is local batch size times data parallel size times number of micro batches.")
    group.add_argument("--batch-size", type=int, default=None, help="Old batch size parameter, do not use. Use --micro-batch-size instead")
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Training batch size. If set, it should be a multiple of micro-batch-size times data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size as the global batch size. This choice will result in 1 for number of micro-batches.",
    )
    group.add_argument(
        "--rampup-batch-size",
        nargs="*",
        default=None,
        help="Batch size ramp up with the following values:"
        "  --rampup-batch-size <start batch size> "
        "                      <batch size incerement> "
        "                      <ramp-up samples> "
        "For example:"
        "   --rampup-batch-size 16 8 300000 \\ "
        "   --global-batch-size 1024"
        "will start with global batch size 16 and over "
        " (1024 - 16) / 8 = 126 intervals will increase"
        "the batch size linearly to 1024. In each interval"
        "we will use approximately 300000 / 126 = 2380 samples.",
    )
    group.add_argument("--recompute-activations", action="store_true", help="recompute activation to allow for training with larger models, sequences, and batch sizes.")
    group.add_argument(
        "--recompute-granularity",
        type=str,
        default=None,
        choices=["full", "selective"],
        help="Checkpoint activations to allow for training with larger models, sequences, and batch sizes. It is supported at two granularities 1) full: whole transformer layer is recomputed, 2) selective: core attention part of the transformer layer is recomputed.",
    )
    group.add_argument("--no-check-for-nan-in-loss-and-grad", action="store_false", help="Check for NaNs in loss and grad", dest="check_for_nan_in_loss_and_grad")
    group.add_argument("--distribute-saved-activations", action="store_true", help="If set, distribute recomputed activations across model parallel group.")
    group.add_argument(
        "--recompute-method",
        type=str,
        default=None,
        choices=["uniform", "block"],
        help="1) uniform: uniformly divide the total number of "
        "Transformer layers and recompute the input activation of "
        "each divided chunk at specified granularity, "
        "2) recompute the input activations of only a set number of "
        "individual Transformer layers per pipeline stage and do the "
        "rest without any recomputing at specified granularity"
        "default) do not apply activations recompute to any layers",
    )
    group.add_argument("--recompute-num-layers", type=int, default=None, help="1) uniform: the number of Transformer layers in each uniformly divided recompute unit, 2) block: the number of individual Transformer layers to recompute within each pipeline stage.")
    group.add_argument("--activation-offload", action="store_true", help="enable activation cpu offload on transformer forward, requires using activation checkpointing (recompute)")
    group.add_argument("--no-clone-scatter-output-in-embedding", action="store_false", help="If not set, clone the output of the scatter in embedding layer to GC original tensor.", dest="clone_scatter_output_in_embedding")
    group.add_argument("--consumer-profile", action="store_true", help="Enable transformer blocks torch profiling.")
    group.add_argument("--producer-profile", action="store_true", help="Enable data producer torch profiling.")
    group.add_argument("--profile-path", type=str, default=None, help="Directory to save torch profiling traces.")
    group.add_argument(
        "--profile",
        action="store_true",
        help="Enable nsys profiling. When using this option, nsys options should be specified in commandline. An example nsys commandline is `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop`.",
    )
    group.add_argument("--profile-step-start", type=int, default=10, help="Global step to start profiling.")
    group.add_argument("--profile-step-end", type=int, default=12, help="Global step to stop profiling.")
    group.add_argument("--profile-ranks", nargs="+", type=int, default=[0], help="Global ranks to profile.")
    group.add_argument("--tp-comm-overlap", action="store_true", help="Enables the  overlap of Tensor parallel communication and GEMM kernels.")
    group.add_argument("--tp-comm-overlap-cfg", type=str, default=None, help="Config file when tp_comm_overlap is enabled.")
    group.add_argument("--disable-tp-comm-overlap-ag", action="store_false", help=("Disables the All-Gather overlap with GEMM by pipelining the GEMM and All-Gather."), dest="tp_comm_overlap_ag")
    group.add_argument("--disable-tp-comm-overlap-rs", action="store_false", help=("Disables the Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter."), dest="tp_comm_overlap_rs")
    group.add_argument("--disable-tp-comm-bulk-dgrad", action="store_false", help="Disables the All-Gather overlap with bprop activation gradient GEMM.", dest="tp_comm_bulk_dgrad")
    group.add_argument("--disable-tp-comm-bulk-wgrad", action="store_false", help="Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.", dest="tp_comm_bulk_wgrad")
    group.add_argument("--use-cpu-initialization", action="store_true", default=None, help="If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.")
    group.add_argument("--empty-unused-memory-level", default=0, type=int, choices=[0, 1, 2], help="Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.0=off, 1=moderate, 2=aggressive.")

    # deprecated
    group.add_argument("--checkpoint-activations", action="store_true", help="Checkpoint activation to allow for training with larger models, sequences, and batch sizes.")
    group.add_argument("--train-iters", type=int, default=None, help="Total number of iterations to train over all training runs. Note that either train-iters or train-samples should be provided.")
    group.add_argument("--train-samples", type=int, default=None, help="Total number of samples to train over all training runs. Note that either train-iters or train-samples should be provided.")
    group.add_argument("--log-interval", type=int, default=100, help="Report loss and timing interval.")
    group.add_argument("--exit-interval", type=int, default=None, help="Exit the program after the iteration is divisible by this value.")
    group.add_argument("--exit-duration-in-mins", type=int, default=None, help="Exit the program after this many minutes.")
    group.add_argument("--exit-signal-handler", action="store_true", help="Dynamically save the checkpoint and shutdown the training if SIGTERM is received")
    group.add_argument("--tensorboard-dir", type=str, default=None, help="Write TensorBoard logs to this directory.")
    group.add_argument("--no-masked-softmax-fusion", action="store_false", help="Disable fusion of query_key_value scaling, masking, and softmax.", dest="masked_softmax_fusion")
    group.add_argument("--no-bias-gelu-fusion", action="store_false", help="Disable bias and gelu fusion.", dest="bias_gelu_fusion")
    group.add_argument("--no-bias-swiglu-fusion", action="store_false", help="Disable bias and swiglu fusion, the fusion is available only when using megatron-core.", dest="bias_swiglu_fusion")
    group.add_argument("--no-bias-dropout-fusion", action="store_false", help="Disable bias and dropout fusion.", dest="bias_dropout_fusion")
    group.add_argument("--no-rope-fusion", action="store_false", help="Disable rope fusion, the fusion is available only when using megatron-core.", dest="apply_rope_fusion")
    group.add_argument("--use-flash-attn", action="store_true", help="use FlashAttention implementation of attention. https://arxiv.org/abs/2205.14135")
    group.add_argument("--disable-bias-linear", action="store_false", help="Disable bias in the linear layers", dest="add_bias_linear")
    group.add_argument("--add-qkv-bias", action="store_true", help="Enable bias only in the QKV linear layers", dest="add_qkv_bias")
    group.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer function")
    group.add_argument("--dataloader-type", type=str, default=None, choices=["common", "single", "cyclic", "external"], help="Single pass vs multiple pass data loader")
    group.add_argument("--no-async-tensor-model-parallel-allreduce", action="store_false", help="Disable asynchronous execution of tensor-model-parallel all-reduce with weight gradient compuation of a column-linear layer.", dest="async_tensor_model_parallel_allreduce")
    group.add_argument("--no-persist-layer-norm", action="store_true", help="Disable using persistent fused layer norm kernel. This kernel supports only a set of hidden sizes. Please check persist_ln_hidden_sizes if your hidden size is supported.")
    group.add_argument("--sequence-parallel", action="store_true", help="Enable sequence parallel optimization.")
    group.add_argument("--no-gradient-accumulation-fusion", action="store_false", help="Disable fusing gradient accumulation to weight gradient computation of linear layers", dest="gradient_accumulation_fusion")
    group.add_argument("--use-mcore-models", action="store_true", help="Use the implementation from megatron core")
    group.add_argument(
        "--manual-gc",
        action="store_true",
        help="Disable the threshold-based default garbage "
        "collector and trigger the garbage collection manually. "
        "Manual garbage collection helps to align the timing of "
        "the collection across ranks which mitigates the impact "
        "of CPU-associated jitters. When the manual gc is enabled, "
        "garbage collection is performed only at the start and the "
        "end of the validation routine by default.",
    )
    group.add_argument("--manual-gc-interval", type=int, default=0, help="Training step interval to trigger manual garbage collection. When the value is set to 0, garbage collection is not triggered between training steps.")
    group.add_argument("--no-manual-gc-eval", action="store_false", help="When using manual garbage collection, disable garbage collection at the start and the end of each evaluation run.", dest="manual_gc_eval")
    group.add_argument("--disable-tp-comm-split-ag", action="store_false", help="Disables the All-Gather overlap with fprop GEMM.", dest="tp_comm_split_ag")
    group.add_argument("--disable-tp-comm-split-rs", action="store_false", help="Disables the Reduce-Scatter overlap with fprop GEMM.", dest="tp_comm_split_rs")

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")

    group.add_argument("--seed", type=int, default=1234, help="Random seed used for python, numpy, pytorch, and cuda.")
    group.add_argument("--data-parallel-random-init", action="store_true", help="Enable random initialization of params across data parallel ranks")
    group.add_argument("--init-method-std", type=float, default=0.02, help="Standard deviation of the zero mean normal distribution used for weight initialization.")
    group.add_argument("--init-method-xavier-uniform", action="store_true", help="Enable Xavier uniform parameter initialization")

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title="learning rate")

    group.add_argument("--lr", type=float, default=None, help="Initial learning rate. Depending on decay style and initial warmup, the learning rate at each iteration would be different.")
    group.add_argument("--lr-decay-style", type=str, default="linear", choices=["constant", "linear", "cosine", "inverse-square-root"], help="Learning rate decay function.")
    group.add_argument("--lr-decay-iters", type=int, default=None, help="number of iterations to decay learning rate over, If None defaults to `--train-iters`")
    group.add_argument("--lr-decay-samples", type=int, default=None, help="number of samples to decay learning rate over, If None defaults to `--train-samples`")
    group.add_argument("--lr-warmup-fraction", type=float, default=None, help="fraction of lr-warmup-(iters/samples) to use for warmup (as a float)")
    group.add_argument("--lr-warmup-iters", type=int, default=0, help="number of iterations to linearly warmup learning rate over.")
    group.add_argument("--lr-warmup-samples", type=int, default=0, help="number of samples to linearly warmup learning rate over.")
    group.add_argument("--lr-warmup-init", type=float, default=0.0, help="Initial value for learning rate warmup. The scheduler starts warmup from this value.")
    group.add_argument("--warmup", type=int, default=None, help="Old lr warmup argument, do not use. Use one of the--lr-warmup-* arguments above")
    group.add_argument("--min-lr", type=float, default=0.0, help="Minimum value for learning rate. The schedulerclip values below this threshold.")
    group.add_argument(
        "--override-opt_param-scheduler",
        action="store_true",
        help="Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Notethat all the above values will be reset.",
    )
    group.add_argument("--use-checkpoint-opt_param-scheduler", action="store_true", help="Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.")
    group.add_argument("--decoupled-lr", type=float, default=None, help="Separate learning rate for the input and output layer")
    group.add_argument("--decoupled-min-lr", type=float, default=None, help="Minimum value for learning rate for the input and output layer. The schedulerclip values below this threshold")

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title="checkpointing")

    group.add_argument("--save", type=str, default=None, help="Output directory to save checkpoints to.")
    group.add_argument("--save-interval", type=int, default=None, help="Number of iterations between checkpoint saves.")
    group.add_argument("--no-save-optim", action="store_true", default=None, help="Do not save current optimizer.")
    group.add_argument("--no-save-rng", action="store_true", default=None, help="Do not save current rng state.")
    group.add_argument("--load", type=str, default=None, help="Directory containing a model checkpoint.")
    group.add_argument("--no-load-optim", action="store_true", default=None, help="Do not load optimizer when loading checkpoint.")
    group.add_argument("--no-load-rng", action="store_true", default=None, help="Do not load rng state when loading checkpoint.")
    group.add_argument("--finetune", action="store_true", help="Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.")
    group.add_argument("--pretrained-checkpoint", type=str, default=None, help="Directory containing a pretrained model checkpoint for finetuning.")
    group.add_argument("--ckpt-step", type=int, default=None, help="Checkpoint step to load model from.")
    group.add_argument("--no-initialization", action="store_false", help="Do not perform initialization when building model, can reduce startup time when definitely loading from a checkpoint", dest="perform_initialization")
    group.add_argument("--use-checkpoint-args", action="store_true", help="Override any command line arguments with arguments from the checkpoint")
    group.add_argument("--exit-on-missing-checkpoint", action="store_true", help="If '--load' is set, but checkpoint is not found (e.g., path typo), then exit instead of random initialization.")
    group.add_argument("--use-dist-ckpt", action="store_true", help="Use distributed checkpoint format.")
    group.add_argument("--auto-detect-ckpt-format", action="store_true", help="Determine if the checkpoint format is in legacy or distributed format. If False, expects distributed checkpoint iff args.use_dist_ckpt. Might slow down loading a bit (double rank0 ckpt load).")
    group.add_argument("--dist-ckpt-format", type=str, default="torch_dist", choices=["zarr", "torch_dist"], help="Distributed checkpoint format to use.")
    group.add_argument("--ckpt-fully-parallel-save", action="store_true", help="Apply full save parallelization across DP for distributed checkpoints. Depending on ckpt format might increase number of files in the checkpoint.")
    group.add_argument("--with-ema", action="store_true", help="save checkpoint with ema model")
    group.add_argument("--ema-decay", type=float, default=0.9999, help="decay of ema model")

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title="mixed precision")

    group.add_argument("--fp16", action="store_true", help="Run model in fp16 mode.")
    group.add_argument("--bf16", action="store_true", help="Run model in bfloat16 mode.")
    group.add_argument("--loss-scale", type=float, default=None, help="Static loss scaling, positive power of 2 values can improve fp16 convergence. If None, dynamicloss scaling is used.")
    group.add_argument("--initial-loss-scale", type=float, default=2**32, help="Initial loss-scale for dynamic loss scaling.")
    group.add_argument("--min-loss-scale", type=float, default=1.0, help="Minimum loss scale for dynamic loss scaling.")
    group.add_argument("--loss-scale-window", type=float, default=1000, help="Window over which to raise/lower dynamic scale.")
    group.add_argument("--hysteresis", type=int, default=2, help="hysteresis for dynamic loss scaling")
    group.add_argument("--fp32-residual-connection", action="store_true", help="Move residual connections to fp32.")
    group.add_argument("--apply-query-key-layer-scaling", action="store_true", help="Scale Q * K^T by 1 / layer-number. Useful for fp16 training.")
    group.add_argument("--attention-softmax-in-fp32", action="store_true", help="Run attention masking and softmax in fp32. This flag is ignored unless --no-query-key-layer-scaling is specified.")
    group.add_argument("--accumulate-allreduce-grads-in-fp32", action="store_true", help="Gradient accumulation and all-reduce in fp32.")
    group.add_argument("--fp16-lm-cross-entropy", action="store_true", help="Move the cross entropy unreduced loss calculationfor lm head to fp16.")

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title="distributed")

    group.add_argument("--tensor-model-parallel-size", type=int, default=1, help="Degree of tensor model parallelism.")
    group.add_argument("--pipeline-model-parallel-size", type=int, default=1, help="Degree of pipeline model parallelism.")
    group.add_argument("--pipeline-model-parallel-split-rank", type=int, default=None, help="Rank where encoder and decoder should be split.")
    group.add_argument("--model-parallel-size", type=int, default=None, help="Old model parallel argument, do not use. Use --tensor-model-parallel-size instead.")
    group.add_argument("--num-layers-per-virtual-pipeline-stage", type=int, default=None, help="Number of layers per virtual pipeline stage")
    group.add_argument("--no-overlap-p2p-communication", action="store_false", help="overlap pipeline parallel communication with forward and backward chunks", dest="overlap_p2p_comm")
    group.add_argument("--distributed-backend", default="nccl", choices=["nccl", "gloo"], help="Which backend to use for distributed training.")
    group.add_argument("--distributed-timeout-minutes", type=int, default=10, help="Timeout minutes for torch.distributed.")
    group.add_argument("--overlap-grad-reduce", action="store_true", default=False, help="If set, overlap DDP grad reduce.")
    group.add_argument("--no-delay-grad-reduce", action="store_false", help="If not set, delay / synchronize grad reductions in all but first PP stage.", dest="delay_grad_reduce")
    group.add_argument("--overlap-param-gather", action="store_true", default=False, help="If set, overlap param all-gather in distributed optimizer.")
    group.add_argument("--delay-param-gather", action="store_true", default=False, help="If set, delay / synchronize param all-gathers in all but first PP stage.")
    group.add_argument("--no-scatter-gather-tensors-in-pipeline", action="store_false", help="If not set, use scatter/gather to optimize communication of tensors in pipeline.", dest="scatter_gather_tensors_in_pipeline")
    group.add_argument("--use-ring-exchange-p2p", action="store_true", default=False, help="If set, use custom-built ring exchange for p2p communications. Note that this option will require a custom built image that support ring-exchange p2p.")
    group.add_argument("--local_rank", type=int, default=None, help="local rank passed from distributed launcher.")
    group.add_argument("--lazy-mpu-init", type=bool, required=False, help="If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead.Also turns on --use-cpu-initialization flag. This is for external DDP manager.")
    group.add_argument("--standalone-embedding-stage", action="store_true", default=False, help="If set, *input* embedding layer is placed on its own pipeline stage, without any transformer layers. (For T5, this flag currently only affects the encoder embedding.)")
    group.add_argument("--use-distributed-optimizer", action="store_true", help="Use distributed optimizer.")
    group.add_argument("--use-zero2", action="store_true", help="Use DeepSpeed Zero2 distributed optimizer.")
    group.add_argument("--context-parallel-size", type=int, default=1, help="Degree of context parallelism.")
    group.add_argument(
        "--nccl-communicator-config-path", type=str, default=None, help="Path to the yaml file with NCCL communicator configurations. The number of min/max thread groups and thread group cluster size of each communicator can be configured by setting `min_ctas`, `max_ctas`, and `cga_cluster_size`."
    )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title="validation")

    group.add_argument("--eval-iters", type=int, default=100, help="Number of iterations to run for evaluationvalidation/test for.")
    group.add_argument("--eval-interval", type=int, default=1000, help="Interval between running evaluation on validation set.")
    group.add_argument("--test-mode", action="store_true", help="Run all real-time test alongside the experiment.")
    group.add_argument("--skip-train", action="store_true", default=False, help="If set, bypass the training loop, optionally do evaluation for validation/test, and exit.")

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title="data and dataloader")

    group.add_argument(
        "--data-path",
        nargs="*",
        default=None,
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-weight dataset1-path dataset2-weight "
        "dataset2-path ... It is used with --split when a "
        "single dataset used for all three: train, valid "
        "and test. It is exclusive to the other "
        "--*-data-path args",
    )
    group.add_argument("--split", type=str, default="969, 30, 1", help="Comma-separated list of proportions for training, validation, and test split. For example the split `90,5,5` will use 90%% of data for training, 5%% for validation and 5%% for test.")
    group.add_argument("--train-data-path", nargs="*", default=None, help="Path to the training dataset. Accepted format:1) a single data path, 2) multiple datasets in theform: dataset1-weight dataset1-path dataset2-weight dataset2-path ...")
    group.add_argument("--valid-data-path", nargs="*", default=None, help="Path to the validation dataset. Accepted format:1) a single data path, 2) multiple datasets in theform: dataset1-weight dataset1-path dataset2-weight dataset2-path ...")
    group.add_argument("--test-data-path", nargs="*", default=None, help="Path to the test dataset. Accepted format:1) a single data path, 2) multiple datasets in theform: dataset1-weight dataset1-path dataset2-weight dataset2-path ...")
    group.add_argument("--data-cache-path", default=None, help="Path to a directory to hold cached index files.")
    group.add_argument("--no-mmap-bin-files", action="store_false", help="Disable mmap-ing of .bin files.", dest="mmap_bin_files")
    group.add_argument("--mock-data", action="store_true", help="Skip data loading and validation and opt for artificial generation of mock data when an implementation is available.")
    group.add_argument("--num-workers", type=int, default=2, help="Dataloader number of workers.")

    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title="autoresume")

    group.add_argument("--adlr-autoresume", action="store_true", help="Enable autoresume on adlr cluster.")
    group.add_argument("--adlr-autoresume-interval", type=int, default=1000, help="Intervals over which check for autoresumetermination signal")

    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    group.add_argument("--expert-model-parallel-size", type=int, default=1, help="Degree of expert model parallelism.")
    group.add_argument("--num-experts", type=int, default=None, help="Number of Experts in MoE (None means no MoE)")
    group.add_argument(
        "--moe-router-load-balancing-type",
        type=str,
        choices=["aux_loss", "sinkhorn", "none"],
        default="aux_loss",
        help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".',
    )
    group.add_argument("--moe-router-topk", type=int, default=2, help="Number of experts to route to for each token. The default is 2.")
    group.add_argument(
        "--moe-grouped-gemm",
        action="store_true",
        help="When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).",
    )
    group.add_argument("--moe-aux-loss-coeff", type=float, default=0.0, help="Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.")
    group.add_argument("--moe-z-loss-coeff", type=float, default=None, help="Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.")
    group.add_argument("--moe-input-jitter-eps", type=float, default=None, help="Add noise to the input tensor by applying jitter with a specified epsilon value.")
    group.add_argument("--moe-token-dropping", action="store_true", help="This feature involves selectively dropping and padding tokens for each expert to achieve a specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note: Currently unsupported.")
    group.add_argument("--moe-token-dispatcher-type", type=str, choices=["allgather", "alltoall"], default="allgather", help=".")
    group.add_argument("--moe-per-layer-logging", action="store_true", help="Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.")

    return parser


def _add_experimental_args(parser):
    group = parser.add_argument_group(title="experimental")

    group.add_argument(
        "--spec",
        type=str,
        default=None,
        nargs="*",
        help="Specify the <module_location function_name> pair "
        "that returns a spec to customize a model, transformer "
        "block, or transformer layer, depending on the use case."
        "To use local spec specify local as the argument."
        "For more details, see the model class, "
        "`transformer_block.py`, or `transformer_layer.py`",
    )
    group.add_argument("--yaml-cfg", type=str, default=None, help="Config file to add additional arguments")

    return parser


META_TYPE = "__type__"
META_TENSOR = "tensor"
META_DICT = "dict"


def build_meta_tree(obj):
    """
    Build a meta tree from the batch (structure and tensor info only, no data).
    """
    if isinstance(obj, torch.Tensor):
        return {
            META_TYPE: META_TENSOR,
            "shape": tuple(obj.shape),
            "dtype": obj.dtype,
        }
    elif isinstance(obj, dict):
        return {
            META_TYPE: META_DICT,
            "items": {k: build_meta_tree(v) for k, v in obj.items()},
        }
    else:
        raise TypeError(f"Unsupported type in batch meta: {type(obj)}")


def allocate_from_meta(meta, device):
    """
    Build an empty batch from the meta tree (tensors allocated but not filled).
    """
    if meta[META_TYPE] == META_TENSOR:
        return torch.empty(
            meta["shape"],
            dtype=meta["dtype"],
            device=device,
        )
    elif meta[META_TYPE] == META_DICT:
        return {k: allocate_from_meta(v, device) for k, v in meta["items"].items()}
    else:
        raise ValueError(f"Unknown meta type: {meta[META_TYPE]}")


def broadcast_tensor_tree(obj, broadcast_tensor_fn):
    """
    Used by rank 0: recursively broadcast tensors.
    """
    if isinstance(obj, torch.Tensor):
        broadcast_tensor_fn(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            broadcast_tensor_tree(v, broadcast_tensor_fn)
    else:
        raise TypeError(f"Unsupported type in broadcast: {type(obj)}")


def recv_tensor_tree(obj, broadcast_tensor_fn):
    """
    Used by non-rank-0 ranks: recursively receive tensors.
    """
    if isinstance(obj, torch.Tensor):
        broadcast_tensor_fn(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            recv_tensor_tree(v, broadcast_tensor_fn)
    else:
        raise TypeError(f"Unsupported type in recv: {type(obj)}")

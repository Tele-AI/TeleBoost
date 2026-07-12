# Copyright (c) 2025 TeleAI-infra and Nvidia Megatron-LM Team. All rights reserved.
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.

import inspect
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from typing import Any, Optional

import megatron.core.parallel_state as ps
import torch
import torch.distributed as dist

_TENSOR_CONTEXT_PARALLEL_GROUP = None
_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_CONTEXT_PARALLEL_RANK = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None
# TP DP CP PP EP altogether, except dist-vae, etc.
_TRANSFORMER_MODEL_GROUP = None
_TRANSFORMER_THIS_MODEL_GROUP = None
# group that include all ranks
WORLD_GROUP = None
# groups that include the first tp-cp ranks and the vae rank
_DATA_TRANSMIT_GROUP = []


@dataclass
class CommPair:
    producer: int
    consumer: int or list
    dp_rank: int
    dp_size: int


_DATA_PRODUCER_CONSUMER_GROUP = None

_DISTRIBUTED_OP_ORIGINALS: dict[str, Any] = {}
_DISTRIBUTED_OP_WRAPPERS: dict[str, Any] = {}
_DISTRIBUTED_OP_MODELS_NUM: Optional[int] = None


def _stamp_distributed_wrapper(wrapper, original):
    """Mark an installed wrapper so module reloads cannot stack it again."""
    wrapper._teleboost_distributed_wrapper = True
    wrapper._teleboost_original = original
    return wrapper


def _translated_broadcast_src(src, group, *, model_group, get_world_size):
    """Map a global source rank onto the equivalent rank in ``group``.

    Multi-model TeleTron historically accepts a source rank from the first
    model replica and shifts it by the per-model stride.  Use the captured
    *unpatched* world-size function here: consulting ``dist.get_world_size``
    after installing our default-group wrapper would mistake the model group
    for the real world and make the translation depend on installation order.
    """
    if src is None:
        return None
    group_ranks = dist.get_process_group_ranks(group)
    if src in group_ranks:
        return src
    stride = get_world_size(group=model_group)
    world_size = get_world_size()
    candidate = src
    for _ in range((world_size + stride - 1) // stride + 1):
        candidate += stride
        if candidate in group_ranks:
            return candidate
    raise RuntimeError(f"broadcast: cannot align src into group_ranks={group_ranks} using stride={stride} (world={world_size})")


def apply_distributed_op_patches(models_num=1):
    """Route default collectives to TeleTron's transformer model group.

    Installation is process-global because legacy Megatron call sites omit the
    ``group=`` argument.  Keep that compatibility surface bounded: wrappers are
    installed once, retain PyTorch's public arguments/return values, and are
    restored by :func:`restore_distributed_op_patches` during model-parallel
    teardown.  Reapplying with the same topology is a no-op; changing topology
    without first destroying model parallelism is an error.
    """
    global _DISTRIBUTED_OP_MODELS_NUM

    if not isinstance(models_num, int) or isinstance(models_num, bool) or models_num < 1:
        raise ValueError(f"models_num must be a positive integer, got {models_num!r}")
    if _DISTRIBUTED_OP_WRAPPERS:
        if _DISTRIBUTED_OP_MODELS_NUM != models_num:
            raise RuntimeError(f"TeleTron distributed wrappers are already active for models_num={_DISTRIBUTED_OP_MODELS_NUM}; destroy model parallelism before switching to models_num={models_num}")
        return

    names = ["barrier", "all_reduce", "_all_gather_base", "get_world_size", "broadcast"]
    if models_num > 1:
        names.extend(["get_rank", "broadcast_object_list"])

    originals = {}
    for name in names:
        current = getattr(torch.distributed, name)
        # If this module was reloaded while its old wrappers remained installed,
        # recover the true underlying function instead of wrapping a wrapper.
        originals[name] = getattr(current, "_teleboost_original") if getattr(current, "_teleboost_distributed_wrapper", False) else current

    def default_group():
        return get_this_transformer_model_group() if models_num > 1 else get_transformer_model_group()

    @wraps(originals["barrier"])
    def barrier(group=None, async_op=False, device_ids=None):
        if group is None:
            group = default_group()
        return originals["barrier"](group=group, async_op=async_op, device_ids=device_ids)

    @wraps(originals["all_reduce"])
    def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        if group is None:
            group = default_group()
        return originals["all_reduce"](tensor, op=op, group=group, async_op=async_op)

    @wraps(originals["_all_gather_base"])
    def all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
        if group is None:
            group = default_group()
        return originals["_all_gather_base"](
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )

    @wraps(originals["get_world_size"])
    def get_world_size(group=None):
        if group is None:
            group = default_group()
        return originals["get_world_size"](group=group)

    broadcast_supports_group_src = "group_src" in inspect.signature(originals["broadcast"]).parameters

    @wraps(originals["broadcast"])
    def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
        if group is None:
            group = default_group()
        if models_num > 1 and group_src is None:
            src = _translated_broadcast_src(
                src,
                group,
                model_group=default_group(),
                get_world_size=originals["get_world_size"],
            )
        kwargs = {"src": src, "group": group, "async_op": async_op}
        if group_src is not None:
            if not broadcast_supports_group_src:
                raise TypeError("the installed PyTorch broadcast API does not support group_src")
            kwargs["group_src"] = group_src
        return originals["broadcast"](tensor, **kwargs)

    wrappers = {
        "barrier": barrier,
        "all_reduce": all_reduce,
        "_all_gather_base": all_gather_base,
        "get_world_size": get_world_size,
        "broadcast": broadcast,
    }

    if models_num > 1:

        @wraps(originals["get_rank"])
        def get_rank(group=None):
            if group is None:
                group = default_group()
            return originals["get_rank"](group=group)

        object_broadcast_supports_group_src = "group_src" in inspect.signature(originals["broadcast_object_list"]).parameters

        @wraps(originals["broadcast_object_list"])
        def broadcast_object_list(object_list, src=None, group=None, device=None, group_src=None):
            if group is None:
                group = default_group()
            if group_src is None:
                src = _translated_broadcast_src(
                    src,
                    group,
                    model_group=default_group(),
                    get_world_size=originals["get_world_size"],
                )
            kwargs = {"src": src, "group": group, "device": device}
            if group_src is not None:
                if not object_broadcast_supports_group_src:
                    raise TypeError("the installed PyTorch broadcast_object_list API does not support group_src")
                kwargs["group_src"] = group_src
            return originals["broadcast_object_list"](object_list, **kwargs)

        wrappers["get_rank"] = get_rank
        wrappers["broadcast_object_list"] = broadcast_object_list

    wrappers = {name: _stamp_distributed_wrapper(wrapper, originals[name]) for name, wrapper in wrappers.items()}
    installed = []
    try:
        for name, wrapper in wrappers.items():
            setattr(torch.distributed, name, wrapper)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            setattr(torch.distributed, name, originals[name])
        raise

    _DISTRIBUTED_OP_ORIGINALS.update(originals)
    _DISTRIBUTED_OP_WRAPPERS.update(wrappers)
    _DISTRIBUTED_OP_MODELS_NUM = models_num


def restore_distributed_op_patches() -> None:
    """Restore the exact collective functions captured at installation."""
    global _DISTRIBUTED_OP_MODELS_NUM
    for name, original in _DISTRIBUTED_OP_ORIGINALS.items():
        setattr(torch.distributed, name, original)
    _DISTRIBUTED_OP_ORIGINALS.clear()
    _DISTRIBUTED_OP_WRAPPERS.clear()
    _DISTRIBUTED_OP_MODELS_NUM = None


_MCORE_016_INITIALIZE_PARAMETERS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "pipeline_model_parallel_comm_backend",
    "use_sharp",
    "context_parallel_size",
    "hierarchical_context_parallel_sizes",
    "hybrid_context_parallel",
    "expert_model_parallel_size",
    "num_distributed_optimizer_instances",
    "expert_tensor_parallel_size",
    "nccl_communicator_config_path",
    "distributed_timeout_minutes",
    "order",
    "get_embedding_ranks",
    "get_position_embedding_ranks",
    "create_gloo_process_groups",
    "high_priority_stream_groups",
    "sharp_enabled_group",
    "create_all_gather_group",
)


def _validate_mcore_initialize_signature(initialize_model_parallel) -> None:
    """Fail before monkey-patching an unreviewed Megatron-Core initializer."""
    try:
        actual = tuple(inspect.signature(initialize_model_parallel).parameters)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cannot inspect megatron.core.parallel_state.initialize_model_parallel; TeleTron requires the Megatron-Core 0.16.1 API") from exc
    if actual != _MCORE_016_INITIALIZE_PARAMETERS:
        raise RuntimeError(f"Unsupported Megatron-Core initialize_model_parallel API. TeleTron targets the exact 0.16.1 parameter contract; expected={_MCORE_016_INITIALIZE_PARAMETERS}, actual={actual}")


def _validate_mcore_initialize_options(
    *,
    tensor_model_parallel_size,
    pipeline_model_parallel_comm_backend,
    hierarchical_context_parallel_sizes,
    hybrid_context_parallel,
    num_distributed_optimizer_instances,
    expert_tensor_parallel_size,
    order,
    get_embedding_ranks,
    get_position_embedding_ranks,
    create_gloo_process_groups,
    high_priority_stream_groups,
    sharp_enabled_group,
    create_all_gather_group,
) -> None:
    """Reject MCore 0.16 options the legacy TeleTron group builder drops."""
    unsupported = {}

    def reject(name, value, supported):
        if not supported:
            unsupported[name] = value

    reject(
        "pipeline_model_parallel_comm_backend",
        pipeline_model_parallel_comm_backend,
        pipeline_model_parallel_comm_backend is None,
    )
    reject(
        "hierarchical_context_parallel_sizes",
        hierarchical_context_parallel_sizes,
        hierarchical_context_parallel_sizes in (None, []),
    )
    reject("hybrid_context_parallel", hybrid_context_parallel, not hybrid_context_parallel)
    reject(
        "num_distributed_optimizer_instances",
        num_distributed_optimizer_instances,
        num_distributed_optimizer_instances == 1,
    )
    reject(
        "expert_tensor_parallel_size",
        expert_tensor_parallel_size,
        expert_tensor_parallel_size in (None, tensor_model_parallel_size),
    )
    reject("order", order, order == "tp-cp-ep-dp-pp")
    reject("get_embedding_ranks", get_embedding_ranks, get_embedding_ranks is None)
    reject(
        "get_position_embedding_ranks",
        get_position_embedding_ranks,
        get_position_embedding_ranks is None,
    )
    reject(
        "create_gloo_process_groups",
        create_gloo_process_groups,
        create_gloo_process_groups is True,
    )
    reject(
        "high_priority_stream_groups",
        high_priority_stream_groups,
        high_priority_stream_groups in (None, []),
    )
    reject("sharp_enabled_group", sharp_enabled_group, sharp_enabled_group is None)
    reject("create_all_gather_group", create_all_gather_group, not create_all_gather_group)
    if unsupported:
        rendered = ", ".join(f"{name}={value!r}" for name, value in unsupported.items())
        raise NotImplementedError(f"TeleTron's Megatron-Core 0.16.1 group extension does not implement these non-default options: {rendered}")


def initialize_model_parallel_decorators(initialize_model_parallel):
    if getattr(initialize_model_parallel, "_teleboost_initialize_wrapper", False):
        return initialize_model_parallel
    _validate_mcore_initialize_signature(initialize_model_parallel)

    @wraps(initialize_model_parallel)
    def wrapper(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_comm_backend: Optional[str] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[list[int]] = None,
        hybrid_context_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        create_gloo_process_groups: bool = True,
        high_priority_stream_groups: Optional[list[str]] = None,
        sharp_enabled_group: Optional[str] = None,
        create_all_gather_group: Optional[bool] = False,
        *,
        pipeline_model_parallel_split_rank: Optional[int] = None,
    ):
        global WORLD_GROUP
        global _TRANSFORMER_MODEL_GROUP
        global _TRANSFORMER_THIS_MODEL_GROUP
        global _DATA_TRANSMIT_GROUP

        _validate_mcore_initialize_options(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_comm_backend=pipeline_model_parallel_comm_backend,
            hierarchical_context_parallel_sizes=hierarchical_context_parallel_sizes,
            hybrid_context_parallel=hybrid_context_parallel,
            num_distributed_optimizer_instances=num_distributed_optimizer_instances,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            order=order,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            create_gloo_process_groups=create_gloo_process_groups,
            high_priority_stream_groups=high_priority_stream_groups,
            sharp_enabled_group=sharp_enabled_group,
            create_all_gather_group=create_all_gather_group,
        )
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before model parallelism")
        if _TRANSFORMER_MODEL_GROUP is not None or _DISTRIBUTED_OP_WRAPPERS:
            raise RuntimeError("TeleTron model parallelism is already initialized; call destroy_model_parallel() before initializing it again")

        from teleboost.engines.teletron import get_args

        margs = get_args()
        if margs.distributed_vae:
            extra_model_parallel_world_size = margs.distributed_vae_world_size
            total_world_size = torch.distributed.get_world_size()
            models_num = margs.consumer_models_num
            model_world_size = total_world_size - extra_model_parallel_world_size
        else:
            models_num = 1
            model_world_size = torch.distributed.get_world_size()

        if models_num < 1 or model_world_size < 1 or model_world_size % models_num != 0:
            raise ValueError(f"Invalid TeleTron model split: model_world_size={model_world_size}, consumer_models_num={models_num}")

        WORLD_GROUP = torch.distributed.new_group(range(0, torch.distributed.get_world_size()))
        ranks = range(0, model_world_size)
        base_process_group = torch.distributed.new_group(ranks)
        _TRANSFORMER_MODEL_GROUP = base_process_group

        per_model_world_size = model_world_size // models_num
        if models_num > 1:
            if torch.distributed.get_rank() < model_world_size:
                for k in range(models_num):
                    this_start_rank = k * per_model_world_size
                    ranks = range(this_start_rank, this_start_rank + per_model_world_size)
                    base_process_group = torch.distributed.new_group(ranks)
                    if torch.distributed.get_rank() in ranks:
                        _TRANSFORMER_THIS_MODEL_GROUP = base_process_group

        # build DATA_TRANSMIT_GROUP
        torch.distributed.get_rank()

        tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
        per_model_world_size // tensor_and_context_group_size

        if get_transformer_model_group() is not None:
            print("**********start init MP**********************************")
            initialize_model_parallel_base(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank,
                use_sharp,
                context_parallel_size,
                expert_model_parallel_size,
                nccl_communicator_config_path,
                distributed_timeout_minutes,
                _TRANSFORMER_THIS_MODEL_GROUP if models_num > 1 else _TRANSFORMER_MODEL_GROUP,
            )

            if margs.distributed_vae:
                initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
        else:
            print("vae data transmit group", _DATA_TRANSMIT_GROUP, flush=True)
            print("**********start init VAE**********************************")
            if margs.distributed_vae:
                initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)
            return None

        apply_distributed_op_patches(models_num)

    wrapper._teleboost_initialize_wrapper = True
    wrapper._teleboost_original = initialize_model_parallel
    return wrapper


def initialize_comm_pair(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size):
    from teleboost.engines.teletron import get_args

    args = get_args()
    models_num = args.consumer_models_num
    world_size = args.dit_world_size
    model_world_size = args.dit_world_size // models_num
    producer_size = args.distributed_vae_world_size

    # pp_start_rank = 0
    pp_size = model_world_size // pipeline_model_parallel_size

    global _DATA_PRODUCER_CONSUMER_GROUP

    local_rank = torch.distributed.get_rank()

    tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
    num_tensor_and_context_groups: int = pp_size // tensor_and_context_group_size
    print(
        "[CommPair Debug] rank={}, model_world_size={}, producer_size={}, tensor_model_parallel_size={}, context_parallel_size={}, pp_size={}, tensor_and_context_group_size={}, num_tensor_and_context_groups={}".format(
            local_rank,
            model_world_size,
            producer_size,
            tensor_model_parallel_size,
            context_parallel_size,
            pp_size,
            tensor_and_context_group_size,
            num_tensor_and_context_groups,
        )
    )
    assert num_tensor_and_context_groups % producer_size == 0 and num_tensor_and_context_groups // producer_size >= 1
    if get_transformer_model_group() is not None:
        # consumer ranks
        for i in range(num_tensor_and_context_groups):
            start_rank = local_rank // model_world_size * model_world_size
            start_rank = i * tensor_and_context_group_size + start_rank
            if start_rank == local_rank:
                _DATA_PRODUCER_CONSUMER_GROUP = CommPair(i % producer_size + world_size, local_rank, i, num_tensor_and_context_groups)
    else:
        for i in range(num_tensor_and_context_groups * models_num):
            if _DATA_PRODUCER_CONSUMER_GROUP is None:
                _DATA_PRODUCER_CONSUMER_GROUP = []
            start_rank = i * tensor_and_context_group_size
            if i % producer_size == local_rank - world_size:
                _DATA_PRODUCER_CONSUMER_GROUP.append(CommPair(local_rank, start_rank, i % num_tensor_and_context_groups, num_tensor_and_context_groups))


def get_comm_pair():
    return _DATA_PRODUCER_CONSUMER_GROUP


def get_world_group():
    return WORLD_GROUP


def initialize_model_parallel_base(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    base_process_group=None,
):
    assert torch.distributed.is_initialized()
    if base_process_group == -100:
        from teleboost.engines.teletron import get_args

        margs = get_args()
        extra_model_parallel_world_size = margs.distributed_vae_world_size
        total_world_size = torch.distributed.get_world_size()
        world_size = total_world_size - extra_model_parallel_world_size
    else:
        world_size = torch.distributed.get_world_size(base_process_group)

    print(f"base_process_group {base_process_group}, {world_size}\n" * 5, flush=True)

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size) != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by tensor_model_parallel_size ({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) x context_parallel_size ({context_parallel_size})")

    data_parallel_size: int = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size ")

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError("combination of expert model prallellism and context parallelism is not supported")

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    from teleboost.engines.teletron import get_args

    args = get_args()
    models_num = args.consumer_models_num

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than 2 with interleaved schedule")
        # global ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        # global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError("Cannot import `yaml`. Setting custom nccl communicator configs requires the yaml package.")

        with open(nccl_communicator_config_path) as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    all_data_parallel_group_ranks_with_cp = []

    for k in range(models_num):
        this_start_rank = k * world_size
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups + this_start_rank
            end_rank = (i + 1) * num_pipeline_model_parallel_groups + this_start_rank
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("dp", nccl_comm_cfgs))

                group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")

                if rank in ranks:
                    ps._DATA_PARALLEL_GROUP = group
                    ps._DATA_PARALLEL_GROUP_GLOO = group_gloo
                    ps._DATA_PARALLEL_GLOBAL_RANKS = ranks

            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                group_with_cp = torch.distributed.new_group(ranks_with_cp, timeout=timeout, pg_options=ps.get_nccl_options("dp_cp", nccl_comm_cfgs))
                group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, timeout=timeout, backend="gloo")

                if rank in ranks_with_cp:
                    ps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                    ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups

    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=ps.get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    for t in range(models_num):
        this_start_rank = t * world_size
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = i * num_pipeline_model_parallel_groups + j * tensor_model_parallel_size * context_parallel_size + this_start_rank
                end_rank = i * num_pipeline_model_parallel_groups + (j + 1) * tensor_model_parallel_size * context_parallel_size + this_start_rank
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("cp", nccl_comm_cfgs))
                    if rank in ranks:
                        ps._CONTEXT_PARALLEL_GROUP = group
                        ps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [data_parallel_group_ranks_with_cp[i] for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp]
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("mp", nccl_comm_cfgs))
        if rank in ranks:
            ps._MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.

    for i in range(num_tensor_model_parallel_groups):
        start_rank = torch.distributed.get_rank() // world_size * world_size
        ranks = range(i * tensor_model_parallel_size + start_rank, (i + 1) * tensor_model_parallel_size + start_rank)
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("tp", nccl_comm_cfgs))
        if rank in ranks:
            ps._TENSOR_MODEL_PARALLEL_GROUP = group

    for k in range(models_num):
        this_start_rank = k * world_size
        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        for i in range(num_pipeline_model_parallel_groups):
            # start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()
            ranks = range(i + this_start_rank, world_size + this_start_rank, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("pp", nccl_comm_cfgs))
            if rank in ranks:
                ps._PIPELINE_MODEL_PARALLEL_GROUP = group
                ps._PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks, timeout=timeout, pg_options=ps.get_nccl_options("embd", nccl_comm_cfgs))
            if rank in embedding_ranks:
                ps._EMBEDDING_GROUP = group
            if rank in ranks:
                ps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(
                position_embedding_ranks,
                timeout=timeout,
                pg_options=ps.get_nccl_options("embd", nccl_comm_cfgs),
            )
            if rank in position_embedding_ranks:
                ps._POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                ps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        # Build the tensor + data parallel groups.
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            # this_start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()
            start_rank = i * tensor_and_data_group_size_with_cp + this_start_rank
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("tp_dp_cp", nccl_comm_cfgs))
            if rank in ranks:
                ps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
            # this_start_rank = torch.distributed.get_rank() // torch.cuda.device_count() * torch.cuda.device_count()

            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_model_parallel_size + k * tensor_model_parallel_size * context_parallel_size + this_start_rank
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("tp_dp", nccl_comm_cfgs))
                if rank in ranks:
                    ps._TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + context parallel groups
    global _TENSOR_CONTEXT_PARALLEL_GROUP
    assert _TENSOR_CONTEXT_PARALLEL_GROUP is None, "Tensor + context parallel group is already initialized"
    assert ps._TENSOR_AND_CONTEXT_PARALLEL_GROUP is None, "Megatron tensor + context parallel group is already initialized"
    tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
    num_tensor_and_context_groups: int = world_size // tensor_and_context_group_size
    print(f"world_size: {world_size}, {tensor_and_context_group_size}")
    for k in range(models_num):
        this_start_rank = k * world_size
        for i in range(num_tensor_and_context_groups):
            start_rank = i * tensor_and_context_group_size + this_start_rank
            end_rank = start_rank + tensor_and_context_group_size
            ranks = range(start_rank, end_rank)

            group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("tp_cp", nccl_comm_cfgs))
            if rank in ranks:
                _TENSOR_CONTEXT_PARALLEL_GROUP = group
                # Megatron-Core 0.16 names this the tensor-and-context
                # parallel group.  Keep the TeleTron legacy state and the
                # native state on the exact same ProcessGroup so production
                # code can use the supported 0.16 accessors.
                ps._TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

        # Build the tensor + expert parallel groups
        tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size // expert_model_parallel_size
        for i in range(num_tensor_and_data_groups):
            for j in range(num_expert_groups):
                # TPxEP Group
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size + this_start_rank
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size + this_start_rank
                ranks = range(start_rank, end_rank)
                group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("tp_exp", nccl_comm_cfgs))
                if rank in ranks:
                    ps._TENSOR_AND_EXPERT_PARALLEL_GROUP = group
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    group = torch.distributed.new_group(ranks, pg_options=ps.get_nccl_options("exp", nccl_comm_cfgs))
                    if rank in ranks:
                        ps._EXPERT_MODEL_PARALLEL_GROUP = group

        for i in range(num_tensor_and_data_groups):
            start_rank = i * tensor_and_data_group_size + this_start_rank
            end_rank = (i + 1) * tensor_and_data_group_size + this_start_rank
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=ps.get_nccl_options("dp_modulo_exp", nccl_comm_cfgs))
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    ps._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

    # megatron-core 0.16+ split the EP-related groups out from the legacy
    # data-modulo-expert / tensor-and-expert names that teleboost's wrapper
    # builds. The new accessors (get_expert_tensor_parallel_rank,
    # get_expert_data_parallel_group) read from globals teleboost never sets
    # and crash with None / AssertionError. EP=1 is the only path here
    # (EP>1 + CP>1 is asserted unsupported at line 361), so alias the new
    # globals to the existing equivalent groups:
    #   * EP=1 expert TP group ≡ regular TP group
    #   * EP=1 expert DP group ≡ DP-modulo-expert group (already built above)
    ps._EXPERT_TENSOR_PARALLEL_GROUP = ps._TENSOR_MODEL_PARALLEL_GROUP
    ps._EXPERT_DATA_PARALLEL_GROUP = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP
    ps._EXPERT_DATA_PARALLEL_GROUP_GLOO = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    if ps._GLOBAL_MEMORY_BUFFER is None:
        ps._set_global_memory_buffer()


def get_transformer_model_group(check_initialized=True):
    """Get the transformer_model group the caller rank belongs to."""
    if check_initialized:
        assert _TRANSFORMER_MODEL_GROUP is not None, "tensor context parallel group is not initialized"
    # print("get transformer blocks: ", dist.get_rank(_TRANSFORMER_MODEL_GROUP))
    if dist.get_rank(_TRANSFORMER_MODEL_GROUP) == -1:
        return None

    return _TRANSFORMER_MODEL_GROUP


def get_this_transformer_model_group(check_initialized=True):
    """Get the transformer_model group the caller rank belongs to."""
    if check_initialized:
        assert _TRANSFORMER_THIS_MODEL_GROUP is not None, "tensor context parallel group is not initialized"
    # print("get transformer blocks: ", dist.get_rank(_TRANSFORMER_MODEL_GROUP))
    if dist.get_rank(_TRANSFORMER_THIS_MODEL_GROUP) == -1:
        return None

    return _TRANSFORMER_THIS_MODEL_GROUP


def get_tensor_context_parallel_group(check_initialized=True):
    """Backward-compatible alias for Megatron-Core's TP-and-CP group."""
    native_group = ps.get_tensor_and_context_parallel_group(check_initialized=False)
    if native_group is not None:
        return native_group
    if check_initialized:
        assert _TENSOR_CONTEXT_PARALLEL_GROUP is not None, "tensor context parallel group is not initialized"
    return _TENSOR_CONTEXT_PARALLEL_GROUP


def get_tensor_context_parallel_world_size():
    """Backward-compatible TP-and-CP world-size accessor."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    return ps.get_tensor_and_context_parallel_world_size()


def get_tensor_context_parallel_rank():
    """Backward-compatible TP-and-CP local-rank accessor."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    if _MPU_TENSOR_CONTEXT_PARALLEL_RANK is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    return ps.get_tensor_and_context_parallel_rank()


def get_tensor_and_context_parallel_src_rank():
    """Return the global rank for local rank zero of the TP-and-CP group.

    Megatron-Core 0.16.1 exposes source-rank helpers for TP and DP, and a
    global-rank list for the CP-only group, but no source accessor for the
    combined TP-and-CP group.  DPO broadcasts span the combined group, so a
    CP-only source differs across TP columns when TP > 1 and is invalid.  Map
    local rank zero through the actual combined ProcessGroup instead.
    """
    group = ps.get_tensor_and_context_parallel_group()
    return dist.get_global_rank(group, 0)


def get_tensor_context_parallel_src_rank():
    """Backward-compatible alias for the TP-and-CP source rank."""
    return get_tensor_and_context_parallel_src_rank()


def _reset_teletron_parallel_state() -> None:
    """Clear every TeleTron-owned reference so a fresh init is well-defined."""
    global _TENSOR_CONTEXT_PARALLEL_GROUP
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    global _TRANSFORMER_MODEL_GROUP
    global _TRANSFORMER_THIS_MODEL_GROUP
    global WORLD_GROUP
    global _DATA_TRANSMIT_GROUP
    global _DATA_PRODUCER_CONSUMER_GROUP

    _TENSOR_CONTEXT_PARALLEL_GROUP = None
    _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
    _MPU_TENSOR_CONTEXT_PARALLEL_RANK = None
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None
    _TRANSFORMER_MODEL_GROUP = None
    _TRANSFORMER_THIS_MODEL_GROUP = None
    WORLD_GROUP = None
    _DATA_TRANSMIT_GROUP = []
    _DATA_PRODUCER_CONSUMER_GROUP = None


def destroy_model_parallel_wrapper(destroy_model_parallel):
    if getattr(destroy_model_parallel, "_teleboost_destroy_wrapper", False):
        return destroy_model_parallel
    if not callable(destroy_model_parallel):
        raise TypeError(f"megatron.core.parallel_state.destroy_model_parallel must be callable, got {destroy_model_parallel!r}")

    @wraps(destroy_model_parallel)
    def wrapper(*args, **kwargs):
        # Restore PyTorch first so Megatron's own teardown observes the native
        # WORLD default instead of TeleTron's transformer-group default.
        restore_distributed_op_patches()
        try:
            return destroy_model_parallel(*args, **kwargs)
        finally:
            _reset_teletron_parallel_state()

    wrapper._teleboost_destroy_wrapper = True
    wrapper._teleboost_original = destroy_model_parallel
    return wrapper

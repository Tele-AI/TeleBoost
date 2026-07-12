# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025 TeleAI-infra Team.
#
# Original NVIDIA-authored portions are licensed under BSD-3-Clause; see
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/LICENSE.
# TeleAI modifications are licensed under Apache-2.0; see LICENSE at the root.

import math
import time
from abc import ABC

import torch
import torch.distributed as dist


def count_parameters(dim: int, in_dim: int, ffn_dim: int, out_dim: int, text_dim: int, freq_dim: int, patch_size: tuple, num_heads: int, num_layers: int, has_image_input: bool, has_image_pos_emb: bool) -> int:
    total_params = 0

    # Patch embedding (Conv3d)
    patch_volume = math.prod(patch_size)
    patch_emb = in_dim * dim * patch_volume + dim
    total_params += patch_emb

    # Text embedding
    text_emb = text_dim * dim + dim + dim * dim + dim
    total_params += text_emb

    # Time embedding
    time_emb = freq_dim * dim + dim + dim * dim + dim
    total_params += time_emb

    # Time projection
    time_proj = dim * (6 * dim) + (6 * dim)
    total_params += time_proj

    # Each DiTBlock
    block_params = 0
    # SelfAttention
    block_params += 4 * (dim * dim + dim)
    block_params += 2 * dim  # RMSNorm
    # CrossAttention
    block_params += 4 * (dim * dim + dim)
    block_params += 2 * dim
    if has_image_input:
        block_params += 2 * (dim * dim + dim)
        block_params += dim
    # FFN
    block_params += dim * ffn_dim + ffn_dim + ffn_dim * dim + dim
    block_params += 2 * dim  # LayerNorm
    block_params += 6 * dim  # modulation
    total_params += block_params * num_layers

    # Head
    head = dim * (out_dim * patch_volume) + (out_dim * patch_volume)
    head += 2 * dim
    total_params += head

    # Image embedding (optional)
    if has_image_input:
        img_emb = 1280 * dim + dim + dim * dim + dim
        total_params += img_emb
        if has_image_pos_emb:
            total_params += 514 * 1280  # positional embedding

    return total_params / 10**9


def memory_check(search_task, object_shape, args, dit_model_config):
    cp_size = 2**search_task
    a = -0.00000002272673122
    b = 0.00113517676146691
    c = 3.58017744827079687
    sequence_length = (object_shape[1] // 4 + 1) * object_shape[3] / 16 * object_shape[4] / 16 / cp_size
    activate_mem = a * sequence_length * sequence_length + b * sequence_length + c

    # compute params number of model
    num_total_parameters = count_parameters(
        dit_model_config.dim,
        dit_model_config.in_dim,
        dit_model_config.ffn_dim,
        dit_model_config.out_dim,
        dit_model_config.text_dim,
        dit_model_config.freq_dim,
        dit_model_config.patch_size,
        dit_model_config.num_heads,
        dit_model_config.num_layers,
        dit_model_config.has_image_input,
        dit_model_config.has_image_pos_emb,
    )
    if args.bf16 or args.fp16:
        weights_mem = num_total_parameters * 2
    else:
        weights_mem = num_total_parameters * 4
    optimizer_mem = weights_mem * 3 / args.dit_world_size
    grad_mem = weights_mem / args.dit_world_size
    flatten_param_mem = grad_mem * 3
    predict_mem = max(activate_mem, grad_mem + flatten_param_mem) + weights_mem + optimizer_mem
    print(f"rank {torch.distributed.get_rank()} num_layers is {dit_model_config.num_layers} predict_mem is {predict_mem} num_total_parameters is {num_total_parameters:.2f}")
    if predict_mem < 75:
        return True
    return False


class Costtime:
    def __init__(self, time, sync_flag=False):
        self.time = time
        self.sync_flag = sync_flag


class TimeRecord:
    def __init__(self, data_type, cp_size, record_shape):
        self.data_type = data_type
        self.cp_size = cp_size
        record_shape = tuple(record_shape)
        self.record_shape = min(record_shape[3], record_shape[4])

    def __eq__(self, other):
        if not isinstance(other, TimeRecord):
            return False
        return self.data_type == other.data_type and self.cp_size == other.cp_size and self.record_shape == other.record_shape

    def __hash__(self):
        return hash((self.data_type, self.cp_size, self.record_shape))


class Timer(ABC):
    """
    Timer class with ability to start/stop.

    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    def __init__(self, name, barrier_group=None):
        """Initialize Timer.

        Args:
            name (str): Name of the timer.
        """
        self.name = name
        self._elapsed = 0.0
        self._active_time = 0.0
        self._started = False
        # Note that None will default to the global process group
        self._barrier_group = barrier_group
        self._start_time = time.time()

    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        assert not self._started, "timer has already been started"
        if self._barrier_group is not None:
            torch.distributed.barrier(group=self._barrier_group)
        torch.cuda.synchronize()
        self._start_time = time.time()
        self._started = True

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, "timer is not started"
        torch.cuda.synchronize()
        stop_time = time.time()
        elapsed = stop_time - self._start_time
        self._elapsed += elapsed
        self._active_time += elapsed
        self._started = False

    def reset(self):
        """Reset timer."""
        # Don't reset _active_time
        self._elapsed = 0.0
        self._started = False

    def elapsed(self, reset=True):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started
        if self._started:
            self.stop()
        _elapsed = self._elapsed
        if reset:
            self.reset()
        if _started:
            self.start()
        return _elapsed

    def elapsed_average(self, reset=True):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        _started = self._started
        if self._started:
            self.stop()
        _elapsed = self._elapsed
        if self._barrier_group is not None:
            time_tensor = torch.tensor(_elapsed, device=torch.cuda.current_device())
            dist.all_reduce(time_tensor, group=self._barrier_group, op=dist.ReduceOp.SUM)
            avg_time = time_tensor.item() / dist.get_world_size(group=self._barrier_group)
        else:
            avg_time = _elapsed
        if reset:
            self.reset()
        if _started:
            self.start()
        return avg_time


class Timers:
    """Class for a group of Timers."""

    def __init__(self, args, dit_model_config):
        """Initialize group of timers."""
        self.args = args
        self.dit_model_config = dit_model_config
        self._timers = {}
        self.dit_times_record = {}
        self.vae_times_record = {}

    # Get the average elapsed time across ranks via all-reduce and store it in the dict
    def get_elapsed_time(self, name):
        average_elapsed_time = self._timers[name].elapsed_average()
        return average_elapsed_time

    def print_record_time(self, iteration=None):
        record_str = ""
        for record_key, costtime in self.vae_times_record.items():
            record_str += f"{record_key.data_type}-{record_key.cp_size}-{record_key.record_shape}:{costtime.time} {costtime.sync_flag}\n"
        for record_key, costtime in self.dit_times_record.items():
            record_str += f"{record_key.data_type}-{record_key.cp_size}-{record_key.record_shape}:{costtime.time} {costtime.sync_flag}\n"
        record_str = f"rank {torch.distributed.get_rank()} iteration {iteration} record_message is:\n{record_str}"
        print(record_str)
        return record_str

    # Sync collected perf data between producer and consumer via send/recv_object_list, then broadcast to the remaining ranks on each side

    def get_record_key(self, name, task, object_shape):
        cp_size = 2**task
        return TimeRecord(name, cp_size, object_shape)

    # Based on the current data shape, determine the next CP size, num_micro_batchs, and whether online profiling is enabled

    # Search for the optimal CP size based on online-collected perf data
    def search_best_task(self, pre_data_shape, pre_task, object_shape, max_cp_size):
        pre_dit_record_key = self.get_record_key("dit", pre_task, pre_data_shape)
        # If no record exists for the previous shape, fall back to the largest CP size and wait for the previous shape's info to sync
        if pre_dit_record_key not in self.dit_times_record or not self.dit_times_record[pre_dit_record_key].sync_flag:
            print(f"rank{torch.distributed.get_rank()} cannot find pre_shape {pre_data_shape} task {pre_task} record time in {self.print_record_time()} for object_shape {object_shape}")
            search_task = int(math.log2(max_cp_size))
            while search_task >= 0:
                vae_record_key = self.get_record_key("vae", search_task, object_shape)
                dit_record_key = self.get_record_key("dit", search_task, object_shape)
                if dit_record_key in self.dit_times_record and vae_record_key in self.vae_times_record:
                    break
                search_task -= 1
            return search_task

        pre_dit_time = self.dit_times_record[pre_dit_record_key].time
        search_task = int(math.log2(max_cp_size))

        max_throughputs = 0
        best_task = 0
        frames = object_shape[1]

        global_batch_size = self.args.global_batch_size
        dit_world_size = self.args.dit_world_size
        encoder_world_size = self.args.distributed_vae_world_size
        total_world_size = dit_world_size + encoder_world_size

        while search_task >= 0:
            vae_record_key = self.get_record_key("vae", search_task, object_shape)
            dit_record_key = self.get_record_key("dit", search_task, object_shape)
            if dit_record_key in self.dit_times_record and vae_record_key in self.vae_times_record:
                vae_time = self.vae_times_record[vae_record_key].time
                dit_time = self.dit_times_record[dit_record_key].time
                cp_size = 2**search_task
                per_vae_micro_batch_size = dit_world_size // cp_size // encoder_world_size
                num_micro_batchs = global_batch_size // (dit_world_size // cp_size)
                # Overlap between pre_dit and the current vae
                first_time = max(0, per_vae_micro_batch_size * vae_time - pre_dit_time)
                # Execution time of the current dit
                second_time = dit_time + (num_micro_batchs - 1) * max(dit_time, vae_time * per_vae_micro_batch_size)
                tmp_step_time = first_time + second_time
                print(f"rank {torch.distributed.get_rank()} cp_size {cp_size} tmp_step_time is {tmp_step_time}")
                tmp_throughputs = frames * global_batch_size / tmp_step_time / (total_world_size / 8)
                if tmp_throughputs > max_throughputs:
                    max_throughputs = tmp_throughputs
                    best_task = search_task

            search_task -= 1

        return best_task

    # Timer control functions
    def get_timer(self, name, barrier_group=None):
        if name not in self._timers:
            self._timers[name] = Timer(name, barrier_group)

    def start_timer(self, name):
        assert name in self._timers, "timer {} is not initialized.".format(name)
        self._timers[name].start()

    def stop_timer(self, name):
        self._timers[name].stop()

    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """Returns elapsed times of timers in names.
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.

        Args:
            names (List[str]): list of timer names
            reset (bool): reset the timer after recording the elapsed time
            barrier (bool): if set, do a global barrier before time measurments

        Returns:
            torch.tensor: Tensor of size [world_size, len(names)] with times in float.
        """

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros((world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device())
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

        # See the note above for why we are not using gather.
        torch.distributed.all_gather_into_tensor(
            rank_name_to_time.view(-1),
            rank_name_to_time[rank, :].view(-1),
        )

        return rank_name_to_time

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            rank_to_time = rank_name_to_time[:, i]
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]
            # If the timer exists:
            if rank_to_time.numel() > 0:
                name_to_min_max_time[name] = (
                    rank_to_time.min().item() / normalizer,
                    rank_to_time.max().item() / normalizer,
                )
        return name_to_min_max_time

    def log(
        self,
        names: list[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """logs the timers passed in names to stdout. Example usage is to log average per step value for timer 'foo',
          this function can be called with normalizer factor set to logging interval.

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the last rank. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)
        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1
        if rank == torch.distributed.get_rank() and output_string is not None:
            print(output_string, flush=True)

    def write(
        self,
        names: list[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write timers to a tensorboard writer. Note that we only report maximum time across ranks to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor. Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments. Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + "-time", max_time, iteration)

# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from teleboost.algorithms.wan_transition import (
    align_wan_log_probs_for_loss,
    compute_flow_grpo_window,
    compute_wan_pixel_weight_maps_with_fallback,
    finalize_wan_transition_fields,
    make_wan_solver_metadata,
    reduce_wan_log_density,
    validate_wan_solver_metadata,
)
from teleboost.algorithms.solver_contract import SolverContract


def test_wan_baseline_log_density_distinguishes_unbatched_and_batched():
    unbatched = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    batched = torch.stack((unbatched, unbatched + 10), dim=0)

    one = reduce_wan_log_density(unbatched, pixel_enabled=False)
    many = reduce_wan_log_density(batched, pixel_enabled=False)

    assert one.ndim == 0
    assert torch.equal(one, unbatched.mean())
    assert many.shape == (2,)
    assert torch.equal(many, batched.mean(dim=(1, 2, 3, 4)))


def test_wan_pixel_log_density_sums_only_channel_axis():
    unbatched = torch.randn(3, 2, 4, 5)
    batched = torch.randn(2, 3, 2, 4, 5)

    assert torch.equal(
        reduce_wan_log_density(unbatched, pixel_enabled=True),
        unbatched.sum(dim=0),
    )
    assert torch.equal(
        reduce_wan_log_density(batched, pixel_enabled=True),
        batched.sum(dim=1),
    )


def test_wan_scalar_loss_boundary_rejects_channel_vector():
    with pytest.raises(AssertionError, match="new_log_probs.*must be.*B"):
        align_wan_log_probs_for_loss(
            torch.randn(16),
            torch.randn(16),
            torch.randn(1),
            batch_size=1,
            pixel_enabled=False,
        )

    new, old = align_wan_log_probs_for_loss(
        torch.tensor(0.2),
        torch.tensor([0.1]),
        torch.tensor([1.0]),
        batch_size=1,
        pixel_enabled=False,
    )
    assert new.shape == old.shape == (1,)


def test_wan_dense_loss_boundary_adds_single_sample_batch_axis():
    new, old = align_wan_log_probs_for_loss(
        torch.randn(2, 3, 4),
        torch.randn(1, 2, 3, 4),
        torch.randn(1, 2, 3, 4),
        batch_size=1,
        pixel_enabled=True,
    )
    assert new.shape == old.shape == (1, 2, 3, 4)


def test_flow_window_size_one_never_selects_terminal_only_transition():
    # Ask the sampler for its latest possible start. It must stop at index 6;
    # index 7 is the global sigma->0 transition for num_steps=8.
    window = compute_flow_grpo_window(
        1,
        (0, 8),
        8,
        randint=lambda _low, high: high,
    )
    assert window == (6, 7)

    with pytest.raises(ValueError, match="only the final sigma->0"):
        compute_flow_grpo_window(1, (7, 8), 8)


def _transition_fields(batch_size=2, steps=2):
    return {
        "latents": torch.randn(batch_size, steps, 3, 2, 4, 4),
        "next_latents": torch.randn(batch_size, steps, 3, 2, 4, 4),
        "log_probs": torch.randn(batch_size, steps),
        "timesteps": torch.arange(steps).repeat(batch_size, 1),
        "prev_sample_mean": torch.randn(batch_size, steps, 3, 2, 4, 4),
        "timestep_indices": torch.arange(steps).repeat(batch_size, 1),
    }


def test_nonterminal_window_keeps_every_transition_field():
    fields = _transition_fields(steps=1)
    finalized, indices = finalize_wan_transition_fields(
        fields,
        transition_indices=(3,),
        num_steps=8,
    )
    assert indices == (3,)
    assert {tensor.shape[1] for tensor in finalized.values()} == {1}


def test_terminal_transition_is_trimmed_from_every_field():
    fields = _transition_fields(steps=2)
    finalized, indices = finalize_wan_transition_fields(
        fields,
        transition_indices=(6, 7),
        num_steps=8,
    )
    assert indices == (6,)
    assert set(finalized) == set(fields)
    assert {tensor.shape[1] for tensor in finalized.values()} == {1}
    for name in fields:
        assert torch.equal(finalized[name], fields[name][:, :1])


def test_transition_field_length_mismatch_fails_before_dataproto():
    fields = _transition_fields(steps=2)
    fields["log_probs"] = torch.randn(2, 1)
    with pytest.raises(ValueError, match="log_probs.*S=1.*expected 2"):
        finalize_wan_transition_fields(
            fields,
            transition_indices=(2, 3),
            num_steps=8,
        )


def test_vipo_batch_failure_returns_uniform_map_with_latent_geometry():
    latents = torch.randn(2, 1, 3, 4, 5, 6)
    videos = torch.randn(2, 3, 8, 20, 24)

    def fail(**_kwargs):
        raise RuntimeError("DINO failed")

    maps, error = compute_wan_pixel_weight_maps_with_fallback(
        fail,
        videos=videos,
        latents=latents,
    )
    assert isinstance(error, RuntimeError)
    assert maps.shape == (2, 4, 5, 6)
    assert maps.dtype == torch.float32
    assert torch.equal(maps, torch.ones_like(maps))


@pytest.mark.parametrize(
    ("pixel_enabled", "reduction"),
    [(False, "mean"), (True, "channel_sum_dense")],
)
def test_wan_rollout_solver_metadata_is_checked_at_actor_boundary(
    pixel_enabled,
    reduction,
):
    metadata = make_wan_solver_metadata(
        batch_size=2,
        sigma_form="flow_grpo",
        pixel_enabled=pixel_enabled,
    )
    contract = SolverContract(
        name="wan_flow_grpo",
        sigma_form="flow_grpo",
        rollout_transition="wan_sde",
        recompute_transition="wan_sde",
        eval_transition="wan_ode",
        logprob_reduction=reduction,
    )
    validate_wan_solver_metadata(
        metadata,
        batch_size=2,
        expected_contract=contract,
    )

    metadata["sigma_form"][1] = "dancegrpo"
    with pytest.raises(ValueError, match="inconsistent within the batch"):
        validate_wan_solver_metadata(
            metadata,
            batch_size=2,
            expected_contract=contract,
        )

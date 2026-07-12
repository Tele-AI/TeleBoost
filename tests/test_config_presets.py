from teleboost.config.preset_loader import load_overlay_preset, load_program_preset


def test_program_preset_loads_by_public_program_name():
    preset = load_program_preset("wan.grpo.fsdp", "flow_grpo")
    assert preset["program"] == "wan.grpo.fsdp"
    assert preset["overrides"]["actor_rollout_ref.actor.sigma_form"] == "flow_grpo"


def test_overlay_preset_is_composable_and_program_neutral():
    preset = load_overlay_preset("grpo_guard")
    assert "program" not in preset
    assert preset["overrides"]["actor_rollout_ref.actor.grpo_guard.enable"] is True

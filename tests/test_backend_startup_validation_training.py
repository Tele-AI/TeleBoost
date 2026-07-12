# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
"""Training-stack startup validation tests."""

from teleboost.programs.main import _validate_config
from teleboost.programs.wan.backend import WanBackendSpec
from tests.test_backend_startup_validation import _config


def test_registry_reward_is_validated_even_when_upstream_manager_is_disabled(
    monkeypatch,
):
    cfg = _config(
        "diffusion",
        reward_type="joint",
        adapter="",
        enable=False,
    )
    backend = WanBackendSpec()
    called = []
    monkeypatch.setattr(
        backend,
        "validate_reward",
        lambda config: called.append(config),
    )

    _validate_config(cfg, backend)

    assert called == [cfg]

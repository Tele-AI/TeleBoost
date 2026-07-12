"""Spec for the shared config-access helpers.

These were four byte-identical copies of ``_select`` plus ad-hoc ``_as_*``
wrappers across fsdp_worker/dp_actor/rollout/fsdp_model; this
pins the single canonical behavior so the consolidation can't silently drift.
"""

from teleboost.config.access import as_bool, as_float, as_int, select


def test_select_dict_dotted_path():
    cfg = {"a": {"b": {"c": 7}}}
    assert select(cfg, "a.b.c") == 7


def test_select_missing_returns_default():
    assert select({"a": {}}, "a.b.c", default="dflt") == "dflt"
    assert select({}, "x", default=None) is None


def test_select_attribute_object():
    class N:
        pass

    root = N()
    root.inner = N()
    root.inner.val = 42
    assert select(root, "inner.val") == 42
    assert select(root, "inner.missing", default=-1) == -1


def test_select_none_node_short_circuits():
    assert select(None, "a.b", default="d") == "d"


def test_as_bool_string_truthiness():
    for s in ("1", "true", "TRUE", "Yes", "on", " on "):
        assert as_bool({"k": s}, "k", False) is True
    for s in ("0", "false", "no", "off", "", "random"):
        assert as_bool({"k": s}, "k", True) is False


def test_as_bool_non_string():
    assert as_bool({"k": 1}, "k", False) is True
    assert as_bool({"k": 0}, "k", True) is False
    assert as_bool({}, "k", True) is True  # default path


def test_as_int_and_as_float_coerce():
    assert as_int({"k": "5"}, "k", 0) == 5
    assert as_float({"k": "1.5"}, "k", 0.0) == 1.5
    assert as_int({}, "k", 9) == 9
    assert as_float({}, "k", 2.0) == 2.0


def test_reward_adapter_uses_canonical_select_without_legacy_facade():
    from teleboost.reward import routing

    assert routing._select is select
    assert not hasattr(routing, "select")
    assert not hasattr(routing, "select_config")

    config = {
        "reward": {
            "reward_model": {
                "adapter": " VIDEO_VLM ",
                "enable": True,
            }
        },
    }
    assert routing.is_video_vlm_reward_config(config)

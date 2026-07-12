"""Failed judge rewards must fail loud, not train."""

import pytest

from teleboost.reward.adapters.common import require_judge_success


def _ok(score: float) -> dict:
    return {"score": score, "raw": "综合得分:90分"}


def _failed(reason: str = "<error: reward server 500>") -> dict:
    return {"score": 0.0, "raw": reason, "failed": True}


def test_all_failed_batch_raises():
    with pytest.raises(RuntimeError, match="4/4 samples"):
        require_judge_success([_failed() for _ in range(4)], "video VLM")


def test_partial_failure_raises():
    with pytest.raises(RuntimeError, match="1/3 samples"):
        require_judge_success([_ok(0.9), _failed(), _ok(0.85)], "video VLM")


def test_genuine_zero_scores_are_not_failures():
    require_judge_success([_ok(0.0), _ok(0.0)], "video VLM")


def test_failed_judge_returns_carry_failed_flag():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel_path in ("teleboost/reward/adapters/video_vlm_score.py",):
        src = (root / rel_path).read_text()
        failed_returns = [line for line in src.splitlines() if 'return {"score": 0.0, "raw":' in line]
        assert failed_returns, f"{rel_path}: failed return sites moved; update this test"
        for line in failed_returns:
            assert '"failed": True' in line, f"{rel_path}: failed result without flag: {line.strip()}"


@pytest.mark.parametrize(
    "module_name",
    ["teleboost.reward.adapters.video_vlm_score"],
)
def test_unparseable_judge_output_is_an_explicit_failure(module_name):
    from importlib import import_module

    parsed = import_module(module_name).parse_eval_score("I cannot score this media")
    assert parsed["failed"] is True
    assert parsed["score"] == 0.0


@pytest.mark.parametrize(
    "module_name",
    ["teleboost.reward.adapters.video_vlm_score"],
)
def test_out_of_range_judge_output_is_an_explicit_failure(module_name):
    from importlib import import_module

    parsed = import_module(module_name).parse_eval_score("合计:101分")
    assert parsed["failed"] is True
    assert parsed["score"] == 0.0


@pytest.mark.parametrize(
    "module_name",
    ["teleboost.reward.adapters.video_vlm_score"],
)
def test_structured_unitless_judge_output_is_supported(module_name):
    from importlib import import_module

    parsed = import_module(module_name).parse_eval_score("dim1:60,dim2:80,dim3:70,dim4:70,dim5:80,合计:72")
    assert parsed.get("failed") is not True
    assert parsed["score_raw"] == 72.0
    assert parsed["score"] == 0.72
    assert len([name for name in parsed if name.startswith("dim")]) == 5


@pytest.mark.parametrize(
    "module_name",
    [
        "teleboost.reward.adapters.video_vlm_score",
    ],
)
def test_inconsistent_total_reconciles_to_dimension_mean(module_name):
    """The prompt defines 合计 as the dimension mean: when the judge mis-adds,
    the parsed dimensions win and the sample must NOT fail."""
    from importlib import import_module

    module = import_module(module_name)
    parsed = module.parse_eval_score("dim1:30分,dim2:35分,dim3:30分,dim4:25分,dim5:20分,合计:30分")
    assert not parsed.get("failed")
    assert parsed["total_reconciled"] is True
    assert parsed["stated_total"] == 30.0
    assert parsed["score"] == pytest.approx(0.28)
    assert parsed["score_raw"] == pytest.approx(28.0)

"""Shared parsing primitives for structured VLM judge scores."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_NUMBER = r"(\d+(?:\.\d+)?)"
# The unit is optional only for a complete structured field. This accepts the
# real-model form ``合计:72`` without turning arbitrary prose numbers into a
# reward.
_FIELD_END = r"(?:\s*分)?(?=\s*(?:[,，;；。\n]|$))"


def labelled_score_pattern(label: str) -> str:
    """Build one safe score pattern for both unitful and unitless fields."""

    return rf"(?<!\w){re.escape(label)}[：:]\s*{_NUMBER}{_FIELD_END}"


def dimension_patterns(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple((labelled_score_pattern(f"dim{index}"), name) for index, name in enumerate(names, start=1))


def parse_structured_score(
    output_text: str,
    *,
    total_patterns: Sequence[str],
    dimensions: Sequence[tuple[str, str]],
    max_total_mean_delta: float = 1.0,
) -> dict[str, Any]:
    total: float | None = None
    for pattern in total_patterns:
        match = re.search(pattern, output_text, flags=re.IGNORECASE)
        if match:
            try:
                total = float(match.group(1))
            except (IndexError, ValueError):
                continue
            break

    if total is None:
        return {
            "score_raw": 0.0,
            "score": 0.0,
            "raw": output_text,
            "failed": True,
            "error": "judge output did not contain a parseable total score",
        }

    if not 0.0 <= total <= 100.0:
        return {
            "score_raw": total,
            "score": 0.0,
            "raw": output_text,
            "failed": True,
            "error": f"judge total score is outside [0, 100]: {total}",
        }

    parsed_dimensions: dict[str, float] = {}
    missing_dimensions: list[str] = []
    for pattern, name in dimensions:
        match = re.search(pattern, output_text, flags=re.IGNORECASE)
        if match is None:
            missing_dimensions.append(name)
            continue
        try:
            value = float(match.group(1))
        except (IndexError, ValueError):
            missing_dimensions.append(name)
            continue
        if not 0.0 <= value <= 100.0:
            return {
                "score_raw": total,
                "score": 0.0,
                "raw": output_text,
                "failed": True,
                "error": f"judge dimension {name} is outside [0, 100]: {value}",
            }
        parsed_dimensions[name] = value

    if missing_dimensions:
        return {
            "score_raw": total,
            "score": 0.0,
            "raw": output_text,
            "failed": True,
            "error": "judge output is missing dimensions: " + ", ".join(missing_dimensions),
        }

    dimension_mean = sum(parsed_dimensions.values()) / len(parsed_dimensions)
    if abs(total - dimension_mean) > max_total_mean_delta:
        # The prompt DEFINES the total as the dimension mean, so when the judge
        # mis-adds (e.g. states 30 for dims averaging 28) the parsed dimensions
        # are the judgment and the stated total is a transcription slip.
        # Reconcile instead of failing the sample; missing or out-of-range
        # fields above still fail fast.
        return {
            "score_raw": dimension_mean,
            "score": dimension_mean / 100.0,
            "raw": output_text,
            "total_reconciled": True,
            "stated_total": total,
            **parsed_dimensions,
        }

    return {
        "score_raw": total,
        "score": total / 100.0,
        "raw": output_text,
        **parsed_dimensions,
    }

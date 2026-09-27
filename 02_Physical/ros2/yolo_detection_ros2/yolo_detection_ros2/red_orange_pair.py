"""Decision policy for the red-cuboid/orange-bridge pair classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


PAIR_CLASSES = frozenset({"red_cuboid", "orange_bridge"})


@dataclass(frozen=True)
class PairDecision:
    class_name: str
    confidence: float
    margin: float
    corrected: bool


def choose_pair_class(
    original_name: str,
    names: Mapping[int, str],
    probabilities: Sequence[float],
    *,
    min_confidence: float,
    min_margin: float,
) -> PairDecision:
    """Accept a pair-classifier result only when it is decisive.

    The second stage is intentionally restricted to detections already labelled
    red_cuboid or orange_bridge.  Low-confidence or low-margin results preserve
    the primary detector label instead of forcing a risky class change.
    """
    if original_name not in PAIR_CLASSES or len(probabilities) < 2:
        return PairDecision(original_name, 0.0, 0.0, False)
    order = sorted(
        range(len(probabilities)),
        key=lambda index: float(probabilities[index]),
        reverse=True,
    )
    best_index, runner_up_index = order[:2]
    best_name = str(names.get(best_index, ""))
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[runner_up_index])
    accepted = (
        best_name in PAIR_CLASSES
        and confidence >= min_confidence
        and margin >= min_margin
    )
    class_name = best_name if accepted else original_name
    return PairDecision(
        class_name=class_name,
        confidence=confidence,
        margin=margin,
        corrected=accepted and class_name != original_name,
    )

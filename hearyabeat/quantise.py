"""Snap detected hits to a 16th-note grid."""

from dataclasses import dataclass

import numpy as np

from . import STEPS_PER_BAR

MAX_BARS = 4


@dataclass(frozen=True)
class Hit:
    time: float
    label: str
    level: float
    confidence: float = 1.0  # 0 to 1, how sure the classifier was of the label


@dataclass(frozen=True)
class GridHit:
    step: int
    label: str
    velocity: float
    confidence: float = 1.0


def count_bars(duration, t0, bpm):
    bar_seconds = 60.0 / bpm * 4
    return int(np.clip(round((duration - t0) / bar_seconds), 1, MAX_BARS))


def quantise(hits, bpm, t0, n_steps):
    """Place hits on steps. One hit per step, and the strongest one wins."""
    step_s = 60.0 / bpm / 4
    strongest = {}
    for hit in hits:
        step = int(round((hit.time - t0) / step_s))
        if step == n_steps:  # a slightly early downbeat of the next pass
            step = 0
        if not 0 <= step < n_steps:
            continue
        if step not in strongest or hit.level > strongest[step].level:
            strongest[step] = hit
    if not strongest:
        return []
    top = max(hit.level for hit in strongest.values())
    # Compressed, so quiet ghost notes survive but dynamics still come through.
    return [
        GridHit(step, hit.label, float(0.55 + 0.45 * np.sqrt(hit.level / top)), hit.confidence)
        for step, hit in sorted(strongest.items())
    ]


__all__ = ["Hit", "GridHit", "STEPS_PER_BAR", "count_bars", "quantise"]

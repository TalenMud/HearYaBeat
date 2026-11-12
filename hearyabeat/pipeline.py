"""Audio in, drum loop out. No device access here, so it all runs in tests."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import LABELS, SR, STEPS_PER_BAR
from .classify import UNSURE, extract_features, scorer_for
from .detect import TooFewHits, detect_onsets, estimate_grid
from .quantise import Hit, count_bars, quantise
from .render import render_loop

__all__ = ["Result", "TooFewHits", "LOOSE_TIMING", "analyse", "lanes", "confidence_lanes", "unsure_hits", "save_session", "load_session"]


@dataclass
class Result:
    bpm: int
    bars: int
    n_steps: int
    t0: float
    confidence: float
    hits: list  # every detected Hit, labelled
    grid: list  # the quantised GridHits
    loop: np.ndarray
    sr: int = SR
    bass_notes: list = field(default_factory=list)
    features: np.ndarray = None  # one row per hit, in FEATURE_NAMES order
    timings_ms: dict = field(default_factory=dict)  # stage -> milliseconds, in pipeline order
    take_s: float = 0.0  # length of the analysed take


LOOSE_TIMING = 0.5  # below this timing fit, the hits barely sit on any grid


def analyse(y, samples, sr=SR, hint_bpm=None, downbeat=None, profile=None, lock_bpm=False):
    """Turn a beatbox take into a labelled grid and a rendered loop."""
    timings = {}
    clock = time.perf_counter()

    def lap(stage):
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = (now - clock) * 1000
        clock = now

    times, levels = detect_onsets(y, sr)
    lap("detect")
    grid_fit = estimate_grid(times, y, sr, hint_bpm=hint_bpm, downbeat=downbeat, lock_bpm=lock_bpm)
    lap("tempo")

    score = scorer_for(profile)
    hits, features = [], []
    for t, level in zip(times, levels):
        vector = extract_features(y, t, sr)
        label, confidence = score(vector)
        hits.append(Hit(float(t), label, float(level), confidence))
        features.append(vector)
    lap("classify")

    duration = y.size / sr
    bars = count_bars(duration, grid_fit.t0, grid_fit.bpm)
    n_steps = bars * STEPS_PER_BAR
    grid = quantise(hits, grid_fit.bpm, grid_fit.t0, n_steps)
    loop = render_loop(grid, grid_fit.bpm, n_steps, samples, sr)
    lap("render")
    return Result(grid_fit.bpm, bars, n_steps, grid_fit.t0, grid_fit.confidence, hits, grid, loop, sr,
                  features=np.array(features).reshape(len(hits), -1), timings_ms=timings, take_s=duration)


def lanes(result):
    """Return {label: [bool per step]} for drawing the step grid."""
    table = {label: [False] * result.n_steps for label in LABELS}
    for hit in result.grid:
        table[hit.label][hit.step] = True
    return table


def confidence_lanes(result):
    """Return {label: [confidence per step, or None where there's no hit]}."""
    table = {label: [None] * result.n_steps for label in LABELS}
    for hit in result.grid:
        table[hit.label][hit.step] = hit.confidence
    return table


def unsure_hits(result, threshold=UNSURE):
    return [hit for hit in result.grid if hit.confidence < threshold]


def save_session(path, bpm, bars, n_steps, bass_notes):
    """Remember tempo and bass, so a later hum or new take can build on them."""
    data = {
        "bpm": bpm,
        "bars": bars,
        "n_steps": n_steps,
        "bass_notes": [list(note) for note in bass_notes],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def load_session(path):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["bass_notes"] = [tuple(note) for note in data.get("bass_notes", [])]
    return data

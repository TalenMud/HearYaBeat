"""Ground-truth labels from the script of a take, not from what the app hears.

If you beatbox a pattern you decided in advance, at a known tempo, the labels
can be written from the script: each hit sits where the grid says it should.
That saves labelling by hand, and keeps the truth independent of the app.

This module must never use the detector or the classifier. If the truth came
from them, evaluating against it would only measure the app against itself.
A test checks the imports.

Because the times are where you meant to hit, not where you did, timing error
against these labels includes your own drift from the grid, on top of any
error in the detector.
"""

import math

from . import LABELS, STEPS_PER_BAR
from .audio_io import PRE_ROLL_S

REST = "-"


def parse_pattern(text):
    """Split a pattern into bars of labels, with None for a rest.

    Bars are separated by |, and the hits in a bar by spaces, e.g.
    "kick hat snare hat | kick kick snare hat". Each bar is spread evenly over
    16 steps, so it needs 1, 2, 4, 8 or 16 slots: 4 for quarter notes, 8 for
    8ths. Use - for a rest.
    """
    bars = []
    for number, chunk in enumerate(text.split("|"), start=1):
        slots = chunk.lower().split()
        if not slots:
            raise ValueError(f"Bar {number} of the pattern is empty.")
        if STEPS_PER_BAR % len(slots):
            raise ValueError(f"Bar {number} has {len(slots)} slots. Use 1, 2, 4, 8 or 16, "
                             f"so each one lands on the 16th-note grid (- is a rest).")
        for slot in slots:
            if slot != REST and slot not in LABELS:
                raise ValueError(f"Bar {number}: {slot!r} isn't a sound. Use {', '.join(LABELS)} or - for a rest.")
        bars.append([None if slot == REST else slot for slot in slots])
    if all(label is None for bar in bars for label in bar):
        raise ValueError("The pattern is all rests, so there's nothing to label.")
    return bars


def scripted_truth(pattern, bpm, bars, offset_s=PRE_ROLL_S):
    """Return (times, labels) for a pattern performed exactly on the grid.

    offset_s is where the first downbeat falls in the recording. A take saved
    by the app starts PRE_ROLL_S before its first downbeat, so that's the
    default. The pattern's bars repeat until `bars` bars are filled.
    """
    if bpm <= 0 or bars < 1:
        raise ValueError("BPM must be above 0 and bars at least 1.")
    script = parse_pattern(pattern)
    bar_s = 60.0 / bpm * 4
    times, labels = [], []
    for bar in range(bars):
        slots = script[bar % len(script)]
        for slot, label in enumerate(slots):
            if label is not None:
                times.append(offset_s + bar * bar_s + slot * bar_s / len(slots))
                labels.append(label)
    return times, labels


def check_fits(times, duration_s, bpm):
    """Raise if scripted hits fall after the recording ends, which means the numbers don't match the take."""
    late = [t for t in times if t > duration_s]
    if late:
        bars_heard = math.floor(duration_s / (240.0 / bpm))
        raise ValueError(f"{len(late)} scripted hit(s) fall after the recording ends at {duration_s:.2f} s. "
                         f"Check the BPM, bars and offset: the file holds about {bars_heard} bar(s) at {bpm:g} BPM.")

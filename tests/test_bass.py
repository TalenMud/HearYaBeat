import numpy as np
import pytest

from hearyabeat import SR
from hearyabeat.bass import (
    BASS_HIGH_MIDI,
    BASS_LOW_MIDI,
    HumError,
    fit_key,
    hum_to_notes,
    mix,
    render_bass,
    segment_notes,
    snap_to_scale,
)
from hearyabeat.fake_voice import make_hum

BPM = 100
# A minor pentatonic: A3 for a beat, C4, E4, then a rest and G3.
LINE = [57] * 4 + [60] * 4 + [64] * 4 + [None] * 2 + [55] * 2


def test_hum_gives_the_right_key_and_timing():
    y = make_hum(LINE, BPM)
    root, notes = hum_to_notes(y, BPM, len(LINE))
    assert root == 9  # A
    assert [(n[0], n[1]) for n in notes] == [(0, 4), (4, 4), (8, 4), (14, 2)]
    # Pitch classes survive the move into the bass register.
    assert [n[2] % 12 for n in notes] == [9, 0, 4, 7]
    assert all(BASS_LOW_MIDI <= n[2] <= BASS_HIGH_MIDI for n in notes)


def test_out_of_tune_hum_snaps_into_the_scale():
    assert snap_to_scale(57.4, 9) == 57
    assert snap_to_scale(58.9, 9) == 60  # between A and C, closer to C


def test_key_is_found_regardless_of_root():
    for root in (0, 4, 9):
        scale = [root + s + 48 for s in (0, 3, 5, 7, 10)]
        assert fit_key(np.array(scale * 3, dtype=float)) == root


def test_one_step_slides_are_absorbed():
    assert segment_notes([57, 57, 58, 60, 60]) == [(0, 3, 57), (3, 2, 60)]


def test_silence_raises_a_helpful_error():
    quiet = np.random.default_rng(0).standard_normal(SR * 2) * 1e-5
    with pytest.raises(HumError):
        hum_to_notes(quiet, BPM, 16)


def test_bass_is_exact_length_and_ends_silent():
    length = 100000
    bass = render_bass([(0, 16, 36)], 120, length)
    assert bass.size == length
    assert abs(bass[0]) < 0.05
    last_step_end = int(round(16 * 60 / 120 / 4 * SR))
    assert abs(bass[last_step_end - 1]) < 0.05


def test_mix_keeps_length_and_does_not_clip():
    loop = np.ones(1000) * 0.95
    bass = np.ones(1000)
    mixed = mix(loop, bass)
    assert mixed.size == 1000
    assert np.max(np.abs(mixed)) <= 0.95 + 1e-9

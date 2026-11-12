"""Actions shared by the CLI and the Streamlit page."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io, bass
from .classify import HITS_PER_SOUND, CalibrationError, build_profile, calibration_features, load_profile, save_profile
from .pipeline import LOOSE_TIMING, Result, analyse, lanes, load_session, save_session
from .render import load_samples, load_wav, loop_samples, save_wav

__all__ = [
    "HITS_PER_SOUND", "CalibrationError", "TakeOutcome", "process_take", "add_hum",
    "calibrate_take", "finish_calibration", "lanes", "load_session", "LOOSE_TIMING",
]

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
PROFILES = ROOT / "profiles"
OUTPUT = ROOT / "output"
LOOP_PATH = OUTPUT / "loop.wav"
BASS_PATH = OUTPUT / "loop_bass.wav"
TAKE_PATH = OUTPUT / "last_take.wav"
HUM_PATH = OUTPUT / "last_hum.wav"
SESSION_PATH = OUTPUT / "session.json"
CALIBRATION_SECONDS = 4


@dataclass
class TakeOutcome:
    result: Result
    bass_kept: bool  # a previous bassline carried over to this take
    bass_cleared: bool  # a previous bassline no longer fitted
    path: Path


def process_take(y, hint_bpm=None, downbeat=None, profile_name=None, lock_bpm=False):
    """Analyse a take, export the loop, and carry the bassline over if it fits."""
    profile = load_profile(profile_name, PROFILES) if profile_name else None
    result = analyse(
        y, load_samples(SAMPLES), hint_bpm=hint_bpm, downbeat=downbeat, profile=profile, lock_bpm=lock_bpm
    )
    save_wav(TAKE_PATH, y)
    save_wav(LOOP_PATH, result.loop)

    previous = load_session(SESSION_PATH)
    had_bass = bool(previous and previous["bass_notes"])
    fits = had_bass and previous["bpm"] == result.bpm and previous["n_steps"] == result.n_steps
    path = LOOP_PATH
    if fits:
        result.bass_notes = previous["bass_notes"]
        rendered = bass.render_bass(result.bass_notes, result.bpm, result.loop.size, result.sr)
        path = save_wav(BASS_PATH, bass.mix(result.loop, rendered), result.sr)
    elif BASS_PATH.exists():
        BASS_PATH.unlink()
    save_session(SESSION_PATH, result.bpm, result.bars, result.n_steps, result.bass_notes)
    return TakeOutcome(result, fits, had_bass and not fits, path)


def add_hum(hum, state):
    """Fit a hum to the current loop. Returns (root name, notes, mixed audio path)."""
    bpm, n_steps = state["bpm"], state["n_steps"]
    root, notes = bass.hum_to_notes(hum, bpm, n_steps)
    save_wav(HUM_PATH, hum)
    length = loop_samples(bpm, n_steps)
    loop = np.resize(load_wav(LOOP_PATH), length)
    path = save_wav(BASS_PATH, bass.mix(loop, bass.render_bass(notes, bpm, length)))
    save_session(SESSION_PATH, bpm, state["bars"], n_steps, notes)
    return bass.NOTE_NAMES[root], notes, path


def warm_up():
    """Analyse a short synthetic take, so librosa compiles its code now.

    The first analysis in a process spends a few seconds compiling (about
    3 s measured here, in onset detection and the beat tracker), which
    would otherwise land on the first real take. Tempo is detected freely
    here, not locked, so the beat tracker actually runs. Nothing is saved.
    Returns how long it took in ms.
    """
    from .fake_voice import BASIC_BAR, make_take

    start = time.perf_counter()
    analyse(make_take(BASIC_BAR, 100, bars=1, seed=0)[0], load_samples(SAMPLES))
    return (time.perf_counter() - start) * 1000


def calibrate_take(seconds=CALIBRATION_SECONDS, mic=None):
    """Record one calibration take. Raises CalibrationError on the wrong hit count."""
    y = audio_io.record_seconds(seconds, mic=mic)
    calibration_features(y)
    return y


def finish_calibration(name, takes):
    """Save a profile. Returns (path, LeaveOneOut) so the caller can say how separable the sounds are."""
    profile = build_profile(name, takes)
    return save_profile(profile, PROFILES), profile.leave_one_out()

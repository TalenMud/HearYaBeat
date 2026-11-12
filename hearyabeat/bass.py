"""Turn a hummed line into a synth bassline locked to the loop."""

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from . import SR
from .render import normalise, step_position

MINOR_PENTATONIC = (0, 3, 5, 7, 10)
PITCH_SR = 22050  # pyin is much faster here and loses nothing at humming pitch
BASS_LOW_MIDI = 28  # E1, about 41 Hz
BASS_HIGH_MIDI = 45  # A2, 110 Hz
BASS_CENTRE_MIDI = 36
MIN_NOTE_STEPS = 2
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class HumError(ValueError):
    """No usable pitch was found in the hum."""


def step_pitches(y, bpm, n_steps, sr=SR):
    """Return the median hummed MIDI pitch per step, NaN where nothing was voiced."""
    y22 = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=PITCH_SR)
    hop = 256
    f0, voiced, _ = librosa.pyin(
        y22,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=PITCH_SR,
        frame_length=2048,
        hop_length=hop,
    )
    times = librosa.times_like(f0, sr=PITCH_SR, hop_length=hop)
    step_s = 60.0 / bpm / 4
    pitches = np.full(n_steps, np.nan)
    for step in range(n_steps):
        inside = (times >= step * step_s) & (times < (step + 1) * step_s)
        if inside.sum() and voiced[inside].mean() > 0.5:
            pitches[step] = float(np.nanmedian(librosa.hz_to_midi(f0[inside & voiced])))
    return pitches


def fit_key(pitches, scale=MINOR_PENTATONIC):
    """Find the root whose scale sits closest to the pitches actually hummed."""
    voiced = pitches[~np.isnan(pitches)]
    best_root, best_cost = 0, np.inf
    for root in range(12):
        classes = (voiced - root) % 12
        distance = np.min(
            [np.minimum(np.abs(classes - s), 12 - np.abs(classes - s)) for s in scale], axis=0
        )
        # The tiny credit breaks ties in favour of the note the line sits on.
        on_root = np.minimum(classes, 12 - classes) < 0.5
        cost = float(distance.sum()) - 0.01 * float(on_root.sum())
        if cost < best_cost:
            best_root, best_cost = root, cost
    return best_root


def snap_to_scale(midi, root, scale=MINOR_PENTATONIC):
    """Return the nearest MIDI note in the scale."""
    allowed = [root + s + 12 * octave for octave in range(-1, 11) for s in scale]
    return int(min(allowed, key=lambda note: abs(note - midi)))


def to_bass_range(notes):
    """Move a list of (step, length, midi) down into the bass register."""
    if not notes:
        return []
    shift = 12 * int(round((BASS_CENTRE_MIDI - float(np.median([n[2] for n in notes]))) / 12))
    moved = []
    for step, length, midi in notes:
        midi = midi + shift
        while midi > BASS_HIGH_MIDI:
            midi -= 12
        while midi < BASS_LOW_MIDI:
            midi += 12
        moved.append((step, length, midi))
    return moved


def segment_notes(snapped):
    """Group per-step notes (None for rests) into (step, length, midi)."""
    notes = []
    for step, midi in enumerate(snapped):
        if midi is None:
            continue
        if notes and notes[-1][2] == midi and notes[-1][0] + notes[-1][1] == step:
            notes[-1][1] += 1
        else:
            notes.append([step, 1, midi])

    # One-step notes are nearly always the slide between two real notes.
    cleaned = []
    for step, length, midi in notes:
        if length >= MIN_NOTE_STEPS:
            cleaned.append([step, length, midi])
        elif cleaned and cleaned[-1][0] + cleaned[-1][1] == step:
            cleaned[-1][1] += length
    return [tuple(note) for note in cleaned]


def hum_to_notes(y, bpm, n_steps, sr=SR):
    """Return (root pitch class, [(step, length, midi)]) for a hum."""
    pitches = step_pitches(y, bpm, n_steps, sr)
    if np.isnan(pitches).all():
        raise HumError("Couldn't hear a pitch. Try humming louder, or closer to the mic.")
    root = fit_key(pitches)
    snapped = [None if np.isnan(p) else snap_to_scale(p, root) for p in pitches]
    notes = to_bass_range(segment_notes(snapped))
    if not notes:
        raise HumError("Couldn't hear a steady note. Try holding each note a little longer.")
    return root, notes


def render_bass(notes, bpm, length, sr=SR):
    """Synthesise notes into a buffer of exactly length samples."""
    out = np.zeros(length)
    lowpass = butter(2, 420, "lowpass", fs=sr, output="sos")
    for step, steps, midi in notes:
        start = step_position(step, bpm, sr)
        end = min(step_position(step + steps, bpm, sr), length)
        n = end - start
        if n <= 0:
            continue
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(n) / sr
        saw = 2 * ((freq * t) % 1) - 1
        tone = 0.5 * sosfilt(lowpass, saw) + 0.8 * np.sin(2 * np.pi * freq * t)
        # The envelope goes on after the filter, so every note ends at zero and
        # nothing can click, including at the loop join.
        attack, release = min(int(0.005 * sr), n // 2), min(int(0.02 * sr), n // 2)
        envelope = 0.65 + 0.35 * np.exp(-t / 0.12)
        envelope[:attack] *= np.linspace(0, 1, attack, endpoint=False)
        if release:
            envelope[-release:] *= np.linspace(1, 0, release)
        out[start:end] += tone * envelope
    return out


def mix(loop, bass, gain=0.8):
    return normalise(loop + normalise(bass) * gain)


def note_label(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

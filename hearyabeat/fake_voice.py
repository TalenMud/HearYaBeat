"""Synthetic stand-ins for a beatboxing voice, with a known right answer.

Used by the tests and to make examples/demo_take.wav. These are much cleaner
than a real voice, so passing on them says nothing about real-mic accuracy.

Regenerate the demo take with: python -m hearyabeat.fake_voice
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from . import SR
from .render import save_wav

# "boom cha" with hats on the 8ths, over one bar.
BASIC_BAR = [
    (0, "kick"), (2, "hat"), (4, "snare"), (6, "hat"),
    (8, "kick"), (10, "kick"), (12, "snare"), (14, "hat"),
]


def _noise(rng, n):
    return rng.standard_normal(n)


def voice_kick(rng, pitch=1.0, sr=SR):
    t = np.arange(int(0.11 * sr)) / sr
    f = 130 * pitch
    tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    return tone * np.exp(-t / 0.035)


def voice_snare(rng, pitch=1.0, sr=SR):
    t = np.arange(int(0.13 * sr)) / sr
    low, high = 1500 * pitch, min(6000 * pitch, sr / 2 - 100)
    sos = butter(2, [low, high], "bandpass", fs=sr, output="sos")
    return sosfilt(sos, _noise(rng, t.size)) * np.exp(-t / 0.05) * 2.2


def voice_hat(rng, pitch=1.0, sr=SR):
    t = np.arange(int(0.045 * sr)) / sr
    sos = butter(4, min(7500 * pitch, sr / 2 - 500), "highpass", fs=sr, output="sos")
    return sosfilt(sos, _noise(rng, t.size)) * np.exp(-t / 0.012) * 1.6


VOICES = {"kick": voice_kick, "snare": voice_snare, "hat": voice_hat}


def make_take(pattern, bpm, bars=2, sr=SR, jitter_s=0.0, noise=0.0005, seed=0,
              pitch=1.0, lead_s=0.05, tail_s=0.25, shapes=None):
    """Perform a one-bar pattern for several bars. Returns (audio, onset times)."""
    rng = np.random.default_rng(seed)
    shapes = shapes or VOICES
    step_s = 60.0 / bpm / 4
    y = np.zeros(int((lead_s + bars * 16 * step_s + tail_s) * sr))
    onsets = []
    for bar in range(bars):
        for step, label in pattern:
            t = lead_s + (bar * 16 + step) * step_s + rng.normal(0, jitter_s)
            t = max(t, 0.0)
            start = int(t * sr)
            sound = shapes[label](rng, pitch, sr) * rng.uniform(0.7, 1.0)
            end = min(start + sound.size, y.size)
            y[start:end] += sound[: end - start]
            onsets.append(t)
    y += rng.standard_normal(y.size) * noise
    return y / max(1.0, float(np.max(np.abs(y)))) * 0.8, np.array(onsets)


def make_repeats(label, count=3, gap_s=0.5, sr=SR, seed=0, pitch=1.0, noise=0.0005, shapes=None):
    """A calibration take: one sound repeated with a gap between."""
    rng = np.random.default_rng(seed)
    shapes = shapes or VOICES
    y = np.zeros(int((0.2 + count * gap_s + 0.3) * sr))
    for i in range(count):
        start = int((0.2 + i * gap_s) * sr)
        sound = shapes[label](rng, pitch, sr) * rng.uniform(0.7, 1.0)
        y[start : start + sound.size] += sound
    y += rng.standard_normal(y.size) * noise
    return y / max(1.0, float(np.max(np.abs(y)))) * 0.8


def make_hum(midi_by_step, bpm, sr=SR, seed=0, noise=0.0003):
    """A hummed line: one MIDI note (or None for silence) per 16th step."""
    rng = np.random.default_rng(seed)
    step = int(round(60.0 / bpm / 4 * sr))
    y = np.zeros(step * len(midi_by_step))
    phase = 0.0
    for i, midi in enumerate(midi_by_step):
        if midi is None:
            continue
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(step)
        chunk = sum(
            amp * np.sin(2 * np.pi * freq * harmonic * t / sr + phase * harmonic)
            for harmonic, amp in ((1, 1.0), (2, 0.45), (3, 0.2))
        )
        phase += 2 * np.pi * freq * step / sr
        y[i * step : (i + 1) * step] = chunk
    y += rng.standard_normal(y.size) * noise
    return y / max(1.0, float(np.max(np.abs(y)))) * 0.7


# One bar of A minor pentatonic, one entry per 16th step (None is a breath).
DEMO_RIFF = [57] * 4 + [None] * 2 + [60] * 2 + [64] * 4 + [62] * 2 + [55] * 2


def demo_hum(bpm, n_steps, sr=SR):
    """A hum of DEMO_RIFF that fits any loop, for trying the bassline without a mic."""
    bars = -(-n_steps // 16)
    return make_hum((DEMO_RIFF * bars)[:n_steps], bpm, sr=sr, seed=5)


if __name__ == "__main__":
    examples = Path(__file__).resolve().parent.parent / "examples"
    from .evaluate import save_truth

    audio, onsets = make_take(BASIC_BAR, bpm=100, bars=2, jitter_s=0.008, seed=7)
    print("Wrote", save_wav(examples / "demo_take.wav", audio))
    # make_take returns onsets in the order it performed them: bar by bar, through the pattern.
    labels = [label for _ in range(2) for _, label in BASIC_BAR]
    print("Wrote", save_truth(examples / "demo_take.csv", onsets, labels))
    print("Wrote", save_wav(examples / "demo_hum.wav", demo_hum(100, 32)))

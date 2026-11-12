"""Synthesise the default drum kit, so the repo ships with licence-free sounds.

Overwrite samples/kick.wav, snare.wav or hat.wav with your own recordings to
change the kit. Any sample rate or channel count is accepted.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from . import SR


def _time(duration):
    return np.arange(int(duration * SR)) / SR


def kick():
    t = _time(0.32)
    freq = 48 + 130 * np.exp(-t / 0.035)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / 0.11)
    click = np.exp(-t / 0.002) * 0.35
    return _finish(body + click)


def snare(seed=1):
    t = _time(0.24)
    rng = np.random.default_rng(seed)
    noise = sosfilt(butter(2, 1400, "highpass", fs=SR, output="sos"), rng.standard_normal(t.size))
    rattle = noise * np.exp(-t / 0.07)
    tone = (np.sin(2 * np.pi * 185 * t) + 0.6 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.05)
    return _finish(0.14 * rattle + 0.55 * tone)


def hat(seed=2):
    t = _time(0.09)
    rng = np.random.default_rng(seed)
    noise = sosfilt(butter(4, 7500, "highpass", fs=SR, output="sos"), rng.standard_normal(t.size))
    return _finish(noise * np.exp(-t / 0.018)) * 0.6


def _finish(x):
    fade = min(256, x.size)
    x = x.copy()
    x[-fade:] *= np.linspace(1, 0, fade)
    return x / (np.max(np.abs(x)) + 1e-12)


def generate(directory):
    """Write the three default samples into directory and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, make in (("kick", kick), ("snare", snare), ("hat", hat)):
        path = directory / f"{name}.wav"
        sf.write(path, make(), SR, subtype="PCM_16")
        paths.append(path)
    return paths

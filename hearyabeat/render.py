"""Audio file handling, and placing drum samples on the grid."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import LABELS, SR


def load_wav(path, sr=SR):
    """Read an audio file as mono float at sample rate sr."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def save_wav(path, y, sr=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


def load_samples(directory, sr=SR):
    """Load kick, snare and hat from directory."""
    directory = Path(directory)
    missing = [n for n in LABELS if not (directory / f"{n}.wav").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {directory}. Run: python cli.py gen-samples"
        )
    return {name: load_wav(directory / f"{name}.wav", sr) for name in LABELS}


def step_seconds(bpm):
    return 60.0 / bpm / 4


def loop_samples(bpm, n_steps, sr=SR):
    return int(round(n_steps * step_seconds(bpm) * sr))


def step_position(step, bpm, sr=SR):
    # From absolute time, so rounding cannot accumulate into drift.
    return int(round(step * step_seconds(bpm) * sr))


def normalise(y, peak=0.95):
    top = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (peak / top) if top > 0 else y


def render_loop(grid, bpm, n_steps, samples, sr=SR):
    """Render GridHits to one seamless pass of the loop.

    Sample tails that run past the end wrap to the start, so the join is
    exactly what you would hear on the second time round.
    """
    length = loop_samples(bpm, n_steps, sr)
    out = np.zeros(length, dtype=np.float64)
    for hit in grid:
        sample = samples[hit.label]
        start = step_position(hit.step, bpm, sr)
        indices = (start + np.arange(sample.size)) % length
        np.add.at(out, indices, sample * hit.velocity)
    return normalise(out)

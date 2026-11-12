"""Find each hit in a take, and work out the tempo and grid behind them."""

from dataclasses import dataclass

import librosa
import numpy as np

from . import HOP, MAX_BPM, MIN_BPM, SR

MIN_GAP_S = 0.06  # two onsets closer than this are one hit
ABSOLUTE_FLOOR = 1e-3  # below this peak RMS the take counts as silence
RELATIVE_FLOOR = 0.07  # hits quieter than this fraction of the peak are noise
GHOST_FRACTION = 0.3  # and so are hits this far below a typical hit
LEVEL_WINDOW_S = 0.08
MAX_BACKTRACK_S = 0.05
ATTACK_FRACTION = 0.15  # an attack starts where the level passes this share of its peak
COMMON_BPM = 105


class TooFewHits(ValueError):
    """The take did not contain enough hits to analyse."""


@dataclass(frozen=True)
class Grid:
    bpm: int
    t0: float  # time of step 0, in seconds
    confidence: float  # 0 to 1, how tightly the hits sit on the grid


def detect_onsets(y, sr=SR):
    """Return (onset times in seconds, RMS level of each hit)."""
    empty = np.array([]), np.array([])
    if y.size < HOP * 4:
        return empty
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    peak_rms = float(rms.max())
    if peak_rms < ABSOLUTE_FLOOR:
        return empty

    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    peaks = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=HOP,
        units="frames",
        backtrack=False,
        wait=max(1, int(MIN_GAP_S * sr / HOP)),
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=0.05,
    )
    if len(peaks) == 0:
        return empty

    window = max(1, int(LEVEL_WINDOW_S * sr / HOP))
    levels = np.array([rms[p : p + window].max() for p in peaks])
    keep = levels >= RELATIVE_FLOOR * peak_rms
    peaks, levels = peaks[keep], levels[keep]
    if len(peaks) == 0:
        return empty

    times, levels = _merge_close(
        np.array([_attack_start(y, int(p) * HOP, sr) for p in peaks]), levels
    )
    # Breaths and mouth noise sit far below the real hits. The 90th percentile
    # stands for a typical real hit, so one loud hit doesn't silence the rest.
    keep = levels >= GHOST_FRACTION * np.percentile(levels, 90)
    return times[keep], levels[keep]


def _merge_close(times, levels):
    """A "ts" can show two peaks. Once both are walked back to their attack
    they sit only a few ms apart, so keep the first time and the louder level."""
    order = np.argsort(times)
    merged_times, merged_levels = [], []
    for t, level in zip(times[order], levels[order]):
        if merged_times and t - merged_times[-1] < MIN_GAP_S:
            merged_levels[-1] = max(merged_levels[-1], level)
        else:
            merged_times.append(t)
            merged_levels.append(level)
    return np.array(merged_times), np.array(merged_levels)


def _attack_start(y, centre, sr):
    """Walk back from a hit's loudest point to where its attack begins.

    librosa's onset_backtrack lands on RMS dips 12 to 33 ms early on these
    sounds, and by a different amount per hit, so this uses a much finer
    envelope instead.
    """
    low = max(0, centre - int(MAX_BACKTRACK_S * sr))
    high = min(y.size, centre + int(0.04 * sr))
    segment = np.asarray(y[low:high], dtype=np.float64)
    if segment.size == 0:
        return centre / sr
    size = min(64, segment.size)
    envelope = np.sqrt(np.convolve(segment**2, np.ones(size) / size, mode="same"))
    loudest = int(np.argmax(envelope))
    quiet = np.flatnonzero(envelope[: loudest + 1] < ATTACK_FRACTION * envelope[loudest])
    start = int(quiet[-1]) + 1 if quiet.size else 0
    return (low + start) / sr


def _beat_track_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    except Exception:  # very short takes can fail inside the tracker; it is only a hint
        return None
    tempo = float(np.atleast_1d(tempo)[0])
    return tempo if tempo > 0 else None


def _closeness(bpm, target, width):
    return float(np.exp(-0.5 * (np.log2(bpm / target) / width) ** 2))


def _metric_weight(steps):
    """Hits usually land on beats and 8ths. This separates true tempos from
    other grids that happen to contain the same onsets.

    Beats and 8ths score the same on purpose. Rewarding beats more would make
    every pattern prefer double its real tempo, where its 8ths become beats.
    """
    return float(np.where(steps % 2 == 0, 1.0, 0.2).mean())


def _fit_grid(times, bpm, downbeat):
    """Fit a 16th grid at bpm. Returns (t0, tightness, steps)."""
    step = 15.0 / bpm
    resultant = np.exp(2j * np.pi * times / step).mean()
    tightness = float(np.abs(resultant))
    grid_phase = float(np.angle(resultant)) / (2 * np.pi) * step
    deviation = (times[0] - grid_phase + step / 2) % step - step / 2
    t0 = float(times[0] - deviation)
    if downbeat is not None:
        # The onsets give the fine phase. The count-in only says how many whole
        # steps of rest came before the first hit.
        t0 -= max(0, int(round((t0 - downbeat) / step))) * step
    steps = np.rint((times - t0) / step).astype(int)
    return t0, tightness, steps


def estimate_grid(times, y, sr=SR, hint_bpm=None, downbeat=None, lock_bpm=False):
    """Choose the BPM and grid that best explain the onset times.

    beat_track alone is unreliable on a few seconds of audio, so it is only one
    vote. hint_bpm is the count-in tempo the performer was following. With
    lock_bpm, a click played all through the take, so hint_bpm is the tempo
    and only the grid's position is fitted.
    """
    if len(times) < 2 or (len(times) < 4 and hint_bpm is None):
        raise TooFewHits(f"Only heard {len(times)} hit(s). Try a longer or louder take.")
    if lock_bpm and hint_bpm is None:
        raise ValueError("lock_bpm needs hint_bpm")

    if len(times) < 4 or lock_bpm:
        bpm = int(np.clip(round(hint_bpm), MIN_BPM, MAX_BPM))
        t0, tightness, _ = _fit_grid(times, bpm, downbeat)
        return Grid(bpm, t0, tightness)

    tracked = _beat_track_bpm(y, sr)
    best, best_score = None, -1.0
    for bpm in range(MIN_BPM, MAX_BPM + 1):
        t0, tightness, steps = _fit_grid(times, bpm, downbeat)
        hint = _closeness(bpm, hint_bpm, 0.08) if hint_bpm else 0.0
        vote = 0.0
        if tracked:
            vote = max(_closeness(bpm, tracked * m, 0.05) for m in (0.5, 1, 2))
        # A half or double of the true tempo fits equally well, so a gentle
        # pull towards common tempos settles those ties.
        common = _closeness(bpm, COMMON_BPM, 0.7)
        score = (
            tightness
            * (0.55 + 0.45 * _metric_weight(steps))
            * (1 + 0.3 * hint + 0.1 * vote + 0.08 * common)
        )
        if score > best_score:
            best, best_score = Grid(bpm, t0, tightness), score
    return best

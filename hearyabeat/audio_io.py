"""Microphone and speaker access.

sounddevice is imported inside each function, so analysing a WAV file still
works on a machine without PortAudio.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import SR
from .render import loop_samples

PRE_ROLL_S = 0.05
TAIL_S = 0.25
TICK_S = 0.03


class AudioDeviceError(RuntimeError):
    """The microphone or speakers could not be used."""


@dataclass(frozen=True)
class Progress:
    """Where the performer is, for drawing a live progress display."""

    counting_in: bool
    beat: int  # 1 to 4 within the bar
    bar: int  # 1-based, and 0 during the count-in
    bars: int
    fraction: float  # how much of the take is done, 0 to 1
    count_left: int  # clicks still to come in the count-in


def progress_at(elapsed, bpm, bars, count_in_beats=4):
    """Return the Progress for a time in seconds since the first click."""
    beat_s = 60.0 / bpm
    if elapsed < count_in_beats * beat_s:
        beat = int(max(elapsed, 0.0) // beat_s)
        return Progress(True, beat % 4 + 1, 0, bars, 0.0, count_in_beats - beat)
    into_take = elapsed - count_in_beats * beat_s
    total_beats = bars * 4
    beat = min(int(into_take // beat_s), total_beats - 1)
    fraction = min(into_take / (total_beats * beat_s), 1.0)
    return Progress(False, beat % 4 + 1, beat // 4 + 1, bars, fraction, 0)


def _sounddevice():
    try:
        import sounddevice
    except OSError as error:
        raise AudioDeviceError(
            "PortAudio is not installed. On Linux: sudo apt install libportaudio2"
        ) from error
    return sounddevice


def _click(sr, accent, volume=0.8):
    t = np.arange(int(0.03 * sr)) / sr
    return np.sin(2 * np.pi * (1500 if accent else 1000) * t) * np.exp(-t / 0.006) * volume


def click_track(bpm, beats, sr=SR, volume=0.8):
    """Clicks on every beat, with the first beat of each bar accented."""
    beat = 60.0 / bpm
    length = int(round(beats * beat * sr))
    track = np.zeros(length)
    for i in range(beats):
        click = _click(sr, accent=i % 4 == 0, volume=volume)
        start = int(round(i * beat * sr))
        track[start : start + click.size] += click[: length - start]
    return track


def count_in_track(bpm, beats, sr=SR):
    """Return (click audio, sample at which the take begins)."""
    track = click_track(bpm, beats, sr)
    return track, track.size


def _play_and_record(track, sr, mic, speaker, on_tick=None):
    """Run output and input on one stream. Returns (recording, latency in samples).

    on_tick, if given, is called about every 30 ms with the seconds of audio
    the performer has heard so far.
    """
    sd = _sounddevice()
    try:
        recording = sd.playrec(
            track.astype(np.float32).reshape(-1, 1),
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=(mic, speaker),
        )
        stream = sd.get_stream()
        input_latency, output_latency = stream.latency
        started = time.monotonic()
        while on_tick is not None and stream.active:
            on_tick(time.monotonic() - started - output_latency)
            time.sleep(TICK_S)
        sd.wait()
    except sd.PortAudioError as error:
        raise AudioDeviceError(
            f"Could not open the audio devices ({error}). "
            "Check the mic is connected and allowed, or pick devices with --mic and --speaker."
        ) from error
    latency = int(round((input_latency + output_latency) * sr))
    return recording[:, 0].astype(np.float64), latency


def _ticker(on_progress, bpm, bars, count_in_beats):
    if on_progress is None:
        return None
    return lambda elapsed: on_progress(progress_at(elapsed, bpm, bars, count_in_beats))


def record_take(bpm, bars, count_in_beats=4, sr=SR, mic=None, speaker=None,
                click=True, on_progress=None):
    """Play a count-in and record a take.

    With click, the click carries on quietly through the take to keep the
    performer in time. That needs headphones, or the mic hears it too.
    Returns (audio, downbeat time within that audio). Everything before the
    pre-roll is cut, which keeps the count-in out of the analysis.
    """
    bar_seconds = 60.0 / bpm * 4
    clicks, take_start = count_in_track(bpm, count_in_beats, sr)
    take_length = int(round((bars * bar_seconds + TAIL_S) * sr))
    during = np.zeros(take_length + sr)
    if click:
        guide = click_track(bpm, bars * 4, sr, volume=0.45)
        during[: guide.size] = guide
    track = np.concatenate([clicks, during])
    recording, latency = _play_and_record(
        track, sr, mic, speaker, _ticker(on_progress, bpm, bars, count_in_beats)
    )
    pre_roll = int(PRE_ROLL_S * sr)
    start = take_start + latency - pre_roll
    return recording[start : start + pre_roll + take_length], PRE_ROLL_S


def record_seconds(seconds, sr=SR, mic=None):
    """Record a plain take with no click (used for calibration)."""
    sd = _sounddevice()
    try:
        recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=mic)
        sd.wait()
    except sd.PortAudioError as error:
        raise AudioDeviceError(f"Could not open the microphone ({error}).") from error
    return recording[:, 0].astype(np.float64)


def record_over_loop(loop, bpm, n_steps, sr=SR, mic=None, speaker=None, count_in_beats=4,
                     on_progress=None):
    """Play a count-in and one loop pass, and return the hum aligned to the loop."""
    length = loop_samples(bpm, n_steps, sr)
    clicks, loop_start = count_in_track(bpm, count_in_beats, sr)
    track = np.concatenate([clicks, loop, np.zeros(sr)])
    bars = max(1, n_steps // 16)
    recording, latency = _play_and_record(
        track, sr, mic, speaker, _ticker(on_progress, bpm, bars, count_in_beats)
    )
    start = loop_start + latency
    return recording[start : start + length]


def play(y, sr=SR, repeats=1, speaker=None):
    """Play audio, blocking until it finishes."""
    sd = _sounddevice()
    try:
        sd.play(np.tile(y, repeats).astype(np.float32), samplerate=sr, device=speaker)
        sd.wait()
    except sd.PortAudioError as error:
        raise AudioDeviceError(f"Could not open the speakers ({error}).") from error


def list_devices():
    return str(_sounddevice().query_devices())

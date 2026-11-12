import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cli
from hearyabeat.evaluate import evaluate_take, load_truth
from hearyabeat.render import load_wav
from hearyabeat.script import check_fits, parse_pattern, scripted_truth

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DEMO_PATTERN = "kick hat snare hat kick kick snare hat"  # BASIC_BAR, in 8ths


def test_the_script_never_loads_the_detector_or_classifier():
    # Truth built by the app would make the evaluation circular. A fresh
    # process shows what importing the module pulls in, directly or not.
    code = ("import sys, hearyabeat.script; "
            "print(sorted(m for m in sys.modules if m.startswith('hearyabeat')))")
    loaded = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout
    for module in ("detect", "classify", "pipeline", "quantise", "session"):
        assert f"hearyabeat.{module}'" not in loaded


def test_hits_land_on_the_grid_from_the_offset():
    times, labels = scripted_truth("kick - snare hat", bpm=120, bars=2, offset_s=0.1)
    # At 120 BPM a bar is 2 s and each of 4 slots is 0.5 s.
    assert np.allclose(times, [0.1, 1.1, 1.6, 2.1, 3.1, 3.6])
    assert labels == ["kick", "snare", "hat"] * 2


def test_bars_cycle_through_the_pattern():
    _, labels = scripted_truth("kick | snare | hat", bpm=100, bars=4)
    assert labels == ["kick", "snare", "hat", "kick"]


def test_bars_can_use_different_resolutions():
    times, _ = scripted_truth("kick snare | kick hat hat hat", bpm=60, bars=2, offset_s=0)
    # A bar at 60 BPM is 4 s: halves in bar 1, quarters in bar 2.
    assert np.allclose(times, [0, 2, 4, 5, 6, 7])


@pytest.mark.parametrize("pattern, message", [
    ("kick hat snare", "3 slots"),
    ("kick clap", "isn't a sound"),
    ("kick | ", "empty"),
    ("- - - -", "all rests"),
])
def test_bad_patterns_say_what_is_wrong(pattern, message):
    with pytest.raises(ValueError, match=message):
        parse_pattern(pattern)


def test_hits_after_the_recording_ends_are_refused():
    times, _ = scripted_truth("kick hat snare hat", bpm=100, bars=4)
    with pytest.raises(ValueError, match="after the recording ends"):
        check_fits(times, duration_s=5.0, bpm=100)


def test_scripted_truth_for_the_demo_matches_its_exact_labels_within_its_jitter():
    exact, exact_labels = load_truth(EXAMPLES / "demo_take.csv")
    times, labels = scripted_truth(DEMO_PATTERN, bpm=100, bars=2)
    assert labels == exact_labels
    # The demo was performed with 8 ms of random jitter around this grid.
    assert np.max(np.abs(np.array(times) - exact)) < 0.035


def test_timing_error_against_the_script_includes_the_performers_drift():
    y = load_wav(EXAMPLES / "demo_take.wav")
    exact = evaluate_take(y, *load_truth(EXAMPLES / "demo_take.csv"))
    scripted = evaluate_take(y, *scripted_truth(DEMO_PATTERN, bpm=100, bars=2))
    assert scripted.recall == 1.0 and scripted.accuracy == 1.0
    # Same detections, so the extra error is the demo's jitter, not the detector.
    assert scripted.median_error_ms > exact.median_error_ms + 3


def test_cli_writes_labels_next_to_the_wav_and_wont_overwrite(tmp_path, capsys):
    wav = tmp_path / "take.wav"
    wav.write_bytes((EXAMPLES / "demo_take.wav").read_bytes())
    args = ["script-labels", str(wav), "--pattern", DEMO_PATTERN, "--bpm", "100", "--bars", "2"]
    assert cli.main(args) == 0
    times, labels = load_truth(tmp_path / "take.csv")
    assert len(times) == 16 and labels[:4] == ["kick", "hat", "snare", "hat"]
    assert "includes your own drift" in capsys.readouterr().out

    assert cli.main(args) == 1
    assert "already exists" in capsys.readouterr().err
    assert cli.main(args + ["--force"]) == 0


def test_cli_refuses_a_script_longer_than_the_take(tmp_path, capsys):
    out = tmp_path / "labels.csv"
    assert cli.main(["script-labels", str(EXAMPLES / "demo_take.wav"), "--pattern", DEMO_PATTERN,
                     "--bpm", "100", "--bars", "4", "--out", str(out)]) == 1
    assert "after the recording ends" in capsys.readouterr().err
    assert not out.exists()

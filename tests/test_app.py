"""CLI, session rules and the Streamlit page, with audio devices stubbed out."""

import numpy as np
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import cli
from hearyabeat import SR, audio_io, samples_gen, session
from hearyabeat.fake_voice import BASIC_BAR, make_hum, make_take
from hearyabeat.render import load_wav, loop_samples, save_wav


APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Keep every test away from the real output and profile folders."""
    samples_gen.generate(tmp_path / "samples")
    output = tmp_path / "output"
    monkeypatch.setattr(session, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(session, "PROFILES", tmp_path / "profiles")
    monkeypatch.setattr(session, "LOOP_PATH", output / "loop.wav")
    monkeypatch.setattr(session, "BASS_PATH", output / "loop_bass.wav")
    monkeypatch.setattr(session, "TAKE_PATH", output / "last_take.wav")
    monkeypatch.setattr(session, "HUM_PATH", output / "last_hum.wav")
    monkeypatch.setattr(session, "SESSION_PATH", output / "session.json")
    return tmp_path


def take(bpm):
    return make_take(BASIC_BAR, bpm, jitter_s=0.005, seed=3)[0]


def hum_for(state):
    line = ([57] * 4 + [60] * 4) * (state["n_steps"] // 8)
    return make_hum(line, state["bpm"])


def test_record_from_file_prints_bpm_and_writes_the_loop(sandbox, capsys):
    path = save_wav(sandbox / "take.wav", take(100))
    assert cli.main(["record", "--input", str(path), "--no-play"]) == 0
    printed = capsys.readouterr().out
    assert "BPM 100" in printed and "kick" in printed
    assert session.LOOP_PATH.exists()


def test_hum_needs_a_loop_first(capsys):
    assert cli.main(["hum", "--no-play"]) == 1
    assert "No loop yet" in capsys.readouterr().err


def test_silent_take_gives_a_message_not_a_traceback(sandbox, capsys):
    path = save_wav(sandbox / "quiet.wav", np.zeros(SR * 2))
    assert cli.main(["record", "--input", str(path), "--no-play"]) == 1
    assert "hit" in capsys.readouterr().err


def test_unknown_profile_gives_a_message(sandbox, capsys):
    path = save_wav(sandbox / "take.wav", take(100))
    assert cli.main(["record", "--input", str(path), "--no-play", "--profile", "nobody"]) == 1
    assert "No profile" in capsys.readouterr().err


def test_hum_from_file_writes_a_mix_as_long_as_the_loop(sandbox):
    session.process_take(take(100), hint_bpm=100, downbeat=0.05)
    state = session.load_session(session.SESSION_PATH)
    hum_path = save_wav(sandbox / "hum.wav", hum_for(state))
    assert cli.main(["hum", "--input", str(hum_path), "--no-play"]) == 0
    assert load_wav(session.BASS_PATH).size == loop_samples(state["bpm"], state["n_steps"])


def test_new_take_at_same_tempo_keeps_the_bass():
    session.process_take(take(100), hint_bpm=100, downbeat=0.05)
    state = session.load_session(session.SESSION_PATH)
    session.add_hum(hum_for(state), state)
    outcome = session.process_take(take(100), hint_bpm=100, downbeat=0.05)
    assert outcome.bass_kept and not outcome.bass_cleared
    assert outcome.path == session.BASS_PATH
    assert session.load_session(session.SESSION_PATH)["bass_notes"]


def test_new_take_at_new_tempo_clears_the_bass():
    session.process_take(take(100), hint_bpm=100, downbeat=0.05)
    state = session.load_session(session.SESSION_PATH)
    session.add_hum(hum_for(state), state)
    outcome = session.process_take(take(140), hint_bpm=140, downbeat=0.05)
    assert outcome.bass_cleared and not outcome.bass_kept
    assert outcome.path == session.LOOP_PATH
    assert not session.BASS_PATH.exists()
    assert session.load_session(session.SESSION_PATH)["bass_notes"] == []


def test_record_take_removes_the_click_and_the_latency(monkeypatch):
    bpm, bars = 120, 1
    clicks, take_start = audio_io.count_in_track(bpm, 4)
    latency = 2205
    performed = np.ones(1000)

    def play_and_record(track, sr, mic, speaker, on_tick=None):
        recording = np.zeros(track.size)
        recording[:take_start] = track[:take_start]  # the click bleeds into the mic
        recording[take_start + latency : take_start + latency + 1000] = performed
        return recording, latency

    monkeypatch.setattr(audio_io, "_play_and_record", play_and_record)
    y, downbeat = audio_io.record_take(bpm, bars)
    pre_roll = int(downbeat * SR)
    assert np.all(y[:pre_roll] == 0)  # none of the count-in survives
    assert np.all(y[pre_roll : pre_roll + 1000] == 1)  # performance starts at the downbeat
    assert y.size == pre_roll + int(round((2.0 + audio_io.TAIL_S) * SR))


def test_hum_recording_is_trimmed_to_one_aligned_loop_pass(monkeypatch):
    bpm, steps = 120, 16
    length = loop_samples(bpm, steps)
    latency = 1500
    _, loop_start = audio_io.count_in_track(bpm, 4)

    def play_and_record(track, sr, mic, speaker, on_tick=None):
        recording = np.zeros(track.size)
        recording[loop_start + latency : loop_start + latency + length] = 1.0
        return recording, latency

    monkeypatch.setattr(audio_io, "_play_and_record", play_and_record)
    hum = audio_io.record_over_loop(np.zeros(length), bpm, steps)
    assert hum.size == length and np.all(hum == 1.0)


def test_page_loads_and_demo_take_shows_a_result():
    page = AppTest.from_file(APP, default_timeout=60).run()
    assert not page.exception
    page.button(key="demo").click().run()
    assert not page.exception
    assert page.metric[0].value == "100"


def test_page_records_with_a_stubbed_mic(monkeypatch):
    monkeypatch.setattr(audio_io, "record_take", lambda bpm, bars, **kw: (take(100), 0.05))
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="record").click().run()
    assert not page.exception
    assert page.metric[0].value == "100"


def test_page_shows_mic_failures_as_a_message(monkeypatch):
    def broken(bpm, bars, **kw):
        raise audio_io.AudioDeviceError("No microphone found.")

    monkeypatch.setattr(audio_io, "record_take", broken)
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="record").click().run()
    assert not page.exception
    assert page.error[0].value == "No microphone found."


def test_progress_follows_count_in_then_bars_and_beats():
    bpm, bars = 120, 2  # half a second per beat
    first = audio_io.progress_at(0.1, bpm, bars)
    assert first.counting_in and first.count_left == 4
    assert audio_io.progress_at(1.6, bpm, bars).count_left == 1
    start = audio_io.progress_at(2.0, bpm, bars)
    assert not start.counting_in and (start.bar, start.beat) == (1, 1) and start.fraction == 0
    later = audio_io.progress_at(2.0 + 5 * 0.5 + 0.1, bpm, bars)  # sixth beat of the take
    assert (later.bar, later.beat) == (2, 2)
    end = audio_io.progress_at(60.0, bpm, bars)
    assert (end.bar, end.beat, end.fraction) == (2, 4, 1.0)


@pytest.mark.parametrize("click", [True, False])
def test_click_carries_on_through_the_take_only_when_asked(monkeypatch, click):
    played = {}

    def play_and_record(track, sr, mic, speaker, on_tick=None):
        played["track"] = track
        return np.zeros(track.size), 0

    monkeypatch.setattr(audio_io, "_play_and_record", play_and_record)
    audio_io.record_take(120, 1, click=click)
    _, take_start = audio_io.count_in_track(120, 4)
    during = played["track"][take_start:]
    assert (np.max(np.abs(during)) > 0.1) == click


def test_record_take_reports_progress(monkeypatch):
    def play_and_record(track, sr, mic, speaker, on_tick=None):
        for t in (0.0, 1.0, 2.1, 3.3):
            on_tick(t)
        return np.zeros(track.size), 0

    monkeypatch.setattr(audio_io, "_play_and_record", play_and_record)
    seen = []
    audio_io.record_take(120, 2, on_progress=seen.append)
    assert [p.counting_in for p in seen] == [True, True, False, False]
    assert seen[-1].bar == 1 and seen[-1].beat == 3


def test_page_draws_progress_while_recording(monkeypatch):
    def recording(bpm, bars, click, on_progress):
        for t in (0.2, 2.5, 4.0):
            on_progress(audio_io.progress_at(t, bpm, bars))
        return take(100), 0.05

    monkeypatch.setattr(audio_io, "record_take", recording)
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="record").click().run()
    assert not page.exception
    assert page.metric[0].value == "100"


def test_demo_take_then_demo_hum_gives_a_bassline():
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="demo").click().run()
    page.button(key="demo_hum").click().run()
    assert not page.exception
    assert page.session_state["audio_path"] == str(session.BASS_PATH)
    assert "minor pentatonic" in page.session_state["bass_text"]
    assert load_wav(session.BASS_PATH).size == load_wav(session.LOOP_PATH).size


@pytest.mark.parametrize("bpm, steps", [(100, 32), (137, 16), (82, 64)])
def test_demo_hum_fits_any_loop(bpm, steps):
    from hearyabeat.bass import hum_to_notes
    from hearyabeat.fake_voice import demo_hum

    root, notes = hum_to_notes(demo_hum(bpm, steps), bpm, steps)
    assert root == 9 and len(notes) == 5 * (steps // 16)


def test_cli_demo_hum_file_adds_a_bassline_to_the_demo_take(capsys):
    root = Path(__file__).resolve().parent.parent
    assert cli.main(["record", "--input", str(root / "examples/demo_take.wav"), "--no-play"]) == 0
    assert cli.main(["hum", "--input", str(root / "examples/demo_hum.wav"), "--no-play"]) == 0
    assert "A minor pentatonic" in capsys.readouterr().out


def calibration_takes(folder):
    from hearyabeat.fake_voice import make_repeats

    folder.mkdir()
    for i, label in enumerate(("kick", "snare", "hat")):
        save_wav(folder / f"{label}.wav", make_repeats(label, seed=i))
    return folder


def test_cli_calibrate_reports_how_separable_the_sounds_are(sandbox, capsys):
    folder = calibration_takes(sandbox / "cal")
    assert cli.main(["calibrate", "--name", "me", "--input-dir", str(folder)]) == 0
    printed = capsys.readouterr().out
    assert "Saved" in printed and "well separated" in printed
    assert (session.PROFILES / "me.json").exists()


def test_page_shows_separation_after_saving_a_profile(sandbox):
    from hearyabeat.fake_voice import make_repeats

    page = AppTest.from_file(APP, default_timeout=60)
    page.session_state["cal_takes"] = {label: make_repeats(label, seed=i)
                                       for i, label in enumerate(("kick", "snare", "hat"))}
    page.run()
    page.text_input(key="cal_name").input("alex")
    page.button(key="cal_save").click().run()
    assert not page.exception
    assert any("well separated" in box.value for box in page.success)


def test_cli_grid_marks_unsure_hits(capsys):
    from hearyabeat.pipeline import Result
    from hearyabeat.quantise import GridHit

    result = Result(100, 1, 16, 0.0, 1.0, [], [GridHit(0, "kick", 1.0, 0.95), GridHit(4, "snare", 1.0, 0.2)],
                    np.zeros(10))
    cli.print_grid(result)
    printed = capsys.readouterr().out
    assert "K..." in printed and "?..." in printed and "1 of 2 hits" in printed


def test_page_explains_faded_cells_when_hits_are_unsure(monkeypatch):
    from hearyabeat import pipeline

    unsure = lambda profile: (lambda features: ("kick", 0.1))
    monkeypatch.setattr(pipeline, "scorer_for", unsure)
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="demo").click().run()
    assert not page.exception
    assert any("Faded cells" in caption.value for caption in page.caption)


def test_pipeline_times_each_stage_and_keeps_features():
    from hearyabeat.classify import FEATURE_NAMES
    from hearyabeat.insight import STAGES, timing_summary
    from hearyabeat.pipeline import analyse
    from hearyabeat.render import load_samples

    result = analyse(take(100), load_samples(session.SAMPLES))
    assert tuple(result.timings_ms) == STAGES
    assert all(ms >= 0 for ms in result.timings_ms.values())
    assert result.features.shape == (len(result.hits), len(FEATURE_NAMES))
    rows, total, factor = timing_summary(result)
    assert total == pytest.approx(sum(result.timings_ms.values()))
    assert factor == pytest.approx(total / 1000 / result.take_s)


def test_scatter_has_a_point_per_hit_and_per_calibration_example():
    from hearyabeat.classify import build_profile
    from hearyabeat.fake_voice import make_repeats
    from hearyabeat.insight import scatter_spec
    from hearyabeat.pipeline import analyse
    from hearyabeat.render import load_samples

    colours = {"kick": "#000001", "snare": "#000002", "hat": "#000003"}
    profile = build_profile("me", {l: make_repeats(l, seed=i) for i, l in enumerate(("kick", "snare", "hat"))})
    result = analyse(take(100), load_samples(session.SAMPLES), profile=profile)
    spec = scatter_spec(result, colours, profile)
    kinds = [p["kind"] for p in spec["data"]["values"]]
    assert kinds.count("this take") == len(result.hits) and kinds.count("calibration (me)") == 9
    assert spec["layer"][0]["encoding"]["color"]["scale"]["range"] == list(colours.values())
    assert len(spec["layer"]) == 1  # no threshold lines when a profile decides

    plain = scatter_spec(analyse(take(100), load_samples(session.SAMPLES)), colours)
    assert len(plain["layer"]) == 3  # the hits, plus the centroid and low-ratio thresholds
    assert all(p["kind"] == "this take" for p in plain["data"]["values"])


def test_page_shows_the_scatter_and_timings():
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="demo").click().run()
    assert not page.exception
    assert any("took" in caption.value and "x its length" in caption.value for caption in page.caption)
    assert len(page.table) == 1


def test_warm_up_runs_the_beat_tracker_and_saves_nothing(monkeypatch):
    from hearyabeat import detect

    calls = []
    real = detect._beat_track_bpm
    monkeypatch.setattr(detect, "_beat_track_bpm", lambda y, sr: calls.append(1) or real(y, sr))
    assert session.warm_up() >= 0
    assert calls  # tempo was detected freely, so the compile actually happens
    assert not session.SESSION_PATH.exists() and not session.LOOP_PATH.exists()


def test_timing_note_appears_only_when_the_warm_up_failed(monkeypatch):
    def page_captions():
        page = AppTest.from_file(APP, default_timeout=60).run()
        page.button(key="demo").click().run()
        assert not page.exception
        return " ".join(caption.value for caption in page.caption)

    assert "one-off" not in page_captions()

    def broken():
        raise RuntimeError("no warm-up today")

    monkeypatch.setattr(session, "warm_up", broken)
    import streamlit as st
    st.cache_resource.clear()
    assert "first run after launch includes a few seconds of one-off" in page_captions()
    st.cache_resource.clear()


def test_envelope_is_small_and_keeps_the_peaks():
    from hearyabeat.insight import WAVEFORM_BINS, envelope

    y = np.zeros(SR * 3)
    y[SR] = -0.9  # a single loud sample must survive downsampling
    points = envelope(y, SR)
    assert len(points) == WAVEFORM_BINS
    assert max(p["peak"] for p in points) == 0.9
    assert points[0]["time_s"] == 0 and points[-1]["time_s"] < 3
    assert envelope(np.zeros(0), SR) == [] and len(envelope(np.ones(10), SR)) == 10


def test_waveform_marks_every_hit():
    from hearyabeat.insight import waveform_spec
    from hearyabeat.pipeline import analyse
    from hearyabeat.render import load_samples

    y = take(100)
    result = analyse(y, load_samples(session.SAMPLES))
    spec = waveform_spec(y, SR, result, {"kick": "#1", "snare": "#2", "hat": "#3"})
    marks = spec["layer"][1]["data"]["values"]
    assert [m["time_s"] for m in marks] == [round(h.time, 4) for h in result.hits]


def test_page_draws_the_waveform_and_the_scatter():
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button(key="demo").click().run()
    assert not page.exception
    assert len(page.get("vega_lite_chart")) == 2


def test_theme_colour_is_the_kick_colour():
    import ast
    import tomllib

    root = Path(APP).parent
    config = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    source = ast.parse((root / "app.py").read_text(encoding="utf-8"))
    colours = next(ast.literal_eval(node.value) for node in source.body
                   if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "COLOURS")
    assert config["theme"]["primaryColor"] == colours["kick"]

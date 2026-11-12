"""HearYaBeat web page. Run: streamlit run app.py"""

from pathlib import Path

import streamlit as st

from hearyabeat import LABELS, MAX_BPM, MIN_BPM, audio_io, bass, session
from hearyabeat.bass import HumError
from hearyabeat.classify import CalibrationError, describe_separation, list_profiles, load_profile
from hearyabeat.insight import scatter_spec, timing_summary, waveform_spec
from hearyabeat.detect import TooFewHits
from hearyabeat.fake_voice import demo_hum
from hearyabeat.pipeline import confidence_lanes, unsure_hits
from hearyabeat.render import load_wav

NO_PROFILE = "Default (no calibration)"
DEMO_TAKE = Path(__file__).parent / "examples" / "demo_take.wav"
FRIENDLY_ERRORS = (TooFewHits, HumError, CalibrationError, FileNotFoundError, ValueError,
                   audio_io.AudioDeviceError)
COLOURS = {"kick": "#e4572e", "snare": "#2e86ab", "hat": "#d9a404"}

st.set_page_config(page_title="HearYaBeat", page_icon="🥁", layout="wide")
state = st.session_state
state.setdefault("outcome", None)
state.setdefault("outcome_profile", None)
state.setdefault("audio_path", None)
state.setdefault("bass_text", None)
state.setdefault("cal_takes", {})
state.setdefault("cal_result", None)


@st.cache_resource(show_spinner="Warming up...")
def warm_up():
    """Once per server: pay librosa's one-off compile (a few seconds) now, not on the first real take.
    Returns the ms it took, or None if it failed, in which case the app still works."""
    try:
        return session.warm_up()
    except Exception:  # a failed warm-up only costs speed, so never block the page on it
        return None


warmed_ms = warm_up()


def cell_opacity(confidence):
    """Fade unsure hits, but never so far that they disappear."""
    return 0.25 + 0.75 * confidence


def grid_html(result):
    table = confidence_lanes(result)
    rows = []
    for label in LABELS:
        cells = []
        for step, confidence in enumerate(table[label]):
            if confidence is None:
                colour, opacity, tip = "rgba(128,128,128,.16)", 1.0, ""
            else:
                colour, opacity = COLOURS[label], cell_opacity(confidence)
                tip = f' title="{label}, {confidence:.0%} sure"'
            gap = 14 if step and step % 16 == 0 else 7 if step and step % 4 == 0 else 2
            cells.append(
                f'<div{tip} style="flex:1;min-width:6px;height:30px;border-radius:4px;'
                f'margin-left:{gap}px;background:{colour};opacity:{opacity:.2f}"></div>'
            )
        rows.append(
            '<div style="display:flex;align-items:center;margin-bottom:4px">'
            f'<div style="width:52px;font-size:14px">{label}</div>'
            f'<div style="display:flex;flex:1">{"".join(cells)}</div></div>'
        )
    return "".join(rows)


def beat_html(progress, verb):
    """A big readout of where the performer is: four beat dots and the bar count."""
    if progress.counting_in:
        heading, active, colour = f"Get ready... {progress.count_left}", progress.beat, "#888"
    else:
        heading, active, colour = f"{verb}: bar {progress.bar} of {progress.bars}", progress.beat, "#e4572e"
    dots = "".join(
        f'<div style="width:46px;height:46px;border-radius:50%;margin:0 8px;'
        f'background:{colour if i + 1 == active else "rgba(128,128,128,.22)"};'
        f'transform:scale({1.15 if i + 1 == active else 1})"></div>'
        for i in range(4)
    )
    return (
        f'<div style="text-align:center;padding:12px 0">'
        f'<div style="font-size:26px;font-weight:700;margin-bottom:12px">{heading}</div>'
        f'<div style="display:flex;justify-content:center">{dots}</div></div>'
    )


class LiveProgress:
    """Updates the beat display and progress bar, only when something changes."""

    def __init__(self, verb):
        self.verb = verb
        self.display = st.empty()
        self.bar = st.progress(0.0)
        self.last = None

    def __call__(self, progress):
        key = (progress.counting_in, progress.bar, progress.beat, round(progress.fraction, 2))
        if key == self.last:
            return
        self.last = key
        self.display.html(beat_html(progress, self.verb))
        self.bar.progress(progress.fraction)

    def clear(self):
        self.display.empty()
        self.bar.empty()


def accept(outcome, profile_name):
    state.outcome = outcome
    state.outcome_profile = profile_name  # the sidebar can change later; the scatter needs this one
    state.audio_path = str(outcome.path)
    if not outcome.bass_kept:
        state.bass_text = None
    if outcome.bass_cleared:
        st.toast("Cleared the old bassline, since the tempo or length changed.")


st.title("HearYaBeat")
st.write("Beatbox a rhythm, get a clean drum loop back, then hum a bassline over it.")

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Whose voice?", [NO_PROFILE, *list_profiles(session.PROFILES)])
    profile = None if choice == NO_PROFILE else choice
    bpm = st.slider("Count-in BPM", MIN_BPM, MAX_BPM, 100)
    bars = st.select_slider("Bars to record", [1, 2, 4], value=2)
    click = st.toggle("Click while I beatbox", value=True,
                      help="Keeps you in time, and locks the loop to this BPM. Needs headphones. "
                           "Turn it off on speakers, and the tempo will be detected instead.")
    st.caption("Use headphones, so the mic doesn't pick up the clicks and the loop.")

left, right = st.columns(2)
try:
    if left.button("🎙️ Record a beat", type="primary", key="record", width="stretch"):
        live = LiveProgress("Beatbox")
        take, downbeat = audio_io.record_take(bpm, bars, click=click, on_progress=live)
        live.clear()
        with st.spinner("Listening back..."):
            accept(session.process_take(take, hint_bpm=bpm, downbeat=downbeat, profile_name=profile,
                                        lock_bpm=click), profile)
    if right.button("Try the demo take (no mic)", key="demo", width="stretch"):
        accept(session.process_take(load_wav(DEMO_TAKE), profile_name=profile), profile)
except FRIENDLY_ERRORS as error:
    st.error(str(error))

outcome = state.outcome
if outcome is not None:
    result = outcome.result
    bpm_col, bars_col, fit_col = st.columns(3)
    bpm_col.metric("Detected BPM", result.bpm)
    bars_col.metric("Bars", result.bars)
    fit_col.metric("Timing fit", f"{result.confidence:.0%}")
    if result.confidence < session.LOOSE_TIMING:
        st.warning("The timing was loose, so this grid is a best guess. "
                   "Try again with the click on and headphones.")
    if session.TAKE_PATH.exists():  # process_take saves the take this result came from
        st.vega_lite_chart(spec=waveform_spec(load_wav(session.TAKE_PATH), result.sr, result, COLOURS),
                           width="stretch")
    st.html(grid_html(result))
    st.html(" ".join(
        f'<span style="color:{COLOURS[label]};font-weight:600;margin-right:14px">■ {label}</span>'
        for label in LABELS
    ))
    unsure = unsure_hits(result)
    if unsure:
        st.caption(f"Faded cells are hits the app wasn't sure about ({len(unsure)} of {len(result.grid)}). "
                   "Hover over a cell to see how sure it was. Calibrating for your voice usually helps.")

    with st.expander("Under the hood"):
        used = load_profile(state.outcome_profile, session.PROFILES) if state.outcome_profile else None
        st.caption("Each point is a hit, placed by its brightness and its low-end energy, and faded by "
                   "how sure the app was. This shows 2 of the 5 features, so points that look close here "
                   "can still be far apart to the classifier."
                   + (" Diamonds are your calibration examples." if used else
                      " Dashed lines are the default thresholds on these two features."))
        st.vega_lite_chart(spec=scatter_spec(result, COLOURS, used), width="stretch")
        rows, total, factor = timing_summary(result)
        st.table({"stage": [stage for stage, _ in rows] + ["total"],
                  "time (ms)": [f"{ms:.1f}" for _, ms in rows] + [f"{total:.1f}"]})
        st.caption(f"Analysing a {result.take_s:.1f} s take took {total:.0f} ms, "
                   f"{factor:.2f}x its length. Under 1x is faster than the take itself, "
                   "which is what running live would need."
                   + ("" if warmed_ms is not None else
                      " The warm-up didn't run, so the first run after launch includes a few seconds of one-off "
                      "compile in librosa."))

    data = Path(state.audio_path).read_bytes()
    st.audio(data, format="audio/wav", loop=True)
    st.download_button("Download the loop", data, file_name="hearyabeat_loop.wav", mime="audio/wav")

    st.subheader("Add a bassline")
    if state.bass_text:
        st.write(state.bass_text)
    hum_col, demo_hum_col = st.columns(2)
    try:
        saved = session.load_session(session.SESSION_PATH)
        hum = None
        if hum_col.button("🎵 Hum over the loop", key="hum", width="stretch"):
            live = LiveProgress("Hum")
            hum = audio_io.record_over_loop(load_wav(session.LOOP_PATH), saved["bpm"], saved["n_steps"],
                                            on_progress=live)
            live.clear()
        if demo_hum_col.button("Try a demo hum (no mic)", key="demo_hum", width="stretch"):
            hum = demo_hum(saved["bpm"], saved["n_steps"])
        if hum is not None:
            with st.spinner("Finding the notes..."):
                root, notes, path = session.add_hum(hum, saved)
            state.audio_path = str(path)
            state.bass_text = f"{root} minor pentatonic: " + " ".join(bass.note_label(n[2]) for n in notes)
            st.rerun()
    except FRIENDLY_ERRORS as error:
        st.error(str(error))

with st.expander("Calibrate for a new voice"):
    st.write(
        f"Teach it your own sounds. For each one, press the button and make that sound "
        f"{session.HITS_PER_SOUND} times, with a gap between each."
    )
    name = st.text_input("Your name", key="cal_name", placeholder="alex")
    columns = st.columns(len(LABELS))
    for column, label in zip(columns, LABELS):
        done = label in state.cal_takes
        if column.button(f"{'✅' if done else '🎙️'} {label.title()} x{session.HITS_PER_SOUND}", key=f"cal_{label}"):
            try:
                with st.spinner(f"Recording {label}s..."):
                    state.cal_takes[label] = session.calibrate_take()
                st.rerun()
            except FRIENDLY_ERRORS as error:
                st.error(f"{error} Press the button to try that one again.")
    if state.cal_result:
        saved_name, loo = state.cal_result
        message = f"Saved profile {saved_name}. {describe_separation(loo)}"
        (st.success if not loo.mistakes else st.warning)(message)
    if len(state.cal_takes) == len(LABELS):
        if st.button("Save profile", key="cal_save"):
            try:
                _, loo = session.finish_calibration(name, state.cal_takes)
                state.cal_takes = {}
                state.cal_result = (name, loo)
                st.rerun()
            except FRIENDLY_ERRORS as error:
                st.error(str(error))

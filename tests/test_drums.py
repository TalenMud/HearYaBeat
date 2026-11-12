from pathlib import Path

import numpy as np
import pytest

from hearyabeat import SR, samples_gen
from hearyabeat.classify import (
    CalibrationError,
    build_profile,
    classify_by_thresholds,
    extract_features,
    load_profile,
    save_profile,
    validate_name,
)
from hearyabeat.detect import TooFewHits, detect_onsets, estimate_grid
from hearyabeat.fake_voice import BASIC_BAR, VOICES, make_repeats, make_take
from hearyabeat.pipeline import analyse, lanes
from hearyabeat.quantise import Hit, quantise
from hearyabeat.render import load_wav, loop_samples, render_loop, step_position


@pytest.fixture(scope="module")
def kit():
    return {"kick": samples_gen.kick(), "snare": samples_gen.snare(), "hat": samples_gen.hat()}


def expected_steps(bars=2):
    return {(bar * 16 + step, label) for bar in range(bars) for step, label in BASIC_BAR}


def test_finds_every_onset_within_20ms():
    y, truth = make_take(BASIC_BAR, 100, jitter_s=0.005, seed=1)
    times, _ = detect_onsets(y)
    assert len(times) == len(truth)
    assert np.max(np.abs(np.sort(times) - np.sort(truth))) < 0.02


def test_silence_has_no_hits():
    times, _ = detect_onsets(np.random.default_rng(0).standard_normal(SR * 2) * 1e-5)
    assert len(times) == 0


@pytest.mark.parametrize("bpm", [75, 90, 100, 120, 140, 170])
@pytest.mark.parametrize("seed", [1, 2])
def test_bpm_with_hint_and_timing_slop(bpm, seed):
    y, _ = make_take(BASIC_BAR, bpm, jitter_s=0.008, seed=seed)
    times, _ = detect_onsets(y)
    grid = estimate_grid(times, y, hint_bpm=bpm, downbeat=0.05)
    assert abs(grid.bpm - bpm) <= 2


@pytest.mark.parametrize("bpm", [85, 100, 125, 150])
def test_bpm_without_hint(bpm):
    y, _ = make_take(BASIC_BAR, bpm, jitter_s=0.005, seed=3)
    times, _ = detect_onsets(y)
    assert abs(estimate_grid(times, y).bpm - bpm) <= 2


def test_performer_drifting_from_count_in_still_detected():
    y, _ = make_take(BASIC_BAR, 112, jitter_s=0.005, seed=4)
    times, _ = detect_onsets(y)
    assert abs(estimate_grid(times, y, hint_bpm=100).bpm - 112) <= 2


def test_too_few_hits_raises():
    with pytest.raises(TooFewHits):
        estimate_grid(np.array([0.1]), np.zeros(SR))


@pytest.mark.parametrize("label", ["kick", "snare", "hat"])
def test_thresholds_label_each_sound(label):
    rng = np.random.default_rng(0)
    y = np.concatenate([VOICES[label](rng), np.zeros(SR // 2)])
    assert classify_by_thresholds(extract_features(y, 0.0)) == label


@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_whole_pipeline_gets_the_pattern_right(kit, bpm):
    y, _ = make_take(BASIC_BAR, bpm, jitter_s=0.008, seed=5)
    result = analyse(y, kit, hint_bpm=bpm, downbeat=0.05)
    assert result.bars == 2
    got = {(hit.step, hit.label) for hit in result.grid}
    wanted = expected_steps()
    assert len(got & wanted) / len(wanted) >= 0.9
    assert sum(lanes(result)["snare"]) >= 3


def test_rest_before_first_hit_is_kept_with_a_count_in(kit):
    pattern = [(s + 4, label) for s, label in BASIC_BAR if s + 4 < 16]
    y, _ = make_take(pattern, 100, jitter_s=0.0, seed=2)
    result = analyse(y, kit, hint_bpm=100, downbeat=0.05)
    assert min(hit.step for hit in result.grid) == 4


def test_loop_length_is_exact(kit):
    for bpm, steps in ((97, 32), (133, 16), (171, 64)):
        assert render_loop([], bpm, steps, kit).size == round(steps * 60 / bpm / 4 * SR)


def test_tails_wrap_so_the_loop_is_seamless(kit):
    from hearyabeat.quantise import GridHit

    bpm, steps = 120, 16
    last = GridHit(15, "kick", 1.0)
    loop = render_loop([last], bpm, steps, kit)
    length = loop_samples(bpm, steps)
    overhang = step_position(15, bpm) + kit["kick"].size - length
    assert overhang > 0
    assert np.max(np.abs(loop[:overhang])) > 0.01
    # The audio crossing the join is continuous.
    assert abs(loop[-1] - loop[0]) < 0.2


def test_collision_keeps_the_strongest_hit():
    hits = [Hit(0.0, "hat", 0.1), Hit(0.01, "kick", 0.9)]
    grid = quantise(hits, 120, 0.0, 16)
    assert [(g.step, g.label) for g in grid] == [(0, "kick")]


def test_hit_at_end_wraps_to_step_zero():
    grid = quantise([Hit(2.0, "kick", 1.0)], 120, 0.0, 16)
    assert [g.step for g in grid] == [0]


def different_voice():
    """Someone whose sounds the default thresholds get wrong: a bright, thin
    kick, and a dark hat."""

    def kick(rng, pitch, sr):
        t = np.arange(int(0.09 * sr)) / sr
        return (np.sin(2 * np.pi * 1100 * t) + 0.5 * np.sin(2 * np.pi * 2200 * t)) * np.exp(-t / 0.02)

    def hat(rng, pitch, sr):
        from scipy.signal import butter, sosfilt

        t = np.arange(int(0.05 * sr)) / sr
        sos = butter(3, [2500, 4200], "bandpass", fs=sr, output="sos")
        return sosfilt(sos, rng.standard_normal(t.size)) * np.exp(-t / 0.012) * 3

    def snare(rng, pitch, sr):
        return VOICES["snare"](rng, 0.5, sr)

    return {"kick": kick, "snare": snare, "hat": hat}


def test_calibration_fixes_a_voice_the_thresholds_get_wrong(kit):
    shapes = different_voice()
    takes = {label: make_repeats(label, seed=i, shapes=shapes) for i, label in enumerate(shapes)}
    profile = build_profile("other", takes)
    y, _ = make_take(BASIC_BAR, 110, seed=9, shapes=shapes)

    wanted = expected_steps()
    plain = analyse(y, kit, hint_bpm=110, downbeat=0.05)
    tuned = analyse(y, kit, hint_bpm=110, downbeat=0.05, profile=profile)
    score = lambda r: len({(h.step, h.label) for h in r.grid} & wanted) / len(wanted)
    assert score(tuned) >= 0.9
    assert score(tuned) > score(plain)


def test_calibration_rejects_the_wrong_hit_count():
    takes = {label: make_repeats(label, count=2) for label in VOICES}
    with pytest.raises(CalibrationError):
        build_profile("x", takes)


def test_profile_round_trip(tmp_path):
    takes = {label: make_repeats(label, seed=i) for i, label in enumerate(VOICES)}
    profile = build_profile("me", takes)
    save_profile(profile, tmp_path)
    loaded = load_profile("me", tmp_path)
    assert np.allclose(loaded.examples, profile.examples)
    assert loaded.labels == profile.labels


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "a b"])
def test_profile_names_cannot_escape_the_folder(bad):
    with pytest.raises(ValueError):
        validate_name(bad)


def test_click_locks_the_tempo_even_when_timing_is_loose(kit):
    # Sloppy enough that a free tempo search can't be trusted.
    y, _ = make_take(BASIC_BAR, 100, jitter_s=0.02, seed=11)
    result = analyse(y, kit, hint_bpm=100, downbeat=0.05, lock_bpm=True)
    assert result.bpm == 100 and result.bars == 2
    got = {(hit.step, hit.label) for hit in result.grid}
    assert len(got & expected_steps()) / len(expected_steps()) >= 0.75


def test_a_split_ts_counts_as_one_hit():
    rng = np.random.default_rng(0)
    y = np.random.default_rng(1).standard_normal(SR) * 0.0005
    for start in (0.3, 0.6):
        for offset in (0.0, 0.025):  # "t" then "s", 25 ms apart
            sound = VOICES["hat"](rng)
            i = int((start + offset) * SR)
            y[i : i + sound.size] += sound * 0.5
    times, _ = detect_onsets(y)
    assert len(times) == 2


def test_breaths_far_below_the_real_hits_are_ignored():
    rng = np.random.default_rng(0)
    y = np.random.default_rng(1).standard_normal(SR * 2) * 0.0005
    for i, start in enumerate((0.2, 0.5, 0.8, 1.1, 1.4)):
        sound = VOICES["kick"](rng) * 0.8
        y[int(start * SR) : int(start * SR) + sound.size] += sound
        if i < 4:  # a quiet puff between each pair of kicks
            puff = VOICES["snare"](rng) * 0.04
            p = int((start + 0.15) * SR)
            y[p : p + puff.size] += puff
    times, _ = detect_onsets(y)
    assert len(times) == 5


def profile_from_points(points_by_label):
    """A Profile from hand-placed points, varying only the first feature."""
    from hearyabeat.classify import FEATURE_NAMES, Profile

    examples, labels = [], []
    for label, points in points_by_label.items():
        for x in points:
            vector = np.zeros(len(FEATURE_NAMES))
            vector[0] = x
            examples.append(vector)
            labels.append(label)
    return Profile("test", np.array(examples), labels)


def test_leave_one_out_flags_two_sounds_that_overlap():
    from hearyabeat.classify import describe_separation

    # Kicks and snares interleave, so a held-out kick's nearest neighbours are
    # snares, and the other way round. Hats sit far away.
    profile = profile_from_points({"kick": [0, 2, 4], "snare": [1, 3, 5], "hat": [100, 101, 102]})
    loo = profile.leave_one_out()
    assert loo.accuracy < 1
    assert {pair for pair, _ in loo.overlapping()} == {("kick", "snare")}
    assert all("hat" not in pair for pair in loo.mistakes)
    message = describe_separation(loo)
    assert "kick and snare sound alike" in message and "calibrate again" in message
    assert "about 33%, a cautious estimate from 9 examples" in message


def test_leave_one_out_is_perfect_for_well_separated_sounds():
    from hearyabeat.classify import describe_separation

    profile = profile_from_points({"kick": [0, 0.5, 1], "snare": [50, 50.5, 51], "hat": [100, 100.5, 101]})
    loo = profile.leave_one_out()
    assert loo.accuracy == 1 and loo.mistakes == {} and loo.n_examples == 9
    message = describe_separation(loo)
    assert "well separated" in message and "about 100%, a cautious estimate from 9 examples" in message


def test_leave_one_out_never_changes_the_profile():
    profile = profile_from_points({"kick": [0, 2, 4], "snare": [1, 3, 5], "hat": [100, 101, 102]})
    before = profile.examples.copy()
    profile.leave_one_out()
    assert np.array_equal(profile.examples, before) and len(profile.labels) == 9


def test_calibrating_the_synthetic_voice_separates_all_three_sounds():
    takes = {label: make_repeats(label, seed=i) for i, label in enumerate(VOICES)}
    assert build_profile("me", takes).leave_one_out().accuracy == 1


def test_threshold_confidence_never_changes_the_label_and_agrees_with_the_rules():
    from hearyabeat.classify import _threshold_scores, classify_by_thresholds_with_confidence

    rng = np.random.default_rng(0)
    for _ in range(2000):
        features = np.array([rng.uniform(7, 14), rng.uniform(0, 0.5), rng.uniform(0, 1),
                             rng.uniform(0, 1), rng.uniform(3, 8)])
        label, confidence = classify_by_thresholds_with_confidence(features)
        assert label == classify_by_thresholds(features) and 0 <= confidence <= 1
        # The signed scores imply the same label, so the margins measure the real boundaries.
        kick, hat = _threshold_scores(features)
        assert label == ("kick" if kick > 0 else "hat" if hat > 0 else "snare")


def test_threshold_confidence_is_low_at_a_boundary_and_high_far_from_it():
    from hearyabeat.classify import KICK_CENTROID_HZ, classify_by_thresholds_with_confidence

    def at_centroid(hz):
        return np.array([np.log2(hz + 1), 0.1, 0.1, 0.3, np.log2(120)])

    assert classify_by_thresholds_with_confidence(at_centroid(KICK_CENTROID_HZ * 1.02))[1] < 0.1
    assert classify_by_thresholds_with_confidence(at_centroid(300)) == ("kick", 1.0)
    assert classify_by_thresholds_with_confidence(at_centroid(2500))[1] > 0.9  # a clear snare


def test_knn_confidence_is_high_inside_a_cluster_and_low_between_two():
    profile = profile_from_points({"kick": [0, 0.5, 1], "snare": [10, 10.5, 11], "hat": [100, 100.5, 101]})

    def at(x):
        vector = np.zeros(5)
        vector[0] = x
        return vector

    from hearyabeat.classify import UNSURE

    assert profile.classify_with_confidence(at(0.5))[1] > 0.9
    # The nearest kick is at 1 and the nearest snare at 10, so 5.5 is the middle.
    for x in (5.3, 5.5, 5.7):
        label, confidence = profile.classify_with_confidence(at(x))
        assert label == profile.classify(at(x)) and confidence < UNSURE
    # And confidence falls steadily from each cluster towards the middle.
    kick_side = [profile.classify_with_confidence(at(x))[1] for x in (1, 2, 3, 4, 5, 5.5)]
    snare_side = [profile.classify_with_confidence(at(x))[1] for x in (10, 9, 8, 7, 6, 5.7)]
    assert kick_side == sorted(kick_side, reverse=True) and snare_side == sorted(snare_side, reverse=True)


def test_knn_confidence_sees_a_close_rival_outside_the_top_three():
    # The 3 nearest are all kicks, so the plain vote is unanimous, but a snare
    # is nearly as close. That must not read as fully confident.
    profile = profile_from_points({"kick": [0, 0.2, 0.4], "snare": [2.2, 5, 6], "hat": [100, 101, 102]})
    query = np.zeros(5)
    query[0] = 1.2
    label, confidence = profile.classify_with_confidence(query)
    assert label == "kick" and confidence < 0.5


def test_knn_confidence_never_changes_the_label():
    profile = profile_from_points({"kick": [0, 2, 4], "snare": [1, 3, 5], "hat": [100, 101, 102]})
    for x in np.linspace(-5, 110, 300):
        query = np.zeros(5)
        query[0] = x
        label, confidence = profile.classify_with_confidence(query)
        assert label == profile.classify(query) and 0 <= confidence <= 1


def test_the_known_mislabel_on_the_demo_take_has_low_confidence(kit):
    """Regression: the synthetic profile mislabels one demo snare as a hat. Before
    the rival-class voters it did so at 100% confidence.

    The margins are deliberately loose, so that small feature changes don't
    flip the test: the mislabel only has to stay under 0.6, and below the
    median confidence of the hits it got right.
    """
    from hearyabeat.evaluate import load_truth, match

    root = Path(__file__).resolve().parent.parent / "examples"
    y = load_wav(root / "demo_take.wav")
    truth_times, truth_labels = load_truth(root / "demo_take.csv")
    profile = build_profile("me", {label: make_repeats(label, seed=i) for i, label in enumerate(VOICES)})
    result = analyse(y, kit, profile=profile)
    pairs = match([h.time for h in result.hits], truth_times)
    wrong = [(result.hits[d], truth_labels[t]) for d, t in pairs if result.hits[d].label != truth_labels[t]]
    right = [result.hits[d].confidence for d, t in pairs if result.hits[d].label == truth_labels[t]]
    assert [(hit.label, truth) for hit, truth in wrong] == [("hat", "snare")]
    mislabel = wrong[0][0].confidence
    assert mislabel < 0.6
    assert np.median(right) > mislabel


def test_grid_cells_carry_each_hit_confidence(kit):
    from hearyabeat.pipeline import confidence_lanes

    y = load_wav(Path(__file__).resolve().parent.parent / "examples" / "demo_take.wav")
    result = analyse(y, kit)
    table = confidence_lanes(result)
    for hit in result.grid:
        assert table[hit.label][hit.step] == hit.confidence
    assert sum(value is not None for row in table.values() for value in row) == len(result.grid)

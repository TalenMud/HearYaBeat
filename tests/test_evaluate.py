from pathlib import Path

import numpy as np
import pytest

import cli
from hearyabeat.evaluate import evaluate_take, load_truth, match, save_truth, score
from hearyabeat.render import load_wav

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_demo_take_scores_perfectly():
    times, labels = load_truth(EXAMPLES / "demo_take.csv")
    report = evaluate_take(load_wav(EXAMPLES / "demo_take.wav"), times, labels)
    assert (report.n_truth, report.n_detected, report.n_matched) == (16, 16, 16)
    assert report.precision == report.recall == report.f1 == 1.0
    # Synthetic truth is exact, so this checks the backtrack lands on the attack.
    assert report.median_error_ms < 2 and report.p95_error_ms < 5
    assert report.accuracy == 1.0
    assert np.array_equal(report.confusion, np.diag([6, 4, 6]))


def test_matching_takes_the_closest_pairs_first():
    # Detection 0 is near both labels; it belongs to the closer one, leaving
    # detection 1 for the other.
    assert match([1.00, 1.04], [1.03, 0.99]) == [(1, 0), (0, 1)]


def test_matching_respects_the_tolerance_and_uses_each_hit_once():
    assert match([1.0], [1.06]) == []
    assert match([1.0], [1.05]) == [(0, 0)]
    assert match([1.0, 1.01], [1.0]) == [(0, 0)]


def test_extra_and_missed_hits_lower_precision_and_recall():
    report = score([0.5, 1.0, 1.5, 2.0], ["kick"] * 4, np.array([1.0, 2.0, 3.0]), ["kick"] * 3)
    assert report.n_matched == 2
    assert report.precision == 0.5
    assert report.recall == pytest.approx(2 / 3)
    assert report.f1 == pytest.approx(4 / 7)


def test_confusion_rows_are_truth_and_columns_are_predictions():
    report = score([1.0, 2.0, 3.0], ["snare", "snare", "hat"],
                   np.array([1.0, 2.0, 3.0]), ["kick", "snare", "hat"])
    assert report.confusion[0, 1] == 1  # a kick heard as a snare
    assert report.class_accuracy["kick"] == 0 and report.class_accuracy["snare"] == 1
    assert report.accuracy == pytest.approx(2 / 3)


def test_timing_error_and_bias_are_reported_in_ms():
    report = score([0.99, 1.98, 3.0], ["kick"] * 3, np.array([1.0, 2.0, 3.0]), ["kick"] * 3)
    assert report.median_error_ms == pytest.approx(10)
    assert report.median_bias_ms == pytest.approx(-10)


def test_nothing_detected_reports_zeros_not_a_crash():
    report = score([], [], np.array([1.0]), ["kick"])
    assert report.recall == report.precision == report.f1 == 0
    assert np.isnan(report.median_error_ms) and np.isnan(report.accuracy)
    assert np.isnan(report.class_accuracy["kick"])


def test_labels_round_trip_and_sort(tmp_path):
    path = save_truth(tmp_path / "t.csv", [2.0, 1.0], ["hat", "kick"])
    times, labels = load_truth(path)
    assert list(times) == [1.0, 2.0] and labels == ["kick", "hat"]


@pytest.mark.parametrize("body", ["when,what\n1,kick\n", "time_s,label\n1,cowbell\n", "time_s,label\nsoon,kick\n"])
def test_bad_label_files_explain_themselves(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth(path)


def test_cli_evaluate_finds_the_labels_next_to_the_wav(capsys):
    assert cli.main(["evaluate", str(EXAMPLES / "demo_take.wav")]) == 0
    printed = capsys.readouterr().out
    assert "16 matched" in printed and "F1 1.00" in printed and "Confusion" in printed


def test_cli_evaluate_without_labels_gives_a_message(tmp_path, capsys):
    wav = tmp_path / "take.wav"
    wav.write_bytes((EXAMPLES / "demo_take.wav").read_bytes())
    assert cli.main(["evaluate", str(wav)]) == 1
    assert "No labels" in capsys.readouterr().err


def test_report_also_scores_at_25ms():
    # Two detections 30 ms off: found at 50 ms, missed at 25 ms.
    report = score([1.03, 2.0], ["kick"] * 2, np.array([1.0, 2.0]), ["kick"] * 2)
    assert report.n_matched == 2 and report.strict.n_matched == 1
    assert report.strict.recall == 0.5 and report.strict.strict is None


def test_no_strict_line_when_already_strict():
    assert score([1.0], ["kick"], np.array([1.0]), ["kick"], tolerance=0.02).strict is None


def test_cli_prints_both_tolerances(capsys):
    assert cli.main(["evaluate", str(EXAMPLES / "demo_take.wav")]) == 0
    printed = capsys.readouterr().out
    assert "within 50 ms: 16 matched" in printed and "within 25 ms: 16 matched" in printed


def test_confidence_splits_accuracy_at_the_cut_off_and_fills_the_buckets():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    truth = ["kick", "kick", "snare", "hat", "hat"]
    guesses = ["kick", "snare", "snare", "hat", "kick"]  # hits 2 and 5 are wrong
    confidences = [0.95, 0.1, 0.6, 1.0, 0.3]
    report = score(times, guesses, times, truth, detected_confidences=confidences)
    assert report.sure == (3, 1.0)
    assert report.unsure == (2, 0.0)
    assert [(low, high, count) for low, high, count, _ in report.reliability] == [
        (0.0, 0.25, 1), (0.25, 0.5, 1), (0.5, 0.75, 1), (0.75, 1.0, 2)]  # 1.0 lands in the top bucket
    assert np.isnan(score(times, guesses, times, truth).sure[1])  # no confidences, no table


def test_empty_buckets_report_no_accuracy():
    report = score([1.0], ["kick"], np.array([1.0]), ["kick"], detected_confidences=[0.9])
    counts = [(count, accuracy) for _, _, count, accuracy in report.reliability]
    assert counts[3] == (1, 1.0) and counts[0][0] == 0 and np.isnan(counts[0][1])


def test_cli_prints_the_reliability_table_marked_uncalibrated(capsys):
    assert cli.main(["evaluate", str(EXAMPLES / "demo_take.wav")]) == 0
    printed = capsys.readouterr().out
    assert "UNCALIBRATED" in printed and "sure   (>= 50%)" in printed and "75%-100%" in printed

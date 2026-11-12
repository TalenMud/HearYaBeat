"""Measure detection and classification against a hand-labelled take.

The tests only use synthetic voices, so they prove the logic. This is how the
app is measured on real recordings: label a take's hits by hand in a CSV with
columns time_s,label, then compare.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import LABELS, SR
from .classify import UNSURE, extract_features, scorer_for
from .detect import detect_onsets

TOLERANCE_S = 0.05  # a detection this close to a labelled hit counts as finding it
STRICT_TOLERANCE_S = 0.025  # also reported, as a tighter check on timing
CONFIDENCE_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Report:
    n_truth: int
    n_detected: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    median_error_ms: float  # of |detected - labelled|, over matched hits; NaN if none
    p95_error_ms: float
    median_bias_ms: float  # signed: negative means detections land early
    class_accuracy: dict  # label -> share of matched hits labelled right, NaN if none
    confusion: np.ndarray  # rows are the labelled class, columns the predicted, in LABELS order
    tolerance_s: float = TOLERANCE_S
    strict: "Report | None" = None  # the same take scored at STRICT_TOLERANCE_S
    # Matched hits split at UNSURE, as (count, accuracy), and the reliability
    # table as [(low, high, count, accuracy)]. Empty when no confidences given.
    sure: tuple = (0, float("nan"))
    unsure: tuple = (0, float("nan"))
    reliability: list = field(default_factory=list)

    @property
    def accuracy(self):
        """Classification accuracy over all matched hits."""
        return float(np.trace(self.confusion) / self.n_matched) if self.n_matched else float("nan")


def load_truth(path):
    """Read a CSV with a time_s,label header. Returns (times, labels), sorted by time."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"time_s", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path} needs a header row: time_s,label")
        for line, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(f"{path} line {line}: label must be one of {', '.join(LABELS)}, not {label!r}")
            try:
                rows.append((float(row["time_s"]), label))
            except ValueError:
                raise ValueError(f"{path} line {line}: time_s must be a number of seconds") from None
    rows.sort()
    return np.array([t for t, _ in rows]), [label for _, label in rows]


def save_truth(path, times, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "label"])
        for t, label in sorted(zip(times, labels)):
            writer.writerow([f"{t:.4f}", label])
    return path


def match(detected, truth, tolerance=TOLERANCE_S):
    """Pair detections with labelled hits, closest pairs first.

    Each detection and each labelled hit is used at most once, and only pairs
    within tolerance count. Returns [(detected index, truth index)].
    """
    detected, truth = np.asarray(detected, dtype=float), np.asarray(truth, dtype=float)
    if detected.size == 0 or truth.size == 0:
        return []
    gaps = np.abs(detected[:, None] - truth[None, :])
    # The tiny allowance keeps a gap of exactly the tolerance inside it, since
    # 1.05 - 1.0 comes out just over 0.05 in floating point.
    candidates = np.argwhere(gaps <= tolerance + 1e-9)
    order = np.argsort(gaps[candidates[:, 0], candidates[:, 1]], kind="stable")
    used_d, used_t, pairs = set(), set(), []
    for d, t in candidates[order]:
        if d not in used_d and t not in used_t:
            used_d.add(int(d))
            used_t.add(int(t))
            pairs.append((int(d), int(t)))
    return sorted(pairs, key=lambda pair: pair[1])


def _ratio(top, bottom):
    return top / bottom if bottom else 0.0


def _accuracy_of(outcomes):
    return (len(outcomes), float(np.mean(outcomes)) if outcomes else float("nan"))


def _reliability(confidences, right):
    """Accuracy per confidence bucket. The top bucket includes 1.0."""
    table = []
    for low, high in zip(CONFIDENCE_BUCKETS[:-1], CONFIDENCE_BUCKETS[1:]):
        inside = [r for c, r in zip(confidences, right)
                  if low <= c < high or (high == CONFIDENCE_BUCKETS[-1] and c == high)]
        table.append((low, high, *_accuracy_of(inside)))
    return table


def score(detected_times, detected_labels, truth_times, truth_labels, tolerance=TOLERANCE_S,
          detected_confidences=None):
    """Build a Report from detections and labels already in hand."""
    pairs = match(detected_times, truth_times, tolerance)
    n_matched = len(pairs)
    precision = _ratio(n_matched, len(detected_times))
    recall = _ratio(n_matched, len(truth_times))
    f1 = _ratio(2 * precision * recall, precision + recall)

    errors = np.array([detected_times[d] - truth_times[t] for d, t in pairs]) * 1000
    if errors.size:
        median, p95, bias = (float(np.median(np.abs(errors))), float(np.percentile(np.abs(errors), 95)),
                             float(np.median(errors)))
    else:
        median = p95 = bias = float("nan")

    index = {label: i for i, label in enumerate(LABELS)}
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for d, t in pairs:
        confusion[index[truth_labels[t]], index[detected_labels[d]]] += 1
    class_accuracy = {
        label: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else float("nan")
        for i, label in enumerate(LABELS)
    }
    strict = None
    if tolerance > STRICT_TOLERANCE_S:
        strict = score(detected_times, detected_labels, truth_times, truth_labels, STRICT_TOLERANCE_S)
    report = Report(len(truth_times), len(detected_times), n_matched, precision, recall, f1,
                    median, p95, bias, class_accuracy, confusion, tolerance, strict)
    if detected_confidences is not None:
        confidences = [float(detected_confidences[d]) for d, _ in pairs]
        right = [detected_labels[d] == truth_labels[t] for d, t in pairs]
        report.sure = _accuracy_of([r for c, r in zip(confidences, right) if c >= UNSURE])
        report.unsure = _accuracy_of([r for c, r in zip(confidences, right) if c < UNSURE])
        report.reliability = _reliability(confidences, right)
    return report


def evaluate_take(y, truth_times, truth_labels, profile=None, sr=SR, tolerance=TOLERANCE_S):
    """Run detection and classification on a take, and score them against its labels.

    Tempo and quantising are left out on purpose: this measures what the app
    heard, before anything is snapped to a grid.
    """
    times, _ = detect_onsets(y, sr)
    scorer = scorer_for(profile)
    scored = [scorer(extract_features(y, t, sr)) for t in times]
    labels = [label for label, _ in scored]
    confidences = [confidence for _, confidence in scored]
    return score(times, labels, np.asarray(truth_times, dtype=float), list(truth_labels), tolerance,
                 detected_confidences=confidences)


def format_report(report):
    def ms(value):
        return "n/a" if np.isnan(value) else f"{value:.1f} ms"

    def share(value):
        return "n/a" if np.isnan(value) else f"{value:.0%}"

    def onsets(r):
        return (f"  within {r.tolerance_s * 1000:g} ms: {r.n_matched} matched   precision {r.precision:.0%}   "
                f"recall {r.recall:.0%}   F1 {r.f1:.2f}")

    lines = [
        f"Onsets: {report.n_detected} detected, {report.n_truth} labelled",
        onsets(report),
        *([onsets(report.strict)] if report.strict else []),
        f"Timing error on matched hits: median {ms(report.median_error_ms)}, "
        f"p95 {ms(report.p95_error_ms)}, bias {ms(report.median_bias_ms)} (negative is early)",
        f"Classification on matched hits: {share(report.accuracy)}",
        "  " + "   ".join(f"{label} {share(report.class_accuracy[label])}" for label in LABELS),
        "",
        "Confusion (rows: labelled, columns: predicted)",
        "         " + "".join(f"{label:>7}" for label in LABELS),
    ]
    for label, row in zip(LABELS, report.confusion):
        lines.append(f"  {label:<7}" + "".join(f"{count:>7}" for count in row))

    if report.reliability:
        def counted(pair):
            count, accuracy = pair
            return f"{count} hit{'' if count == 1 else 's'}, {share(accuracy)} right"

        lines += [
            "",
            "Confidence vs accuracy on matched hits (UNCALIBRATED: confidence is a relative",
            "score, not a probability, so check this table before trusting it)",
            f"  sure   (>= {UNSURE:.0%}): {counted(report.sure)}",
            f"  unsure (<  {UNSURE:.0%}): {counted(report.unsure)}",
            "  confidence   hits   right",
        ]
        for low, high, count, accuracy in report.reliability:
            lines.append(f"  {low:>4.0%}-{high:<4.0%}  {count:>6}   {share(accuracy):>5}")
    return "\n".join(lines)

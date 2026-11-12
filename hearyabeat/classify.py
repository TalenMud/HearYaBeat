"""Decide whether each hit is a kick, snare or hat, by the character of the sound."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import LABELS, SR
from .detect import detect_onsets

WINDOW_S = 0.08
DECAY_WINDOW_S = 0.3
FEATURE_NAMES = ("log_centroid", "zcr", "low_ratio", "flatness", "log_decay")
K = 3
HITS_PER_SOUND = 3

# Hand-tuned defaults, used until someone calibrates.
KICK_CENTROID_HZ = 900
KICK_LOW_RATIO = 0.4
HAT_CENTROID_HZ = 5500
HAT_SOFT_CENTROID_HZ = 4500
HAT_DECAY_MS = 60

# Confidence. In threshold mode, a hit this far past every boundary scores 1.
CLEAR_OCTAVES = 0.5
CLEAR_LOW_RATIO = 0.1
VOTE_EPSILON = 1e-6  # stops a neighbour at distance 0 from dividing by zero
UNSURE = 0.5  # below this confidence, a hit is shown as uncertain


class CalibrationError(ValueError):
    """A calibration take did not contain the hits we asked for."""


def _window(y, start, length):
    chunk = y[start : start + length]
    if chunk.size < length:
        chunk = np.pad(chunk, (0, length - chunk.size))
    return chunk


def extract_features(y, onset_time, sr=SR):
    """Return the feature vector for the hit starting at onset_time."""
    start = int(onset_time * sr)
    frame = _window(y, start, int(WINDOW_S * sr)).astype(np.float64)
    frame = frame - frame.mean()

    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size), n=4096)) ** 2
    freqs = np.fft.rfftfreq(4096, 1 / sr)
    total = spectrum.sum() + 1e-12
    centroid = float((freqs * spectrum).sum() / total)
    low_ratio = float(spectrum[freqs < 250].sum() / total)
    upper = spectrum[freqs >= 200] + 1e-12
    flatness = float(np.exp(np.log(upper).mean()) / upper.mean())
    zcr = float(np.mean(np.signbit(frame[1:]) != np.signbit(frame[:-1])))

    tail = _window(y, start, int(DECAY_WINDOW_S * sr)).astype(np.float64)
    hop = 128
    envelope = np.sqrt(
        np.convolve(tail**2, np.ones(hop) / hop, mode="valid")[::hop] + 1e-12
    )
    peak = int(np.argmax(envelope))
    below = np.flatnonzero(envelope[peak:] < envelope[peak] * 0.1)
    frames = int(below[0]) if below.size else envelope.size - peak
    decay_ms = frames * hop / sr * 1000

    return np.array(
        [np.log2(centroid + 1.0), zcr, low_ratio, flatness, np.log2(decay_ms + 1.0)]
    )


def classify_by_thresholds(features):
    centroid = 2.0 ** features[0] - 1.0
    low_ratio = features[2]
    decay_ms = 2.0 ** features[4] - 1.0
    if low_ratio > KICK_LOW_RATIO or centroid < KICK_CENTROID_HZ:
        return "kick"
    if centroid > HAT_CENTROID_HZ or (centroid > HAT_SOFT_CENTROID_HZ and decay_ms < HAT_DECAY_MS):
        return "hat"
    return "snare"


def _threshold_scores(features):
    """Signed distances behind classify_by_thresholds, in units where 1 is clearly clear.

    Positive kick means a kick rule fires, and positive hat means a hat rule
    fires. Centroid and decay are measured in octaves, since the features are
    log2, and low_ratio in its own units.
    """
    log_centroid, low_ratio, log_decay = features[0], features[2], features[4]
    octave = lambda hz: np.log2(hz + 1.0)
    kick = max(
        (low_ratio - KICK_LOW_RATIO) / CLEAR_LOW_RATIO,
        (octave(KICK_CENTROID_HZ) - log_centroid) / CLEAR_OCTAVES,
    )
    hat = max(
        (log_centroid - octave(HAT_CENTROID_HZ)) / CLEAR_OCTAVES,
        min(
            (log_centroid - octave(HAT_SOFT_CENTROID_HZ)) / CLEAR_OCTAVES,
            (octave(HAT_DECAY_MS) - log_decay) / CLEAR_OCTAVES,
        ),
    )
    return kick, hat


def classify_by_thresholds_with_confidence(features):
    """Return (label, confidence from 0 to 1) for the hand-tuned rules.

    The label is exactly classify_by_thresholds'. Confidence is the distance
    to the nearest threshold that would change it, so a hit sitting right on
    a boundary scores about 0.
    """
    label = classify_by_thresholds(features)
    kick, hat = _threshold_scores(features)
    if label == "kick":
        margin = kick  # every firing kick rule must fail before the label changes
    elif label == "hat":
        margin = min(hat, -kick)  # it becomes a snare, or a kick
    else:
        margin = min(-kick, -hat)
    return label, float(np.clip(margin, 0.0, 1.0))


@dataclass
class Profile:
    """One performer's calibration: labelled examples of their own sounds.

    Hits are labelled by a majority vote of the K=3 nearest examples.

    leave_one_out() is deliberately pessimistic. It uses the same K=3 as the
    shipped classifier so that it describes that classifier, but with one
    example held out its own sound has only 2 examples left. A held-out
    example is then only labelled right if both of its siblings are among
    its 3 nearest neighbours, while the saved profile has all 3 per sound.
    Treat its accuracy as a lower bound, not a measurement.
    """

    name: str
    examples: np.ndarray  # shape (n, n_features)
    labels: list

    def __post_init__(self):
        self.examples = np.asarray(self.examples, dtype=np.float64)
        self._mean = self.examples.mean(axis=0)
        self._scale = np.maximum(self.examples.std(axis=0), 1e-3)

    def _distances(self, features):
        scaled = (self.examples - self._mean) / self._scale
        query = (np.asarray(features) - self._mean) / self._scale
        return np.linalg.norm(scaled - query, axis=1)

    def _nearest(self, features):
        distances = self._distances(features)
        nearest = np.argsort(distances, kind="stable")[:K]
        return [(self.labels[int(i)], float(distances[i])) for i in nearest]

    def classify(self, features):
        votes = {}
        for label, distance in self._nearest(features):
            count, total = votes.get(label, (0, 0.0))
            votes[label] = (count + 1, total + distance)
        # Most votes wins, and on a tie the closer group wins.
        return min(votes, key=lambda label: (-votes[label][0], votes[label][1]))

    def classify_with_confidence(self, features):
        """Return (label, confidence from 0 to 1).

        The label is exactly classify()'s. Confidence is how far that label's
        distance-weighted vote beats the strongest other label, as a share of
        all the weight. Voters are the K nearest examples plus the nearest
        example of every sound.

        The extra voters matter. With 3 examples per sound, the K=3 nearest are
        often all one sound, so a vote among them alone is unanimous, and
        reads as fully confident however close the next sound is. That was
        confidently wrong on a synthetic snare.

        Close neighbours of one sound, with every other sound far away, give
        about 1. A hit about as far from its own sound as from another,
        including one far from every example, falls below UNSURE. Exactly
        between two sounds it bottoms out near 0.3 rather than 0, because
        the winner still has 2 of the 3 nearest votes.
        """
        label = self.classify(features)
        distances = self._distances(features)
        voters = set(np.argsort(distances, kind="stable")[:K].tolist())
        for sound in set(self.labels):
            members = [i for i, name in enumerate(self.labels) if name == sound]
            voters.add(min(members, key=lambda i: distances[i]))
        weights = {}
        for i in voters:
            neighbour = self.labels[i]
            weights[neighbour] = weights.get(neighbour, 0.0) + 1.0 / (float(distances[i]) + VOTE_EPSILON)
        total = sum(weights.values())
        rival = max((w for name, w in weights.items() if name != label), default=0.0)
        return label, float(np.clip((weights.get(label, 0.0) - rival) / total, 0.0, 1.0))

    def leave_one_out(self):
        """Classify each example with a profile built from all the others.

        With only three examples per sound this is the honest estimate of how
        well this person's sounds can be told apart: every example is judged
        by a profile that never saw it, scaling included.
        """
        mistakes = {}
        correct = 0
        for i, truth in enumerate(self.labels):
            others = [j for j in range(len(self.labels)) if j != i]
            held_out = Profile(self.name, self.examples[others], [self.labels[j] for j in others])
            guess = held_out.classify(self.examples[i])
            if guess == truth:
                correct += 1
            else:
                mistakes[(truth, guess)] = mistakes.get((truth, guess), 0) + 1
        return LeaveOneOut(correct / len(self.labels), mistakes, len(self.labels))


@dataclass(frozen=True)
class LeaveOneOut:
    accuracy: float
    mistakes: dict  # (true label, predicted label) -> count
    n_examples: int

    def overlapping(self):
        """Pairs of sounds that were mixed up either way, most mixed up first."""
        pairs = {}
        for (truth, guess), count in self.mistakes.items():
            pair = tuple(sorted((truth, guess), key=LABELS.index))
            pairs[pair] = pairs.get(pair, 0) + count
        return sorted(pairs.items(), key=lambda item: (-item[1], LABELS.index(item[0][0])))


# What to change when two sounds come out too alike.
SEPARATION_TIPS = {
    ("kick", "snare"): "Make the kick deeper and rounder (\"boom\"), and the snare a sharp, breathy \"pff\" or \"k\".",
    ("kick", "hat"): "Make the kick a deep \"boom\", and the hat a short, thin \"ts\".",
    ("snare", "hat"): "Make the hat shorter and thinner (\"ts\"), and the snare longer and punchier (\"pff\").",
}


def describe_separation(loo):
    """Turn a LeaveOneOut into a short message a performer can act on."""
    estimate = f"about {loo.accuracy:.0%}, a cautious estimate from {loo.n_examples} examples"
    if not loo.mistakes:
        return f"Your sounds are well separated: {estimate}."
    lines = [f"Your sounds were told apart {estimate}."]
    for (a, b), count in loo.overlapping():
        lines.append(f"Your {a} and {b} sound alike to the app ({count} mixed up). {SEPARATION_TIPS[(a, b)]}")
    lines.append("Then calibrate again.")
    return " ".join(lines)


def classifier_for(profile):
    return profile.classify if profile is not None else classify_by_thresholds


def scorer_for(profile):
    """Like classifier_for, but each call returns (label, confidence)."""
    return profile.classify_with_confidence if profile is not None else classify_by_thresholds_with_confidence


def calibration_features(y, sr=SR, expected=HITS_PER_SOUND):
    """Return one feature vector per hit, or raise if the hit count is wrong."""
    times, _ = detect_onsets(y, sr)
    if len(times) != expected:
        raise CalibrationError(f"Heard {len(times)} hit(s), expected {expected}.")
    return [extract_features(y, t, sr) for t in times]


def build_profile(name, takes, sr=SR):
    """Build a Profile from a dict of label -> audio of that sound repeated."""
    examples, labels = [], []
    for label in LABELS:
        for vector in calibration_features(takes[label], sr):
            examples.append(vector)
            labels.append(label)
    return Profile(validate_name(name), np.array(examples), labels)


def validate_name(name):
    # Profile names become file names.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name or ""):
        raise ValueError("Profile names can use letters, numbers, - and _ (up to 40 characters).")
    return name


def save_profile(profile, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{validate_name(profile.name)}.json"
    data = {
        "name": profile.name,
        "features": list(FEATURE_NAMES),
        "examples": profile.examples.tolist(),
        "labels": profile.labels,
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def load_profile(name, directory):
    path = Path(directory) / f"{validate_name(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"No profile called {name}. Run: python cli.py calibrate --name {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("features") != list(FEATURE_NAMES):
        raise ValueError(f"Profile {name} was made with different features. Please recalibrate.")
    return Profile(data["name"], np.array(data["examples"]), data["labels"])


def list_profiles(directory):
    directory = Path(directory)
    return sorted(p.stem for p in directory.glob("*.json")) if directory.exists() else []

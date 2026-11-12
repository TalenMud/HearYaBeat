"""What the classifier sees, for the "under the hood" view.

Builds plain Vega-Lite specs with the data inline. Streamlit ships Vega-Lite,
so this needs nothing beyond what's already installed.
"""

import numpy as np

from . import LABELS
from .classify import FEATURE_NAMES, HAT_CENTROID_HZ, KICK_CENTROID_HZ, KICK_LOW_RATIO

CENTROID = FEATURE_NAMES.index("log_centroid")
LOW_RATIO = FEATURE_NAMES.index("low_ratio")
STAGES = ("detect", "tempo", "classify", "render")


def _hz(log_centroid):
    return float(2.0 ** log_centroid - 1.0)


def feature_points(result, profile=None):
    """One row per hit, plus one per calibration example when a profile is given."""
    points = []
    for hit, vector in zip(result.hits, result.features):
        points.append({
            "centroid_hz": _hz(vector[CENTROID]), "low_ratio": float(vector[LOW_RATIO]),
            "label": hit.label, "kind": "this take", "confidence": round(hit.confidence, 2),
            "time_s": round(hit.time, 3),
        })
    if profile is not None:
        for vector, label in zip(profile.examples, profile.labels):
            points.append({
                "centroid_hz": _hz(vector[CENTROID]), "low_ratio": float(vector[LOW_RATIO]),
                "label": label, "kind": f"calibration ({profile.name})", "confidence": None, "time_s": None,
            })
    return points


def scatter_spec(result, colours, profile=None):
    """Brightness against low-end energy: the two features that do most of the work.

    The view shows only 2 of the 5 features, so two points close together
    here can still be far apart to the classifier. Without a profile, dashed
    lines mark the default thresholds those 2 features use.
    """
    points = feature_points(result, profile)
    colour = {"field": "label", "type": "nominal", "title": "sound",
              "scale": {"domain": list(LABELS), "range": [colours[label] for label in LABELS]}}
    layers = [{
        "mark": {"type": "point", "filled": True, "size": 90},
        "encoding": {
            "x": {"field": "centroid_hz", "type": "quantitative", "scale": {"type": "log"},
                  "title": "spectral centroid (Hz, log scale): brighter to the right"},
            "y": {"field": "low_ratio", "type": "quantitative", "title": "share of energy below 250 Hz"},
            "color": colour,
            "shape": {"field": "kind", "type": "nominal", "title": "point",
                      "scale": {"range": ["circle", "diamond"]}},
            "opacity": {"condition": {"test": "datum.kind == 'this take'",
                                      "field": "confidence", "type": "quantitative",
                                      "scale": {"domain": [0, 1], "range": [0.25, 1]}, "legend": None},
                        "value": 0.55},
            "tooltip": [{"field": "label"}, {"field": "kind"}, {"field": "confidence"},
                        {"field": "time_s", "title": "time (s)"},
                        {"field": "centroid_hz", "format": ".0f"}, {"field": "low_ratio", "format": ".2f"}],
        },
    }]
    if profile is None:
        rules = [{"centroid_hz": KICK_CENTROID_HZ, "rule": "kick below"},
                 {"centroid_hz": HAT_CENTROID_HZ, "rule": "hat above"}]
        layers.append({"data": {"values": rules},
                       "mark": {"type": "rule", "strokeDash": [4, 4], "color": "#888"},
                       "encoding": {"x": {"field": "centroid_hz", "type": "quantitative"},
                                    "tooltip": [{"field": "rule"}]}})
        layers.append({"data": {"values": [{"low_ratio": KICK_LOW_RATIO, "rule": "kick above"}]},
                       "mark": {"type": "rule", "strokeDash": [4, 4], "color": "#888"},
                       "encoding": {"y": {"field": "low_ratio", "type": "quantitative"},
                                    "tooltip": [{"field": "rule"}]}})
    return {"data": {"values": points}, "layer": layers, "height": 320}


WAVEFORM_BINS = 600


def envelope(y, sr, bins=WAVEFORM_BINS):
    """Peak level per time slice, as [{"time_s", "peak"}]. Enough to draw, small enough to send."""
    if y.size == 0:
        return []
    edges = np.linspace(0, y.size, min(bins, y.size) + 1).astype(int)
    return [{"time_s": round(float(start / sr), 4), "peak": round(float(np.max(np.abs(y[start:end]))), 4)}
            for start, end in zip(edges[:-1], edges[1:])]


def waveform_spec(y, sr, result, colours):
    """The take's waveform, with a line at each detected hit, coloured by sound and faded by confidence."""
    colour = {"field": "label", "type": "nominal", "legend": None,
              "scale": {"domain": list(LABELS), "range": [colours[label] for label in LABELS]}}
    hits = [{"time_s": round(hit.time, 4), "label": hit.label, "confidence": round(hit.confidence, 2)}
            for hit in result.hits]
    time_axis = {"field": "time_s", "type": "quantitative", "title": "seconds",
                 "scale": {"domain": [0, round(y.size / sr, 3)]}}
    return {
        "height": 110,
        "layer": [
            {"data": {"values": envelope(y, sr)},
             "transform": [{"calculate": "-datum.peak", "as": "trough"}],
             "mark": {"type": "area", "color": "#888", "opacity": 0.5},
             "encoding": {"x": time_axis,
                          "y": {"field": "peak", "type": "quantitative", "axis": None},
                          "y2": {"field": "trough"}}},
            {"data": {"values": hits},
             "mark": {"type": "rule", "strokeWidth": 2},
             "encoding": {"x": time_axis, "color": colour,
                          "opacity": {"field": "confidence", "type": "quantitative",
                                      "scale": {"domain": [0, 1], "range": [0.3, 1]}, "legend": None},
                          "tooltip": [{"field": "label"}, {"field": "confidence"},
                                      {"field": "time_s", "title": "time (s)"}]}},
        ],
    }


def timing_summary(result):
    """Return (rows of (stage, ms), total ms, real-time factor).

    The real-time factor is processing time over the take's length, so under
    1 means it analysed faster than the take lasted. NaN if the length is unknown.
    """
    rows = [(stage, result.timings_ms[stage]) for stage in STAGES if stage in result.timings_ms]
    total = sum(ms for _, ms in rows)
    factor = total / 1000 / result.take_s if result.take_s else float("nan")
    return rows, total, factor

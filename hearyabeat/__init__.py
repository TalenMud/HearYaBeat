"""HearYaBeat: beatbox into the mic, get a clean drum loop back."""

SR = 44100  # one sample rate everywhere
HOP = 512  # about 11.6 ms at 44.1 kHz
LABELS = ("kick", "snare", "hat")
STEPS_PER_BAR = 16
MIN_BPM = 70
MAX_BPM = 180

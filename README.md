# HearYaBeat

Beatbox into the mic ("boom cha boom cha"). HearYaBeat detects the tempo and each sound, swaps them for drum samples, and plays a clean loop. Hum on top to add a bassline.

Built for the Hiya AI Voice Challenge, Creative Voice track.

## Setup

Requires Python 3.11, 3.12 or 3.13.

```
git clone <this repo>
cd HearYaBeat
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS and Linux. Then:

```
pip install -r requirements.txt
```

Platform notes:
- **Linux** needs PortAudio for the mic: `sudo apt install libportaudio2`.
- **macOS** will ask for microphone permission for your terminal the first time you record.

## Try it without a microphone

```
python cli.py record --input examples/demo_take.wav --no-play
```

```
python cli.py hum --input examples/demo_hum.wav --no-play
```

These analyse a synthetic take and print the BPM and step grid, then add a bassline from a synthetic hum. Both example files are generated audio, not a real voice. In the web page, use **Try the demo take** and **Try a demo hum**.

Running these replaces whatever is in `output/`, including your last recorded take.

## Run it

The web page is the easiest way in:

```
streamlit run app.py
```

Or from the terminal:

```
python cli.py calibrate --name you      # optional: teach it your sounds
python cli.py record --profile you --bpm 100 --bars 2
python cli.py hum
```

Use headphones, so the mic doesn't pick up the clicks and the loop.

After four clicks, beatbox for the number of bars you asked for. The click keeps going quietly through the take, with the first beat of each bar accented, and the loop is locked to that tempo. A live display shows which bar and beat you're on. On speakers, turn the click off (`--no-click`, or the toggle in the web page) and the tempo is detected from your beatboxing instead. The results land in `output/`: `loop.wav` is the drums, and `loop_bass.wav` adds the bassline.

Recording a new take at the same tempo and length keeps your bassline. Otherwise the bassline is cleared.

## How it works

1. **Record** with a count-in, and a click through the take unless you turn it off. The count-in is cut from the audio before analysis.
2. **Find hits** with librosa's onset detection, then walk back to the true start of each attack.
3. **Find the tempo.** With the click on, the tempo is the click's. Without it, the tempo is found by scoring every BPM from 70 to 180 on how well the hits sit on a 16th-note grid. The count-in and `librosa.beat.beat_track` are votes, not the answer, because beat tracking alone is unreliable on a few seconds of audio.
4. **Classify each hit** from an 80 ms window, using spectral centroid, zero crossing rate, low-frequency energy, spectral flatness and decay time. Hand-tuned thresholds are the default. Calibration replaces them with 3-nearest-neighbour on your own examples.
5. **Quantise** to 16th notes and render the samples into a seamless loop.
6. **Bassline**: track the hum with `librosa.pyin`, fit a minor pentatonic key, snap the notes, and render a synth bass under the loop.

## How sure it is

The app shows when it isn't sure, instead of hiding it.
- **Faded cells:** each hit gets a confidence from 0 to 1. The web page fades unsure cells, and hovering over a cell shows its value. The terminal prints hits under 50% as `?` in that sound's row.
- **With a calibration profile,** confidence is a distance-weighted vote among the 3 nearest examples, plus the nearest example of every other sound. The extra voters stop a unanimous vote from reading as fully confident when another sound is nearly as close.
- **With the default thresholds,** confidence is how far the hit sits from the nearest threshold that would change its label.
- **After calibrating,** the app estimates how well your sounds can be told apart, by judging each example with a profile built from the other 8. Read it as a cautious, lower-bound estimate, because each held-out sound has only 2 examples left. If two sounds overlap, it says which ones and what to change.

**Under the hood**, in the web page, shows two things:
- **A scatter of every hit** by brightness and low-end energy, alongside your calibration examples or the default thresholds. It shows 2 of the 5 features, so points that look close can still be far apart to the classifier.
- **Per-stage timings** for detect, tempo, classify and render. The page analyses a short synthetic take once at start-up, so that librosa's one-off compile, about 3 s measured here, doesn't land on your first real take.

## Change the drum sounds

The default kit in `samples/` is synthesised, so it carries no licensing issues. To use recordings, overwrite `samples/kick.wav`, `snare.wav` or `hat.wav`. Any sample rate or channel count works. `python cli.py gen-samples` restores the defaults.

## Troubleshooting

| Problem | Try |
| --- | --- |
| Wrong mic or speakers | `python cli.py devices`, then add `--mic N --speaker N` |
| "Could not open the audio devices" | Check the mic is plugged in and allowed in your OS privacy settings |
| Sounds get mislabelled | Run `calibrate`. Leave a clear gap between each sound |
| "The timing was loose" | Keep the click on and use headphones, so you stay on the grid |
| BPM looks half or double with `--no-click` | Set `--bpm` near the tempo you're performing |
| Bass notes are wrong | Use headphones, and hold each note for at least an 8th note |

## Measure it on real takes

The tests prove the logic on synthetic voices. To measure accuracy on a real recording, label its hits by hand in a CSV next to the WAV, with the same name:

```
time_s,label
0.512,kick
0.873,hat
1.240,snare
```

Labels are `kick`, `snare` or `hat`, and times are in seconds from the start of the file. Your last take is always saved as `output/last_take.wav`. Copy it somewhere else first, because the next take replaces it. Audacity's label track is one way to mark the times. Then run:

```
python cli.py evaluate takes/mine.wav                # uses takes/mine.csv
python cli.py evaluate takes/mine.wav --profile you  # with your calibration
```

It runs detection and classification only, before any tempo or grid snapping, and reports:
- **Onset precision, recall and F1.** A detection counts as a hit if it's within 50 ms of a label (`--tolerance-ms` changes this). A second line repeats the scores at a strict 25 ms. Each detection and each label is matched once, closest pairs first.
- **Timing error** on matched hits: median and 95th percentile, plus the median signed bias, where negative means detections land early.
- **Per-class accuracy and a confusion matrix** on matched hits, with labelled class as rows and predicted class as columns.
- **Confidence against accuracy:** accuracy for hits above and below the 50% "unsure" cut-off, and a reliability table with buckets at 0–25%, 25–50%, 50–75% and 75–100%. The output marks this as **uncalibrated**. Confidence is a relative score, not a probability, and this table is how you find out whether it means anything for your voice.

### Labels from a script

Labelling by hand is slow. If you decide a pattern before you record and perform it to the click, the labels can be written from the script:

```
python cli.py record --bpm 100 --bars 2            # perform: kick hat snare hat, in 8ths
copy output\last_take.wav takes\scripted.wav       # cp on macOS and Linux
python cli.py script-labels takes/scripted.wav --pattern "kick hat snare hat kick kick snare hat" --bpm 100 --bars 2
python cli.py evaluate takes/scripted.wav
```

- **The pattern** is one bar, or several separated by `|`, cycling until the bars are filled. Each bar is spread evenly over 16 steps, so it needs 1, 2, 4, 8 or 16 slots. Use `-` for a rest.
- **`--offset`** is the time from the file's start to the first downbeat. Takes saved by the app start 50 ms before it, which is the default.
- It won't overwrite an existing CSV without `--force`, and it refuses if scripted hits fall after the recording ends.

What this measures is different from hand labels:
- **Labelling accuracy** is measured as usual, as long as you performed the sounds you scripted.
- **Timing error is measured against the intended grid, not against when you actually hit.** So it includes your own drift and wobble on top of any detector error. A large bias or p95 here may be you, not the backtrack. Use hand labels to measure the detector's timing on its own.
- **Missed or extra hits.** If you skip a hit or add one, it counts against the app's precision and recall. Listen back before trusting the onset scores.

The script labels are kept apart from the detector. `hearyabeat/script.py` uses only the pattern, tempo and offset, never the audio, and a test checks it never loads the detector or classifier. If it did, the evaluation would be measuring the app against itself.

`examples/demo_take.csv` labels the synthetic demo take. `python cli.py evaluate examples/demo_take.wav` scores it perfectly, which only shows that the harness works.

## Tests

```
pytest
```

The tests use synthetic audio with known answers and stubbed audio devices. They check the logic, not accuracy on real voices.

## Project layout

- `hearyabeat/`: the analysis and audio code. `pipeline.py` is the entry point and never touches a device.
- `hearyabeat/evaluate.py`: scoring against hand labels. `hearyabeat/script.py`: labels from a scripted pattern, independent of the detector. `hearyabeat/insight.py`: the under-the-hood charts and timings.
- `cli.py`, `app.py`: the terminal and web interfaces. `.streamlit/config.toml`: the page theme.
- `samples/`: the drum kit. `profiles/`: saved calibrations. `output/`: renders.
- `ai.md`: how AI tools were used. `WRITEUP.md`: the one-page summary.

# AI usage log

An honest record of which AI tools were used on HearYaBeat, and how.

## Tools

- **Claude Code** (model: Claude Opus 5.5), running in the VS Code extension. Used for planning, writing the code and tests, and drafting the docs.

## Session 1: 10 Nov 2026, plan and first build

### The prompt

I pasted my project brief as the opening prompt. It covered the one-line idea, why voice, the Python stack, the pipeline steps, the demo, deliverables, and a build order.

### Decisions I made when Claude asked

| Question | My choice |
| --- | --- |
| Interface | CLI first, then a local Streamlit page |
| Drum samples | Synthesise them in numpy, with the option to overwrite them with CC0 recordings |
| Scope | Core, calibration, hum bassline. No spoken commands, because they would need a speech recogniser |

### My corrections to Claude's plan

> It's Hiya AI Voice Challenge not a hackathon entry. keep in mind engineers will be cloning and using this so will the version python thing cause any trouble? if not likely to we'll test it from a fresh install again at the end.

This led to pinned dependencies, a no-microphone demo command, Linux and macOS notes in the README, and a fresh-install test across Python versions.

### What Claude changed from my brief, and why

- **Tempo.** My brief used `beat_track`. Claude argued it is unreliable on 4 seconds of audio, so it became one vote among others in a search over 70 to 180 BPM.
- **Onset timing.** The brief said to use librosa's backtracking. A test showed it landing 12 to 33 ms early, by a different amount per hit, so it was replaced with a walk back along a finer energy envelope.
- **Double-time error.** A test showed 85 BPM being read as 170. The scoring had rewarded hits landing on quarter notes, which favoured doubled tempos. It was reweighted.
- **Python.** Claude's plan assumed Python 3.11 and 3.12 were installed because the `py` launcher listed them, but both pointed at deleted files. It used `uv` to fetch working copies.
- **Python 3.11.** The fresh-install test failed on 3.11 because librosa 1.0.0 needs 3.12 or newer. `requirements.txt` now pins older librosa, numpy and scipy for 3.11 only, and the same tests pass there.

### What AI could not check

Claude cannot hear or beatbox. Its tests use synthetic audio and stubbed audio devices, so real microphone behaviour, real-voice accuracy and how the result sounds had to be tested by me.

## Session 2: 12 Nov 2026, first real-mic test

### My feedback after testing

> One of the tricky things I found is that it struggles to kind of help the user in the timing. So when I put in a BPM flag and I put a BPM, I feel like it's hard to know my progress through the bars. [...] it might be worth having a little like slider to show like, oh, you're going through the recording.

### What Claude found in my saved take

Claude analysed `output/last_take.wav`. The count-in was 100 BPM, but I'd drifted to about 94, and my timing wobbled by up to ±70 ms from beat to beat. No single tempo fitted, so the app guessed 70 BPM. The take also showed about six hats counted twice, and about ten quiet breath sounds counted as hits.

### What changed

- **Live progress.** The web page shows four large beat dots, "bar 2 of 4" and a progress bar while recording or humming. The terminal shows the same on one line.
- **Click through the take.** It's on by default and needs headphones. With it on, the loop locks to the click's tempo rather than guessing. `--no-click` turns it off for speakers.
- **Cleaner hits.** Hits within 60 ms are merged, and hits below 30% of a typical hit's level are dropped. My take went from 48 hits to 38.
- **Loose-timing warning.** A low timing fit now says the grid is a best guess and suggests the click.

### Demo bassline

I pointed out that the demo had no bassline. Claude added a demo hum: a synthetic A minor pentatonic riff that fits whatever loop is loaded. It's in the web page as a button, and in the terminal as `examples/demo_hum.wav`. While testing it, Claude's terminal run overwrote my real take in `output/`.

## Session 3: 12 Nov 2026, making it measure itself

### My prompt

> Problem: my tests only use synthetic voices, so they prove the logic, not real-mic accuracy. I want the project to measure itself and expose its uncertainty. [...] Constraints: numpy/scipy/librosa/streamlit only, keep the existing code style, don't touch detection or tempo unless a test shows a bug. Run pytest after each step and stop for me to review.

It listed five steps: an evaluation harness, calibration diagnostics, per-hit confidence, an "under the hood" view with timings, and minimal polish. Claude worked one step at a time, running pytest after each and stopping for my review. Detection and tempo code were not changed.

### My decisions at each review

- **Step 1:** keep the signed median timing error, because it shows whether the backtrack is biased early or late. Keep 50 ms as the default tolerance, but print a second line at 25 ms.
- **Step 2:** keep k=3 in leave-one-out so it describes the shipped classifier, label it "a cautious estimate" in both interfaces, and explain the pessimism in the Profile docstring.
- **Step 3:** accept Claude's rival-class voter (described below). Add accuracy above and below the 50% cut-off, plus a reliability table marked uncalibrated. Make the regression test robust: the mislabelled snare must be under 0.6 and below the median confidence of the correct hits, instead of relying on 49% against 50%.
- **Step 4:** add a one-off warm-up at app start, with a note in the timings if it didn't run.

### What testing caught

- **A floating-point bug in matching.** A hit labelled exactly 50 ms off was rejected, because 1.05 − 1.0 comes out slightly over 0.05.
- **Confidently wrong confidence.** Claude's first per-hit confidence was the vote margin among the 3 nearest examples. A synthetic profile labelled a demo snare as a hat with 100% confidence. With 3 examples per sound, the 3 nearest are often all one sound, so the vote is unanimous however close the next sound is. The fix gives the nearest example of every sound a weighted vote. That hit dropped to 49%.
- **A wrong test guess.** Claude assumed confidence would be near 0 midway between two sounds. It bottoms out near 0.3, because the winner still has 2 of the 3 nearest votes. Claude printed the curve, then fixed the test and the docstring to match the measurement.
- **Start-up cost.** Claude estimated librosa's one-off compile at about 1 s. Measured in a fresh process, it was about 3 s, and the first real take fell from 3,047 ms to 75 ms after the warm-up.
- **Claude's own mistakes.** Shell-driven edits twice turned `\n` escapes into real line breaks, which broke the file until it was fixed. One test Claude wrote had no assertions, and Claude caught it in review.

### Not yet known

All the numbers above come from synthetic audio. Real-voice accuracy, and whether confidence tracks it, will come from my own labelled takes through `cli.py evaluate`. Session 5 has the first of those results.

## Session 4: 12 Nov 2026, scripted labels

### My prompt

> Add a way to make ground-truth labels for a scripted take: given a pattern (e.g. "kick hat snare hat"), BPM, bars and the recording's click start offset, write time_s,label CSV on the grid. Note in the README that this measures labelling accuracy and timing against the intended grid, so timing error includes my own human drift; and keep it clearly separate from the detector so the evaluation isn't circular.

### What Claude built

- `hearyabeat/script.py` and `python cli.py script-labels`. The offset defaults to the 50 ms pre-roll that saved takes start with.
- **Separation from the detector:** a test imports the module in a fresh process and checks that the detector, classifier and pipeline are never loaded.
- **Safety:** it refuses to overwrite an existing CSV without `--force`, because hand labels are slow to make. It also refuses when the script runs past the end of the recording.
- **Drift check on the demo take,** which has 8 ms of synthetic jitter: scoring against the script gives a larger timing error than scoring against the exact labels, from the same detections. A test checks this.

## Session 5: 12 Nov 2026, measuring a real take and shipping

### My prompt

> We're done polishing; let's ship. In order: 1. Run python cli.py evaluate on each take in takes/ (labels from script-labels) and fill the WRITEUP "Results on real voices" section with exactly those numbers. [...] 2. Add a short Session 5 entry to ai.md [...] 3. Clean install in a fresh venv from requirements, then run the full test suite and launch the app once [...] 4. Check .gitignore [...] 5. Commit in a few logical chunks [...], create a public GitHub repo named HearYaBeat, push [...]. No new features.

### What Claude ran

- `python cli.py evaluate takes/take1.wav` on my one real take: scripted labels, default thresholds, no profile.
- A clean copy of the project in a new Python 3.13 venv, installed from `requirements.txt`. Then `pytest` (126 passed), the no-mic demo, and a Streamlit launch, which passed its health check and served the page.

### What the results showed

The numbers are in `WRITEUP.md`. In short:
- **Detection:** it found 9 of the 16 scripted hits, with no false hits.
- **Timing:** every matched hit landed about 42 ms late, so none fell within 25 ms.
- **Classification:** it labelled almost everything as a kick. Snare and hat were 0% right.
- **Confidence:** it was high on most of the wrong answers.

So on a real voice, the uncalibrated classifier and the confidence don't work yet. This is the gap the synthetic tests couldn't show. No code was changed in response, because this session was for shipping only. Calibrating a profile and recording more takes are the next steps.

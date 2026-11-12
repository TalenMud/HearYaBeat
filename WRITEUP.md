# HearYaBeat

**Beatbox a rhythm, and get a clean drum loop back. Hum on top to add a bassline.**

## What it is

HearYaBeat turns your voice into a drum loop. You beatbox after a four-click count-in, and it finds each hit, works out the tempo, tells kicks from snares from hats, snaps everything to a 16th-note grid, and plays the result using drum samples. Hum over the loop and it tracks your pitch, finds a key, and adds a synth bassline in time.

## Who it's for

Musicians and producers who have a rhythm in their head and want to hear it before it's gone, and people with no production experience who can make a beat with their mouth but not with a drum machine.

## The problem

Beatboxing is the fastest way a person can express a rhythm. Clicking one into a step sequencer is slow and kills the idea. Voice keeps you in the creative flow, with your hands free.

The hard part is that no two people beatbox alike. One person's kick is another person's snare. So HearYaBeat has a quick calibration: make each sound three times, and it classifies your hits by comparing them with your own examples.

## How it works

Everything runs locally in Python. librosa finds the hits and tracks the hum's pitch, and numpy handles the features, grid and audio rendering. Each hit is described by how its sound is shaped: brightness, noisiness, low-end energy and decay. A nearest-neighbour search over those features adapts it to each voice.

Tempo needed more than an off-the-shelf beat tracker, which struggles with a few seconds of audio. HearYaBeat searches every tempo from 70 to 180 BPM for the one that places the hits most tightly on a musical grid.

HearYaBeat also measures itself, because its tests only use synthetic voices. An evaluation command scores it against hand-labelled takes: whether it found each hit, how accurately it timed them, and which sounds it confused. Every hit carries a confidence. Unsure hits are faded in the grid, and the evaluation checks whether that confidence actually tracks accuracy.

## Results on real voices

So far this is one take by one performer, scored with the default thresholds (no calibration profile), so it is a first data point rather than a result. The take is 2 bars of a scripted pattern at 100 BPM. The labels came from that script, not from marking hits by hand, so timing error is measured against the intended grid and includes my own drift.

From `python cli.py evaluate`:
- **Onsets:** 9 detected against 16 labelled. Within 50 ms, all 9 matched: precision 100%, recall 56%, F1 0.72. Within 25 ms, none matched (F1 0.00).
- **Timing on matched hits:** median 41.7 ms, p95 43.9 ms, bias +41.7 ms (late). Every matched hit was late by about the same amount. That looks more like a fixed offset than random drift, but one take can't tell the two apart.
- **Classification on matched hits: 33%.** Kick 75% (3 of 4), snare 0%, hat 0%. The weakness is plain: all 4 matched snares, and the 1 matched hat, were labelled as kicks. The default thresholds are not separating my snare from my kick.
- **Confidence:** all 9 matched hits were at or above the 50% cut-off, and only 33% were right. In the 75–100% bucket, 1 of 7 was right. On this take, confidence did not flag the mistakes.

The next steps are to calibrate a profile for my voice and evaluate again, and to record more takes.

Classifying sounds by their character is the same family of problem as spotting a spoofed voice: both ask what a short slice of audio is made of, rather than what words it contains.

## What next

- Spoken commands such as "faster" or "drop", to change the loop hands-free.
- More sounds per voice, such as open hats, toms and claps.
- Live looping, so that you can layer takes without stopping.
- A browser version with no install.

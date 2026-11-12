"""HearYaBeat command line. Run: python cli.py --help"""

import argparse
import sys

from hearyabeat import LABELS, audio_io, bass, samples_gen, session
from hearyabeat.bass import HumError
from hearyabeat import evaluate, script
from hearyabeat.classify import CalibrationError, describe_separation, load_profile
from hearyabeat.detect import TooFewHits
from hearyabeat.classify import UNSURE
from hearyabeat.pipeline import confidence_lanes, unsure_hits
from pathlib import Path
from hearyabeat.render import load_wav

SYMBOLS = {"kick": "K", "snare": "S", "hat": "h"}


def print_grid(result):
    """One row per sound. An unsure hit shows as ? in its row."""
    table = confidence_lanes(result)
    for label in LABELS:
        cells = []
        for step, confidence in enumerate(table[label]):
            if step and step % 16 == 0:
                cells.append("|")
            elif step and step % 4 == 0:
                cells.append(" ")
            if confidence is None:
                cells.append(".")
            else:
                cells.append(SYMBOLS[label] if confidence >= UNSURE else "?")
        print(f"  {label:<6}{''.join(cells)}")
    unsure = unsure_hits(result)
    if unsure:
        print(f"  ? = not sure of the sound ({len(unsure)} of {len(result.grid)} hits). "
              "Calibrating for your voice usually helps.")


class ProgressLine:
    """Redraws one terminal line as the take goes, e.g. Bar 2/4  [X X x .]  [#####.....]"""

    def __init__(self):
        self.last = None

    def __call__(self, progress):
        if progress.counting_in:
            text = f"Get ready... {progress.count_left}"
        else:
            beats = " ".join("X" if i + 1 == progress.beat else "." for i in range(4))
            filled = int(progress.fraction * 20)
            text = f"Bar {progress.bar}/{progress.bars}  [{beats}]  [{'#' * filled}{'.' * (20 - filled)}]"
        if text != self.last:
            print(f"\r  {text:<50}", end="", flush=True)
            self.last = text

    def done(self):
        print()


def show_outcome(outcome):
    result = outcome.result
    counts = {label: sum(session.lanes(result)[label]) for label in LABELS}
    print(f"\nBPM {result.bpm} | {result.bars} bar(s) | timing fit {result.confidence:.0%} | "
          + ", ".join(f"{n} {label}" for label, n in counts.items()))
    print_grid(result)
    if result.confidence < session.LOOSE_TIMING:
        print("\nThe timing was loose, so this grid is a best guess. "
              "Try again with the click on (the default) and headphones.")
    if outcome.bass_kept:
        print("\nKept your bassline, since the tempo and length still match.")
    if outcome.bass_cleared:
        print("\nCleared the old bassline, since the tempo or length changed. Run hum to add a new one.")
    print(f"\nSaved {outcome.path}")


def cmd_record(args):
    if args.input:
        y = load_wav(args.input)
        outcome = session.process_take(y, hint_bpm=args.bpm_hint, profile_name=args.profile)
    else:
        session_profile = args.profile or "default thresholds"
        print(f"Using: {session_profile}. Headphones recommended.")
        guide = "the click keeps going" if not args.no_click else "no click during the take"
        print(f"Four clicks at {args.bpm} BPM, then beatbox for {args.bars} bar(s) ({guide}).")
        line = ProgressLine()
        y, downbeat = audio_io.record_take(args.bpm, args.bars, mic=args.mic, speaker=args.speaker,
                                           click=not args.no_click, on_progress=line)
        line.done()
        print("Got it. Analysing...")
        outcome = session.process_take(y, hint_bpm=args.bpm, downbeat=downbeat, profile_name=args.profile,
                                       lock_bpm=not args.no_click)
    show_outcome(outcome)
    if not args.no_play:
        audio_io.play(load_wav(outcome.path), repeats=args.repeats, speaker=args.speaker)
    return 0


def cmd_calibrate(args):
    takes = {}
    for label in LABELS:
        while True:
            if args.input_dir:
                takes[label] = load_wav(f"{args.input_dir}/{label}.wav")
                break
            input(f"\nPress Enter, then do {session.HITS_PER_SOUND} {label}s, with a gap between each...")
            print("Recording...")
            try:
                takes[label] = session.calibrate_take(mic=args.mic)
                print("Good.")
                break
            except CalibrationError as error:
                print(f"{error} Let's try that one again.")
    path, loo = session.finish_calibration(args.name, takes)
    print(f"Saved {path}\n\n{describe_separation(loo)}")
    return 0


def cmd_hum(args):
    state = session.load_session(session.SESSION_PATH)
    if state is None:
        print("No loop yet. Run: python cli.py record", file=sys.stderr)
        return 1
    if args.input:
        hum = load_wav(args.input)
    else:
        loop = load_wav(session.LOOP_PATH)
        print(f"Four clicks at {state['bpm']} BPM, then hum over one pass of the loop. Use headphones.")
        line = ProgressLine()
        hum = audio_io.record_over_loop(loop, state["bpm"], state["n_steps"], mic=args.mic,
                                        speaker=args.speaker, on_progress=line)
        line.done()
    root, notes, path = session.add_hum(hum, state)
    line = " ".join(bass.note_label(note[2]) for note in notes)
    print(f"\nKey: {root} minor pentatonic\nBass: {line}\nSaved {path}")
    if not args.no_play:
        audio_io.play(load_wav(path), repeats=args.repeats, speaker=args.speaker)
    return 0


def cmd_gen_samples(args):
    for path in samples_gen.generate(session.SAMPLES):
        print("Wrote", path)
    return 0


def cmd_evaluate(args):
    labels = Path(args.labels) if args.labels else Path(args.wav).with_suffix(".csv")
    if not labels.exists():
        raise FileNotFoundError(f"No labels at {labels}. Make a CSV with a time_s,label header, or pass --labels.")
    profile = load_profile(args.profile, session.PROFILES) if args.profile else None
    times, names = evaluate.load_truth(labels)
    report = evaluate.evaluate_take(load_wav(args.wav), times, names, profile=profile,
                                    tolerance=args.tolerance_ms / 1000)
    print(f"{args.wav} against {labels} ({args.profile or 'default thresholds'}, "
          f"{args.tolerance_ms:g} ms tolerance)\n")
    print(evaluate.format_report(report))
    return 0


def cmd_script_labels(args):
    import soundfile as sf

    if not Path(args.wav).exists():
        raise FileNotFoundError(f"No recording at {args.wav}.")
    out = Path(args.out) if args.out else Path(args.wav).with_suffix(".csv")
    if out.exists() and not args.force:
        raise ValueError(f"{out} already exists. Pass --force to replace it, or --out to write somewhere else.")
    times, labels = script.scripted_truth(args.pattern, args.bpm, args.bars, args.offset)
    script.check_fits(times, sf.info(args.wav).duration, args.bpm)
    evaluate.save_truth(out, times, labels)
    print(f"Wrote {len(times)} scripted hits at {args.bpm:g} BPM over {args.bars} bar(s), "
          f"starting {args.offset:g} s in, to {out}")
    print("These are where you meant to hit, so timing error against them includes your own drift.")
    return 0


def cmd_devices(args):
    print(audio_io.list_devices())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="cli.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def device_options(command):
        command.add_argument("--mic", type=int, help="input device number (see: devices)")
        command.add_argument("--speaker", type=int, help="output device number (see: devices)")

    def playback_options(command):
        command.add_argument("--no-play", action="store_true", help="don't play the result")
        command.add_argument("--repeats", type=int, default=4, help="times to play the loop")

    record = commands.add_parser("record", help="beatbox a take, or analyse a WAV")
    record.add_argument("--input", help="analyse this audio file instead of recording")
    record.add_argument("--profile", help="calibration profile to use")
    record.add_argument("--bpm", type=int, default=100, help="count-in tempo")
    record.add_argument("--bars", type=int, default=2, help="bars to record")
    record.add_argument("--bpm-hint", type=int, help="likely tempo, used with --input")
    record.add_argument("--no-click", action="store_true",
                        help="stop the click after the count-in (for speakers; tempo is then detected)")
    device_options(record)
    playback_options(record)
    record.set_defaults(run=cmd_record)

    calibrate = commands.add_parser("calibrate", help="teach it your kick, snare and hat")
    calibrate.add_argument("--name", required=True, help="profile name")
    calibrate.add_argument("--input-dir", help="folder with kick.wav, snare.wav, hat.wav takes")
    device_options(calibrate)
    calibrate.set_defaults(run=cmd_calibrate)

    hum = commands.add_parser("hum", help="hum a bassline over the current loop")
    hum.add_argument("--input", help="use this audio file instead of recording")
    device_options(hum)
    playback_options(hum)
    hum.set_defaults(run=cmd_hum)

    generate = commands.add_parser("gen-samples", help="regenerate the default drum samples")
    generate.set_defaults(run=cmd_gen_samples)

    scoring = commands.add_parser("evaluate", help="score detection and labels against a hand-labelled take")
    scoring.add_argument("wav", help="the take to analyse")
    scoring.add_argument("--labels", help="CSV with time_s,label (default: the WAV's name with .csv)")
    scoring.add_argument("--profile", help="calibration profile to use")
    scoring.add_argument("--tolerance-ms", type=float, default=evaluate.TOLERANCE_S * 1000,
                         help="how close a detection must be to count (default 50)")
    scoring.set_defaults(run=cmd_evaluate)

    scripted = commands.add_parser("script-labels",
                                    help="write labels for a take you performed from a known pattern")
    scripted.add_argument("wav", help="the take the labels are for")
    scripted.add_argument("--pattern", required=True,
                          help='one bar per |, e.g. "kick hat snare hat" (1, 2, 4, 8 or 16 slots, - for a rest)')
    scripted.add_argument("--bpm", type=float, required=True, help="the tempo you performed at")
    scripted.add_argument("--bars", type=int, required=True, help="bars performed")
    scripted.add_argument("--offset", type=float, default=script.PRE_ROLL_S,
                          help=f"seconds from the file's start to the first downbeat "
                               f"(default {script.PRE_ROLL_S:g}, right for takes saved by the app)")
    scripted.add_argument("--out", help="CSV to write (default: the WAV's name with .csv)")
    scripted.add_argument("--force", action="store_true", help="replace an existing CSV")
    scripted.set_defaults(run=cmd_script_labels)

    devices = commands.add_parser("devices", help="list audio devices")
    devices.set_defaults(run=cmd_devices)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (TooFewHits, HumError, CalibrationError, FileNotFoundError, ValueError,
            audio_io.AudioDeviceError) as error:
        print(f"\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

from miditoolkit import MidiFile
import pandas as pd

# stats={};
def stats_on_track(midi_filename="the_strokes-reptilia", verbose=True):
    midi = MidiFile(f"./midi/{midi_filename}.mid")
    beat_count = midi.max_tick / midi.ticks_per_beat
    note_coverage_all_instrument = []
    note_counts_all_instrument = []
    for idx, instruments in enumerate(midi.instruments):

        note_coverage_instrument = 0
        for note in instruments.notes:
            note_coverage_instrument += note.end - note.start

        note_coverage_all_instrument.append(
            100 * (note_coverage_instrument / midi.max_tick)
        )

        note_counts_all_instrument.append(len(instruments.notes))

        if verbose:
            print(instruments.name)
            print(
                f"There are {note_counts_all_instrument[idx]} notes from {instruments.name}"
            )
            print(
                f"{instruments.name} covers {note_coverage_all_instrument[idx]} % of the the track "
            )
            print("-----------------------------")

    stats = dict(
        beat_count=beat_count,
        note_counts_all_instrument=note_counts_all_instrument,
        note_coverage_all_instrument=note_coverage_all_instrument,
    )
    print(stats)
    return stats

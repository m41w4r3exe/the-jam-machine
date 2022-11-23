from miditoolkit import MidiFile
import numpy as np

# stats={};
def stats_on_track(midi_filename="the_strokes-reptilia", verbose=True):
    midi = MidiFile(f"./midi/{midi_filename}.mid")
    beat_count = midi.max_tick / midi.ticks_per_beat
    min_start_all_instruments = []
    max_end_all_instruments = []
    note_coverage_all_instrument = []
    note_coverage_true_all_instrument = []
    note_counts_all_instrument = []
    for idx, instruments in enumerate(midi.instruments):

        note_coverage_instrument = 0
        note_coverage_instrument_idx = []
        for i, note in enumerate(instruments.notes):
            if i == 0:
                min_start_all_instruments.append(note.start / midi.ticks_per_beat)

            note_coverage_instrument += note.end - note.start
            note_coverage_instrument_idx.append(list(range(note.start, note.end)))

        max_end_all_instruments.append(note.end / midi.ticks_per_beat)

        note_coverage_all_instrument.append(
            100 * (note_coverage_instrument / midi.max_tick)
        )
        unique_idx_list = []
        for idx_list in note_coverage_instrument_idx:
            [unique_idx_list.append(idx) for idx in idx_list]
        unique_idx_list = len(np.unique(unique_idx_list))

        note_coverage_true_all_instrument.append(
            100 * (unique_idx_list / midi.max_tick)
        )
        note_counts_all_instrument.append(len(instruments.notes))

        if verbose:
            print(instruments.name)
            print(
                f"There are {note_counts_all_instrument[idx]} notes from {instruments.name}"
            )
            print(
                f"{instruments.name} covers {note_coverage_true_all_instrument[idx]:.0f} % of the the track "
            )
            print(
                f"{instruments.name}",
                f"first note at: {min_start_all_instruments[idx]:.1f} beats;",
                f"last note at: {max_end_all_instruments[idx]:.1f} beats",
            )

            print("-----------------------------")

    stats = dict(
        beat_count=beat_count,
        note_counts_all_instrument=note_counts_all_instrument,
        note_coverage_all_instrument=note_coverage_all_instrument,
        note_coverage_true_all_instrument=note_coverage_true_all_instrument,
        min_start_all_instruments=min_start_all_instruments,
        max_end_all_instruments=max_end_all_instruments,
    )
    print(stats)
    return stats


stats_on_track()

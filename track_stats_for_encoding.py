# classic python
import matplotlib.pyplot as plt
import numpy as np

# midi stuff
from miditoolkit import MidiFile
import mido
import pretty_midi
import note_seq
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile

# path
midi_filename = "the_strokes-reptilia"
path_midi = f"./midi/{midi_filename}.mid"

# MidiTok
# # Our parameters
# pitch_range = range(21, 109)
# beat_res = {(0, 4): 8, (4, 12): 4}
# nb_velocities = 32
# additional_tokens = {
#     "Chord": True,
#     "Rest": True,
#     "Tempo": True,
#     "Program": False,
#     "TimeSignature": False,
#     "rest_range": (2, 8),  # (half, 8 beats)
#     "nb_tempos": 32,  # nb of tempo bins
#     "tempo_range": (40, 250),
# }  # (min, max)

# # Creates the tokenizer and loads a MIDI
# tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
# midi = MidiFile(path_midi)

# # Converts MIDI to tokens, and back to a MIDI
# tokens = tokenizer.midi_to_tokens(midi)
# converted_back_midi = tokenizer.tokens_to_midi(tokens, get_midi_programs(midi))

# # Converts just a selected track
# tokenizer.current_midi_metadata = {
#     "time_division": midi.ticks_per_beat,
#     "tempo_changes": midi.tempo_changes,
# }
# piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

# # And convert it back (the last arg stands for (program number, is drum))
# converted_back_track, tempo_changes = tokenizer.tokens_to_track(
#     piano_tokens, midi.ticks_per_beat, (0, False)
# )


def stats_on_track(midi_filename="the_strokes-reptilia", verbose=True):
    path_midi = f"./midi/{midi_filename}.mid"

    miditoolk_data = MidiFile(path_midi)
    midi_mido = mido.MidiFile(path_midi)
    pretty_midi_data = pretty_midi.PrettyMIDI(path_midi)
    note_seq_data = note_seq.midi_file_to_note_sequence(path_midi)

    print("-----------------------------")
    print(
        f"miditooldkit instruments: {len(midi.instruments)}",
    )
    print(
        f"mido tracks: {len(midi_mido.tracks)}",
    )
    print("-----------------------------")
    # midi_mido.tracks
    # print(midi_mido.tracks)
    # print(midi.instruments)

    beat_count = miditoolk_data.max_tick / miditoolk_data.ticks_per_beat
    min_start_all_instruments = []
    max_end_all_instruments = []
    note_coverage_all_instrument = []
    note_coverage_true_all_instrument = []
    note_counts_all_instrument = []
    instrument_names = []
    for idx, instruments in enumerate(miditoolk_data.instruments):
        instrument_names.append(instruments.name)

        note_coverage_instrument = 0
        note_coverage_instrument_idx = []
        for i, note in enumerate(instruments.notes):
            if i == 0:
                min_start_all_instruments.append(
                    note.start / miditoolk_data.ticks_per_beat
                )

            note_coverage_instrument += note.end - note.start
            note_coverage_instrument_idx.append(list(range(note.start, note.end)))

        max_end_all_instruments.append(note.end / miditoolk_data.ticks_per_beat)

        note_coverage_all_instrument.append(
            100 * (note_coverage_instrument / miditoolk_data.max_tick)
        )
        unique_idx_list = []
        for idx_list in note_coverage_instrument_idx:
            [unique_idx_list.append(idx) for idx in idx_list]
        unique_idx_list = len(np.unique(unique_idx_list))

        note_coverage_true_all_instrument.append(
            100 * (unique_idx_list / miditoolk_data.max_tick)
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
                f"First note at: {min_start_all_instruments[idx]:.1f} beats;",
                f"Fast note at: {max_end_all_instruments[idx]:.1f} beats",
            )

            print("-----------------------------")

    stats = dict(
        track_count_in_mido=len(miditoolk_data_mido.tracks),
        instrument_count_in_miditooldkit=len(miditoolk_data.instruments),
        beat_count=beat_count,
        note_counts_all_instrument=note_counts_all_instrument,
        note_coverage_all_instrument=note_coverage_all_instrument,
        note_coverage_true_all_instrument=note_coverage_true_all_instrument,
        min_start_all_instruments=min_start_all_instruments,
        max_end_all_instruments=max_end_all_instruments,
    )
    print(stats)

    for ui in np.unique(instrument_names):
        all_notes_starts = []
        all_notes_instrument = []
        where_unique_inst = [
            idx for idx, ins in enumerate(instrument_names) if ins == ui
        ]
        if len(where_unique_inst) > 1:
            print(f"{ui} is split in instrument {where_unique_inst}")

            for i, t in enumerate(where_unique_inst):
                for notes in midi.instruments[t].notes:
                    all_notes_starts.append(notes.start)
                    all_notes_instrument.append(i)
            all_notes_starts = np.array(all_notes_starts)
            all_notes_instrument = np.array(all_notes_instrument)
            right_order = np.argsort(all_notes_starts)

            all_notes_starts_reordered = [all_notes_starts[id] for id in right_order]
            all_notes_instrument_reordered = [
                all_notes_instrument[id] for id in right_order
            ]

            fig, ax = plt.subplots()
            plt.plot(all_notes_starts, all_notes_instrument, "o")
            plt.plot(all_notes_starts_reordered, all_notes_instrument_reordered, "-")
            plt.show()
            # plt.close()
            all_notes_starts
    return stats


# stats_on_track(midi_filename="655db1f86bc729a4af2167a2412ec29e")
# stats_on_track(midi_filename="655a978c1de484a12e0ab2fd187d64f8")
# stats_on_track(midi_filename="655c39feef571bbe52e9270994d8e6c5")
stats_on_track()

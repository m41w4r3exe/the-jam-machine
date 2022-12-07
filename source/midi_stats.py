from joblib import Parallel, delayed
from pathlib import Path
import pandas as pd
from pretty_midi import program_to_instrument_name, PrettyMIDI

from utils import compute_list_average

# TODO : implement util function get_files (after merging with master)
# TODO : replace categorize_midi_instrument (after merging with master)
# TODO : add enrichment
# TODO : include types


def categorize_midi_instrument(program_number):
    if 0 <= program_number <= 7:
        return "Piano"
    elif 8 <= program_number <= 15:
        return "Chromatic Percussion"
    elif 16 <= program_number <= 23:
        return "Organ"
    elif 24 <= program_number <= 31:
        return "Guitar"
    elif 32 <= program_number <= 39:
        return "Bass"
    elif 40 <= program_number <= 47:
        return "Strings"
    elif 48 <= program_number <= 55:
        return "Ensemble"
    elif 56 <= program_number <= 63:
        return "Brass"
    elif 64 <= program_number <= 71:
        return "Reed"
    elif 72 <= program_number <= 79:
        return "Pipe"
    elif 80 <= program_number <= 87:
        return "Synth Lead"
    elif 88 <= program_number <= 95:
        return "Synth Pad"
    elif 96 <= program_number <= 103:
        return "Synth Effects"
    elif 104 <= program_number <= 111:
        return "Ethnic"
    elif 112 <= program_number <= 119:
        return "Percussive"
    elif 120 <= program_number <= 127:
        return "Sound Effects"


def track_name(midi_file):
    return Path(midi_file).stem


def n_instruments(pm):
    if pm.instruments:
        return len(pm.instruments)
    else:
        return None


def n_unique_instruments(pm):
    if pm.instruments:
        return len(set([instrument.program for instrument in pm.instruments]))
    else:
        return None


def instrument_names(pm):
    if pm.instruments:
        return [
            list(
                set(
                    [
                        program_to_instrument_name(instrument.program)
                        for instrument in pm.instruments
                    ]
                )
            )
        ]
    else:
        return None


def instrument_families(pm):
    if pm.instruments:
        return [
            list(
                set(
                    [
                        categorize_midi_instrument(instrument.program)
                        for instrument in pm.instruments
                    ]
                )
            )
        ]
    else:
        return None


def number_of_instrument_families(pm):
    if pm.instruments:
        return len(
            set(
                [
                    categorize_midi_instrument(instrument.program)
                    for instrument in pm.instruments
                ]
            )
        )
    else:
        return None


def number_of_notes(pm):
    if pm.instruments:
        return sum([len(instrument.notes) for instrument in pm.instruments])
    else:
        return None


def number_of_unique_notes(pm):
    if pm.instruments:
        return len(
            set(
                [
                    note.pitch
                    for instrument in pm.instruments
                    for note in instrument.notes
                ]
            )
        )
    else:
        return None


def avg_number_of_unique_notes_per_instrument(pm):
    if pm.instruments:
        return compute_list_average(
            [
                len(set([note.pitch for note in instrument.notes]))
                for instrument in pm.instruments
            ]
        )
    else:
        return None


def average_note_duration(pm):
    if pm.instruments:
        return compute_list_average(
            [
                note.end - note.start
                for instrument in pm.instruments
                for note in instrument.notes
            ]
        )
    else:
        return None


def average_note_velocity(pm):
    if pm.instruments:
        return compute_list_average(
            [
                note.velocity
                for instrument in pm.instruments
                for note in instrument.notes
            ]
        )
    else:
        return None


def average_note_pitch(pm):
    if pm.instruments:
        return compute_list_average(
            [note.pitch for instrument in pm.instruments for note in instrument.notes]
        )
    else:
        return None


def range_of_note_pitches(pm):
    if pm.instruments:
        return max(
            [note.pitch for instrument in pm.instruments for note in instrument.notes]
        ) - min(
            [note.pitch for instrument in pm.instruments for note in instrument.notes]
        )
    else:
        return None


def average_range_of_note_pitches_per_instrument(pm):
    if pm.instruments:
        return compute_list_average(
            [
                max([note.pitch for note in instrument.notes])
                - min([note.pitch for note in instrument.notes])
                for instrument in pm.instruments
            ]
        )
    else:
        return None


def number_of_note_pitch_classes(pm):
    if pm.instruments:
        return len(
            set(
                [
                    note.pitch % 12
                    for instrument in pm.instruments
                    for note in instrument.notes
                ]
            )
        )
    else:
        return None


def average_number_of_note_pitch_classes_per_instrument(pm):
    if pm.instruments:
        return compute_list_average(
            [
                len(set([note.pitch % 12 for note in instrument.notes]))
                for instrument in pm.instruments
            ]
        )
    else:
        return None


def number_of_octaves(pm):
    if pm.instruments:
        return len(
            set(
                [
                    note.pitch // 12
                    for instrument in pm.instruments
                    for note in instrument.notes
                ]
            )
        )
    else:
        return None


def average_number_of_octaves_per_instrument(pm):
    if pm.instruments:
        return compute_list_average(
            [
                len(set([note.pitch // 12 for note in instrument.notes]))
                for instrument in pm.instruments
            ]
        )
    else:
        return None


def number_of_notes_per_second(pm):
    if pm.instruments:
        return (
            len([note for instrument in pm.instruments for note in instrument.notes])
            / pm.get_end_time()
        )
    else:
        return None


def shortest_note_length(pm):
    if pm.instruments:
        return min(
            [
                note.end - note.start
                for instrument in pm.instruments
                for note in instrument.notes
            ]
        )
    else:
        return None


def longest_note_length(pm):
    if pm.instruments:
        return max(
            [
                note.end - note.start
                for instrument in pm.instruments
                for note in instrument.notes
            ]
        )
    else:
        return None


def main_key_signature(pm):
    if pm.key_signature_changes:
        return pm.key_signature_changes[0].key_number
    else:
        return None


def n_key_changes(pm):
    if pm.key_signature_changes:
        return len(pm.key_signature_changes)
    else:
        return None


def n_tempo_changes(pm):
    return len(pm.get_tempo_changes())


def average_tempo(pm):
    try:
        return round(pm.estimate_tempo())
    except Exception:
        return None


def tempo_changes(pm):
    return [[pm.get_tempo_changes()]]


def main_time_signature(pm):
    if pm.time_signature_changes:
        return [
            str(ts.numerator) + "/" + str(ts.denominator)
            for ts in pm.time_signature_changes
        ][0]
    else:
        return None


def n_time_signature_changes(pm):
    if pm.time_signature_changes:
        return len(pm.time_signature_changes)
    else:
        return None


def all_time_signatures(pm):
    if pm.time_signature_changes:
        return [
            [
                str(ts.numerator) + "/" + str(ts.denominator)
                for ts in pm.time_signature_changes
            ]
        ]
    else:
        return None


def four_to_the_floor(pm):
    if pm.time_signature_changes:
        time_signatures = [
            str(ts.numerator) + "/" + str(ts.denominator)
            for ts in pm.time_signature_changes
        ]
        # check if time_signatures contains exclusively '2/4' or '4/4'
        return (
            all([ts == "4/4" for ts in time_signatures]) and len(time_signatures) == 1
        )
    else:
        return None


def track_length_in_seconds(pm):
    return pm.get_end_time()


def lyrics_number_of_words(pm):
    if pm.lyrics:
        return len([l.text for l in pm.lyrics])
    else:
        return None


def lyrics_number_of_unique_words(pm):
    if pm.lyrics:
        return len(set([l.text for l in pm.lyrics]))
    else:
        return None


def lyrics_boolean(pm):
    if pm.lyrics:
        return True
    else:
        return False


class MidiStats:
    def single_file_statistics(self, midi_file):
        """
        Compute statistics for a single midi path object.
        """
        # Some Midi files are corrupted and cannot be parsed by PrettyMIDI
        try:
            pm = PrettyMIDI(str(midi_file))
        except Exception:
            return None

        # Compute statistics
        statistics = {
            # track md5 hash name without extension
            "md5": track_name(midi_file),
            # instruments
            "n_instruments": n_instruments(pm),
            "n_unique_instruments": n_unique_instruments(pm),
            "instrument_names": instrument_names(pm),
            "instrument_families": instrument_families(pm),
            "number_of_instrument_families": number_of_instrument_families(pm),
            # notes
            "n_notes": number_of_notes(pm),
            "n_unique_notes": number_of_unique_notes(pm),
            "average_n_unique_notes_per_instrument": avg_number_of_unique_notes_per_instrument(
                pm
            ),
            "average_note_duration": average_note_duration(pm),
            "average_note_velocity": average_note_velocity(pm),
            "average_note_pitch": average_note_pitch(pm),
            "range_of_note_pitches": range_of_note_pitches(pm),
            "average_range_of_note_pitches_per_instrument": average_range_of_note_pitches_per_instrument(
                pm
            ),
            "number_of_note_pitch_classes": number_of_note_pitch_classes(pm),
            "average_number_of_note_pitch_classes_per_instrument": average_number_of_note_pitch_classes_per_instrument(
                pm
            ),
            "number_of_octaves": number_of_octaves(pm),
            "average_number_of_octaves_per_instrument": average_number_of_octaves_per_instrument(
                pm
            ),
            "number_of_notes_per_second": number_of_notes_per_second(pm),
            "shortest_note_length": shortest_note_length(pm),
            "longest_note_length": longest_note_length(pm),
            # key signatures
            "main_key_signature": main_key_signature(pm),  # hacky
            "n_key_changes": n_key_changes(pm),
            # tempo
            "n_tempo_changes": n_tempo_changes(pm),
            "tempo_estimate": average_tempo(pm),  # hacky
            # time signatures
            "main_time_signature": main_time_signature(pm),  # hacky
            "all_time_signatures": all_time_signatures(pm),
            "four_to_the_floor": four_to_the_floor(pm),
            "n_time_signature_changes": n_time_signature_changes(pm),
            # track length
            "track_length_in_seconds": track_length_in_seconds(pm),
            # lyrics
            "lyrics_nb_words": lyrics_number_of_words(pm),
            "lyrics_unique_words": lyrics_number_of_unique_words(pm),
            "lyrics_bool": lyrics_boolean(pm),
        }
        return statistics

    def get_stats(self, input_directory, recursive=False, n_jobs=-1):
        """
        Compute statistics for a list of MIDI files
        """
        midi_files = get_files(input_directory, "mid", recursive)

        statistics = Parallel(n_jobs, verbose=1)(
            delayed(self.single_file_statistics)(midi_file) for midi_file in midi_files
        )

        # Remove None values, where statistics could not be computed
        return [s for s in statistics if s is not None]


def get_files(directory, extension, recursive=False):
    """
    Given a directory, get a list of the file paths of all files matching the
    specified file extension.
    directory: the directory to search as a Path object
    extension: the file extension to match as a string
    recursive: whether to search recursively in the directory or not
    """
    if recursive:
        return list(directory.rglob(f"*.{extension}"))
    else:
        return list(directory.glob(f"*.{extension}"))


if __name__ == "__main__":

    # Select the path to the MIDI files
    input_directory = Path("data/music_picks/electronic_artists").resolve()
    print(input_directory)

    # Select the path to save the statistics
    output_directory = Path("data/music_picks").resolve()

    # Compute statistics using parallel processing
    statistics = MidiStats().get_stats(input_directory, recursive=True)

    # export df to csv
    df = pd.DataFrame(statistics)
    df.to_csv(output_directory / "statistics.csv", index=False)

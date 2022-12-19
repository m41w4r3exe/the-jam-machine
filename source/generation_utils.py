import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from constants import INSTRUMENT_CLASSES
from playback import get_music, show_piano_roll

# matplotlib settings
matplotlib.use("Agg")  # for server
matplotlib.rcParams["xtick.major.size"] = 0
matplotlib.rcParams["ytick.major.size"] = 0
matplotlib.rcParams["axes.facecolor"] = "none"
matplotlib.rcParams["axes.edgecolor"] = "grey"


def define_generation_dir(model_repo_path):
    #### to remove later ####
    if model_repo_path == "models/model_2048_fake_wholedataset":
        model_repo_path = "misnaej/the-jam-machine"
    #### to remove later ####
    generated_sequence_files_path = f"midi/generated/{model_repo_path}"
    if not os.path.exists(generated_sequence_files_path):
        os.makedirs(generated_sequence_files_path)
    return generated_sequence_files_path


def bar_count_check(sequence, n_bars):
    """check if the sequence contains the right number of bars"""
    sequence = sequence.split(" ")
    # find occurences of "BAR_END" in a "sequence"
    # I don't check for "BAR_START" because it is not always included in "sequence"
    # e.g. BAR_START is included the prompt when generating one more bar
    bar_count = 0
    for seq in sequence:
        if seq == "BAR_END":
            bar_count += 1
    bar_count_matches = bar_count == n_bars
    if not bar_count_matches:
        print(f"Bar count is {bar_count} - but should be {n_bars}")
    return bar_count_matches, bar_count


def print_inst_classes(INSTRUMENT_CLASSES):
    """Print the instrument classes"""
    for classe in INSTRUMENT_CLASSES:
        print(f"{classe}")


def check_if_prompt_inst_in_tokenizer_vocab(tokenizer, inst_prompt_list):
    """Check if the prompt instrument are in the tokenizer vocab"""
    for inst in inst_prompt_list:
        if f"INST={inst}" not in tokenizer.vocab:
            instruments_in_dataset = np.sort(
                [tok.split("=")[-1] for tok in tokenizer.vocab if "INST" in tok]
            )
            print_inst_classes(INSTRUMENT_CLASSES)
            raise ValueError(
                f"""The instrument {inst} is not in the tokenizer vocabulary. 
                Available Instruments: {instruments_in_dataset}"""
            )


def forcing_bar_count(input_prompt, generated, bar_count, expected_length):
    """Forcing the generated sequence to have the expected length
    expected_length and bar_count refers to the length of newly_generated_only (without input prompt)"""

    if bar_count - expected_length > 0:  # Cut the sequence if too long
        full_piece = ""
        splited = generated.split("BAR_END ")
        for count, spl in enumerate(splited):
            if count < expected_length:
                full_piece += spl + "BAR_END "

        full_piece += "TRACK_END "
        full_piece = input_prompt + full_piece
        print(f"Generated sequence trunkated at {expected_length} bars")
        bar_count_checks = True

    elif bar_count - expected_length < 0:  # Do nothing it the sequence if too short
        full_piece = input_prompt + generated
        bar_count_checks = False
        print(f"--- Generated sequence is too short - Force Regeration ---")

    return full_piece, bar_count_checks


def get_max_time(inst_midi):
    max_time = 0
    for inst in inst_midi.instruments:
        max_time = max(max_time, inst.get_end_time())
    return max_time


def plot_piano_roll(inst_midi):
    piano_roll_fig = plt.figure(figsize=(25, 3 * len(inst_midi.instruments)))
    piano_roll_fig.tight_layout()
    piano_roll_fig.patch.set_alpha(0)
    inst_count = 0
    beats_per_bar = 4
    sec_per_beat = 0.5
    next_beat = max(inst_midi.get_beats()) + np.diff(inst_midi.get_beats())[0]
    bars_time = np.append(inst_midi.get_beats(), (next_beat))[::beats_per_bar].astype(
        int
    )
    for inst in inst_midi.instruments:
        inst_count += 1
        plt.subplot(len(inst_midi.instruments), 1, inst_count)

        for bar in bars_time:
            plt.axvline(bar, color="grey", linewidth=0.5)
        octaves = np.arange(0, 128, 12)
        for octave in octaves:
            plt.axhline(octave, color="grey", linewidth=0.5)
        plt.yticks(octaves, visible=False)

        p_midi_note_list = inst.notes
        note_time = []
        note_pitch = []
        for note in p_midi_note_list:
            note_time.append([note.start, note.end])
            note_pitch.append([note.pitch, note.pitch])
        note_pitch = np.array(note_pitch)
        note_time = np.array(note_time)

        plt.plot(
            note_time.T,
            note_pitch.T,
            color="purple",
            linewidth=4,
            solid_capstyle="butt",
        )
        plt.ylim(0, 128)
        xticks = np.array(bars_time)[:-1]
        plt.tight_layout()
        plt.xlim(min(bars_time), max(bars_time))
        plt.ylim(max([note_pitch.min() - 5, 0]), note_pitch.max() + 5)
        plt.xticks(
            xticks + 0.5 * beats_per_bar * sec_per_beat,
            labels=xticks.argsort() + 1,
            visible=False,
        )
        plt.title(inst.name, fontsize=10, color="white", verticalalignment="Top")

    return piano_roll_fig

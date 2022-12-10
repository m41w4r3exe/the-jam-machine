import os
import numpy as np
from constants import INSTRUMENT_CLASSES


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
    # find occurences of "BAR_START" in a str
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


def forcing_bar_length(input_prompt, newly_generated_only, bar_count, expected_length):
    """Forcing the generated sequence to have the expected length
    expected_length and bar_count refers to the length of newly_generated_only (without input prompt)"""

    if bar_count - expected_length > 0:  # Cut the sequence if too long
        generated_text = ""
        splited = newly_generated_only.split("BAR_END ")
        for count, spl in enumerate(splited):
            if count < expected_length:
                generated_text += spl + "BAR_END "

        generated_text += "TRACK_END"
        generated_text = input_prompt + generated_text
        print(f"Generated sequence trunkated at {expected_length} bars")

    elif bar_count - expected_length < 0:  # Do nothing it the sequence if too short
        generated_text = input_prompt + newly_generated_only

    bar_count_checks = True
    return generated_text, bar_count_checks

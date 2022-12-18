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

    # def create_hyperparameter_dictionary(self):
    #     self.hyperparameter_dictionary = {
    #         "model_identification": self.model.transformer.base_model.name_or_path,
    #         "max_seq_length": self.max_length,
    #         "generate_until": self.generate_until,
    #     }

    # def update_hyperparameter_dictionnary_bar(self, track, bar_index):
    #     # get the track instrument index to get the density and temperature TO FIX
    #     self.create_track_entry_in_hyperparameter_dict(track)
    #     # for (inst_idx, intrument) in enumerate(self.instruments):
    #     #     if intrument == self.hyperparameter_dictionary[track]["instruments"]:
    #     #         idx = inst_idx

    #     # self.hyperparameter_dictionary[track][f"bar_{bar_index}"] = {
    #     #     "density": self.densities[idx],
    #     #     "temperature": self.temperature[idx],
    #     #     "improv_level": self.no_repeat_ngram_size,
    #     # }

    # def update_hyperparameter_dictionnary__add_track(self, track, instrument):
    #     self.create_track_entry_in_hyperparameter_dict(track)
    #     self.hyperparameter_dictionary[track]["instruments"] = instrument

    # def reorder_tracks(self, order=None):  # TO BE TESTED
    #     if order is None:  # default order
    #         order = range(len(self.piece_dict.keys()))

    #     for count, track in enumerate(self.piece_dict.keys):
    #         inst = track.split("_")[-1]
    #         self.piece_dict[f"TRACK_{order[count]}_{inst}"] = self.piece_dict.pop(track)
    #         self.generated_piece_bar_by_bar_dict[
    #             f"TRACK_{order[count]}_{inst}"
    #         ] = self.generated_piece_bar_by_bar_dict.pop(track)

    # def create_track_entry_in_hyperparameter_dict(self, track):
    # if track not in self.hyperparameter_dictionary.keys():
    #     self.hyperparameter_dictionary[track] = {}

from utils import WriteTextMidiToFile, get_tokenizer
from generation_utils import (
    define_generation_dir,
    bar_count_check,
    print_inst_classes,
    check_if_prompt_inst_in_tokenizer_vocab,
    forcing_bar_length,
)
from load import LoadModel
from tqdm import tqdm
from constants import INSTRUMENT_CLASSES
import numpy as np

## import for execution
from decoder import TextDecoder


class GenerateMidiText:
    """Generating music with Class"""

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        temperature=0.75,
        force_sequence_length=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = model.config.n_positions
        print(
            f"Attention length set to {self.max_length} -> 'model.config.n_positions'"
        )
        self.temperature = temperature
        self.generate_until = "TRACK_END"
        self.force_sequence_length = force_sequence_length

    def tokenize_input_prompt(self, input_prompt, verbose=True):
        input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors="pt")
        if self.device == "cuda":  # TO CHECK - not sure if it works
            input_prompt_ids.cuda()
        if verbose:
            print("Tokenizing input_prompt...")

        return input_prompt_ids

    def generate_sequence_of_token_ids(
        self,
        input_prompt_ids,
        verbose=True,
    ):
        """
        generate a sequence of token ids based on input_prompt_ids
        The sequence length depends on the trained model (8 bars in our case)
        """
        generated_ids = self.model.generate(
            input_prompt_ids,
            max_length=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.encode(self.generate_until)[0],  # good
        )

        if verbose:
            print("Generating a token_id sequence...")

        return generated_ids

    def convert_ids_to_text(self, generated_ids, verbose=True):
        """converts the token_ids to text"""
        generated_text = self.tokenizer.decode(generated_ids[0])
        if verbose:
            print("Converting token sequence to MidiText...")
        return generated_text

    def generate_one_sequence(
        self,
        input_prompt="PIECE_START",
        inst=None,
        density=None,
        verbose=True,
        expected_length=8,
    ):
        """generate a sequence
        - input_prompt combined with inst and density parameters -> input_prompt
        - input_prompt is converted into input_prompt_ids
        - input_prompt_ids are passed to generate_sequence_of_token_ids for generation
        - the generated token_ids are then converted to text"""

        if inst is not None:
            input_prompt = f"{input_prompt} TRACK_START INST={str(inst)} "
            if density is not None:
                input_prompt = f"{input_prompt}DENSITY={str(density)} "

        if inst is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if verbose:
            print("--------------------")
            print(
                f"Generating {inst} - Density {density} - temperature {self.temperature}"
            )
        bar_count_checks = False

        while not bar_count_checks:  # regenerate until right length
            input_prompt_ids = self.tokenize_input_prompt(input_prompt)
            generated_tokens = self.generate_sequence_of_token_ids(input_prompt_ids)
            generated_text = self.convert_ids_to_text(generated_tokens)
            # bar_count_checks
            bar_count_checks, bar_count = bar_count_check(
                generated_text[len(input_prompt) :], expected_length
            )
            if not self.force_sequence_length:
                # set bar_count_checks to true to exist the while loop
                bar_count_checks = True

            if not bar_count_checks and self.force_sequence_length:
                # if the generated sequence is not the expected length
                generated_text, bar_count_checks = forcing_bar_length(
                    input_prompt,
                    generated_text[len(input_prompt) :],
                    bar_count,
                    expected_length,
                )

        return generated_text

    def generate_multi_track_sequence(
        self, inst_density_list=zip(["4", "0", "DRUMS"], [3, 2, 3])
    ):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list

        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        """
        generated_multi_track_dict = {}
        generated_multi_track_sequence = "PIECE_START"
        for count, (inst, density) in enumerate(inst_density_list):
            seq_len = len(generated_multi_track_sequence)
            generated_multi_track_sequence = self.generate_one_sequence(
                input_prompt=f"{generated_multi_track_sequence}",
                inst=inst,
                density=density,
            )
            if count > 0:  # not first iteration
                generated_track = generated_multi_track_sequence[seq_len + 1 :]
            else:
                generated_track = generated_multi_track_sequence

            generated_multi_track_dict[f"TRACK_{count}_INST={inst}"] = generated_track

        generate_features_dict = self.make_feature_dict(self, inst_density_list)

        return (
            generated_multi_track_sequence,
            generated_multi_track_dict,
            generate_features_dict,
        )

    def generate_n_more_bars(self, input_prompt, n_bars=8):
        """Generate n more bars from the input_prompt"""
        new_bars = ""
        for _ in range(n_bars):
            bar_count_matches = False
            while bar_count_matches is False:
                input_prompt, new_bar = self.generate_one_more_bar(self, input_prompt)
                bar_count_matches, _ = bar_count_check(new_bar, 1)
            new_bars += new_bar

        return new_bars

    @staticmethod
    def generate_one_more_bar(self, input_prompt):
        """Generate one more bar from the input_prompt"""
        processed_prompt = self.process_prompt_for_next_bar(input_prompt)
        prompt_plus_bar = self.generate_one_sequence(
            input_prompt=processed_prompt,
            expected_length=1,
        )
        # remove the processed_prompt - but keeping "BAR_START " - and the TRACK_END
        added_bar = prompt_plus_bar[
            len(processed_prompt) - len("BAR_START ") : -len("TRACK_END ")
        ]
        return prompt_plus_bar, added_bar

    @staticmethod
    def make_feature_dict(self, inst_density_list):
        return {
            "model_identification": self.model.transformer.base_model.name_or_path,
            "inst_density_list": list(inst_density_list),
            "temperature": self.temperature,
            "max_seq_length": self.max_length,
            "generate_until": self.generate_until,
        }

    @staticmethod
    def process_prompt_for_next_bar(input_prompt):
        """
        input_prompt should be at least a 8 bar sequence for one instrument
        """
        n_bars = 1
        input_prompt_split = input_prompt.split(" ")
        processed_prompt = ""
        bar_skipped = 0
        skipping_first_bar = False
        first_bar_skipped = False
        for token in input_prompt_split:
            # input_prompt_split[:-1] should exclude TRACK_END
            if first_bar_skipped == False:
                if token == "BAR_START":
                    skipping_first_bar = True

            if skipping_first_bar is True:
                if token != "BAR_END":
                    continue

                else:
                    bar_skipped += 1
                    if bar_skipped == n_bars:
                        skipping_first_bar = False
                        first_bar_skipped = True
                    continue

            if token == "TRACK_END":
                break
            processed_prompt += f"{token} "

        processed_prompt += "BAR_START "
        return processed_prompt


if __name__ == "__main__":

    print("=========================================")
    """" worker """
    DEVICE = "cpu"

    """" define generation parameters """
    USE_FAMILIARIZED_MODEL = False
    force_sequence_length = False
    temperature = 0.5
    if USE_FAMILIARIZED_MODEL:
        model_repo = "misnaej/the-jam-machine-elec-famil"
        instrument_promt_list = ["3", "DRUMS", "4", "5"]
        density_list = [3, 2, 3, 1]
    else:
        model_repo = "misnaej/the-jam-machine"
        instrument_promt_list = ["30", "DRUMS", "33", "51"]
        density_list = [3, 2, 3, 1]
    inst_density_list = zip(instrument_promt_list, density_list)

    """" define generation directory """
    generated_sequence_files_path = define_generation_dir(model_repo)

    """" load model and tokenizer """
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    """" check if the prompt makes sense"""
    check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

    """" instantiate the class for generation """
    genesis = GenerateMidiText(
        model,
        tokenizer,
        DEVICE,
        temperature=temperature,
        force_sequence_length=force_sequence_length,
    )
    """" generate a multi-track sequence """
    (
        generated_multi_track_sequence,
        generated_multi_track_dict,
        generate_features_dict,
    ) = genesis.generate_multi_track_sequence(
        inst_density_list=inst_density_list,
    )

    # """" Generate the next 8 bars """ TO DO
    # input_prompt = generated_multi_track_dict["INST=DRUMS"]
    # added_sequence = genesis.generate_n_more_bars(input_prompt, n_bars=8)
    # added_sequence = f"{input_prompt}{added_sequence}TRACK_END "
    # """" Write to JSON file """
    # WriteTextMidiToFile(
    #     added_sequence,
    #     generated_sequence_files_path,
    #     feature_dict=generate_features_dict,
    # ).text_midi_to_file()

    """ print the generated sequence in terminal"""
    print("=========================================")
    for inst in generated_multi_track_dict.items():
        print(inst)
    print("=========================================")

    """" write to JSON file """
    filename = WriteTextMidiToFile(
        generated_multi_track_sequence,
        generated_sequence_files_path,
        feature_dict=generate_features_dict,
    ).text_midi_to_file()

    """" decode the sequence to MIDI """
    tokenizer = get_tokenizer()
    TextDecoder(tokenizer).write_to_midi(
        generated_multi_track_sequence, filename=filename.split(".")[0]
    )
    print("Et voilà! Your MIDI file is ready! But don't expect too much...")

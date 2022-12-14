from utils import WriteTextMidiToFile, get_miditok
from generation_utils import (
    define_generation_dir,
    bar_count_check,
    check_if_prompt_inst_in_tokenizer_vocab,
    forcing_bar_length,
)
from load import LoadModel
from constants import INSTRUMENT_CLASSES
import numpy as np
from familizer import Familizer

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

    def set_temperature(self, temperature):
        self.temperature = temperature

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

    def generate_one_track(
        self,
        input_prompt="PIECE_START",
        instrument=None,
        density=None,
        verbose=True,
        expected_length=8,
    ):
        """generate a additional track:
        full_piece = input_prompt + generated"""
        if instrument is not None:
            input_prompt = f"{input_prompt} TRACK_START INST={str(instrument)} "
            if density is not None:
                input_prompt = f"{input_prompt}DENSITY={str(density)} "

        if instrument is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if verbose:
            print("--------------------")
            print(
                f"Generating {instrument} - Density {density} - temperature {self.temperature}"
            )
        bar_count_checks = False

        while not bar_count_checks:  # regenerate until right length
            input_prompt_ids = self.tokenize_input_prompt(input_prompt)
            generated_tokens = self.generate_sequence_of_token_ids(input_prompt_ids)
            full_piece = self.convert_ids_to_text(generated_tokens)
            generated = full_piece[len(input_prompt) :]
            # bar_count_checks
            bar_count_checks, bar_count = bar_count_check(generated, expected_length)
            if not self.force_sequence_length:
                # set bar_count_checks to true to exist the while loop
                bar_count_checks = True

            if not bar_count_checks and self.force_sequence_length:
                # if the generated sequence is not the expected length
                full_piece, bar_count_checks = forcing_bar_length(
                    input_prompt,
                    generated,
                    bar_count,
                    expected_length,
                )

        return full_piece

    def generate_piece(self, inst_list=["4", "0", "DRUMS"], density_list=[1, 2, 1]):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list
        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        'generated_piece' keeps track of the entire piece
        'generated_piece' is returned by self.generate_one_track
        # it is returned by self.generate_one_track"""

        self.generated_piece_dict = {}
        generated_piece = "PIECE_START"

        for count, (instrument, density) in enumerate(zip(inst_list, density_list)):
            generated_piece = self.generate_one_track(
                input_prompt=generated_piece,
                instrument=instrument,
                density=density,
            )
            last_track = "TRACK_START " + generated_piece.split("TRACK_START")[-1]
            self.generated_piece_dict[f"TRACK_{count}_INST={instrument}"] = last_track

        self.hyperparameter_dict = self.create_hyperparameter_dictionary(
            self, inst_list, density_list
        )
        self.generated_piece = generated_piece
        return generated_piece

    def wrapping_piece_and_hyperparams():
        pass

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
        prompt_plus_bar = self.generate_one_track(
            input_prompt=processed_prompt,
            expected_length=1,
        )
        # remove the processed_prompt - but keeping "BAR_START " - and the TRACK_END
        added_bar = prompt_plus_bar[
            len(processed_prompt) - len("BAR_START ") : -len("TRACK_END ")
        ]
        return prompt_plus_bar, added_bar

    @staticmethod
    def create_hyperparameter_dictionary(self, inst_list, density_list):
        return {
            "model_identification": self.model.transformer.base_model.name_or_path,
            "inst_list": inst_list,
            "density_list": density_list,
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

    # worker
    DEVICE = "cpu"

    # define generation parameters
    N_FILES_TO_GENERATE = 1
    Temperatures_to_try = [0.7, 0.75]

    USE_FAMILIZED_MODEL = False
    force_sequence_length = True

    if USE_FAMILIZED_MODEL:
        # model_repo = "misnaej/the-jam-machine-elec-famil"
        # model_repo = "misnaej/the-jam-machine-elec-famil-ft32"
        model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
        instrument_promt_list = ['3', '4']
        # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
        density_list = [4, 1]
    else:
        model_repo = "misnaej/the-jam-machine"
        instrument_promt_list = ["30", "DRUMS", "33"]
        density_list = [2, 2, 2]

    # define generation directory
    generated_sequence_files_path = define_generation_dir(model_repo)

    # load model and tokenizer
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    # does the prompt make sense
    check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

    for temperature in Temperatures_to_try:
        print(f"================= TEMPERATURE {temperature} =======================")
        for _ in range(N_FILES_TO_GENERATE):
            print(f"========================================")
            # 1 - instantiate
            generate_midi = GenerateMidiText(
                model,
                tokenizer,
                DEVICE,
                temperature=temperature,
                force_sequence_length=force_sequence_length,
            )
            # 2- generate the first 8 bars for each instrument
            generated_piece = generate_midi.generate_piece(
                inst_list=instrument_promt_list,
                density_list=density_list,
            )
            # 3 - generate the next 8 bars for each instrument
            # TO IMPROVE
            # input_prompt = generated_piece_dict["INST=DRUMS"]
            # added_sequence = generate_midi.generate_n_more_bars(input_prompt, n_bars=8)
            # added_sequence = f"{input_prompt}{added_sequence}TRACK_END "
            # """" Write to JSON file """
            # WriteTextMidiToFile(
            #     added_sequence,
            #     generated_sequence_files_path,
            #     hyperparameter_dict=hyperparameter_dict,
            # ).text_midi_to_file()

            # print the generated sequence in terminal
            print("=========================================")
            for inst in generate_midi.generated_piece_dict.items():
                print(inst)
            print("=========================================")

            # write to JSON file
            filename = WriteTextMidiToFile(
                generate_midi,
                generated_sequence_files_path,
            ).text_midi_to_file()

            # decode the sequence to MIDI """
            decode_tokenizer = get_miditok()
            TextDecoder(decode_tokenizer, USE_FAMILIZED_MODEL).get_midi(
                generated_piece, filename=filename.split(".")[0] + '.mid'
            )
            print("Et voilà! Your MIDI file is ready! But don't expect too much...")

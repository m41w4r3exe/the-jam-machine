from utils import WriteTextMidiToFile, get_tokenizer
from generation_utils import *
from utils import WriteTextMidiToFile, get_miditok
from load import LoadModel
from constants import INSTRUMENT_CLASSES

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
        self.generated_piece_dict = {}
        self.generated_piece_bar_by_bar_dict = {}
        self.set_nb_bars_generated()

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_nb_bars_generated(self, n_bars=8):  # default is a 8 bar model
        self.model_n_bar = n_bars

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
        The sequence length depends on the trained model (self.model_n_bar)
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
        expected_length=None,
    ):
        if expected_length is None:
            expected_length = self.model_n_bar

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
            input_prompt_ids = self.tokenize_input_prompt(input_prompt, verbose=verbose)
            generated_tokens = self.generate_sequence_of_token_ids(
                input_prompt_ids, verbose=verbose
            )
            full_piece = self.convert_ids_to_text(generated_tokens, verbose=verbose)
            generated = full_piece[len(input_prompt) :]
            # bar_count_checks
            bar_count_checks, bar_count = bar_count_check(generated, expected_length)
            if not self.force_sequence_length:
                # set bar_count_checks to true to exist the while loop
                bar_count_checks = True

            if not bar_count_checks and self.force_sequence_length:
                # if the generated sequence is not the expected length
                full_piece, bar_count_checks = forcing_bar_count(
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

        generated_piece = "PIECE_START"
        for count, (instrument, density) in enumerate(zip(inst_list, density_list)):
            generated_piece = self.generate_one_track(
                input_prompt=generated_piece,
                instrument=instrument,
                density=density,
            )
            track_id = f"TRACK_{count}_INST={instrument}"
            last_track = "TRACK_START" + generated_piece.split("TRACK_START")[-1]
            self.generated_piece_dict[track_id] = last_track
            self.track_to_bar_dict(track_id)

        self.hyperparameter_dict = self.create_hyperparameter_dictionary(
            self, inst_list, density_list
        )
        self.generated_piece = generated_piece
        return generated_piece

    def wrapping_piece_and_hyperparams():
        pass

    def generate_n_more_bars(self, n_bars, verbose=True):
        """Generate n more bars from the input_prompt"""
        print(f"================== ")
        print(f"Adding {n_bars} more bars to the piece ")
        for bar_id in range(n_bars):
            print(f"----- Extra bar #{bar_id+1}")
            for track_key in sorted(self.generated_piece_dict.keys()):
                print(f"---- ----- {track_key}")
                # self.generated_piece_dict[f"{track}_new_bars"] = ""
                bar_count_matches = False
                while bar_count_matches is False:
                    input_prompt = self.process_prompt_for_next_bar(self, track_key)
                    input_prompt, new_bar = self.generate_one_more_bar(input_prompt)
                    bar_count_matches, _ = bar_count_check(new_bar, 1)
                self.add_new_bar_to_dict(self, track_key, new_bar)

    @staticmethod
    def add_new_bar_to_dict(self, track_key, new_bar):
        max_index = self.generated_piece_bar_by_bar_dict[track_key]["max_bar_index"]
        self.generated_piece_bar_by_bar_dict[track_key][f"bar_{max_index+1}"] = new_bar
        self.generated_piece_bar_by_bar_dict[track_key]["max_bar_index"] += 1
        self.generated_piece_dict[track_key] += new_bar

    def track_to_bar_dict(self, track):
        self.generated_piece_bar_by_bar_dict[track] = {}
        for index, bar in enumerate(
            self.generated_piece_dict[track].split("BAR_START ")
        ):
            if index == 0:
                self.generated_piece_bar_by_bar_dict[track][f"track_init"] = bar
            elif index < len(self.generated_piece_dict[track].split("BAR_START ")) - 1:
                self.generated_piece_bar_by_bar_dict[track][
                    f"bar_{index-1}"
                ] = f"BAR_START {bar}"
            else:
                self.generated_piece_bar_by_bar_dict[track][
                    f"bar_{index-1}"
                ] = f"BAR_START {bar}".strip("TRACK_END")
        self.generated_piece_bar_by_bar_dict[track]["max_bar_index"] = index - 1

    def bar_dict_to_text(self):
        text = ""
        for track in self.generated_piece_bar_by_bar_dict.keys():
            max_bar_index = self.generated_piece_bar_by_bar_dict[track]["max_bar_index"]
            text += self.generated_piece_bar_by_bar_dict[track][f"track_init"]
            for bar in range(max_bar_index + 1):
                text += self.generated_piece_bar_by_bar_dict[track][f"bar_{bar}"]

            text += "TRACK_END "

        return text

    def delete_one_track(self, track):
        self.generated_piece_dict.pop(track)

    def reorder_tracks(self, order=None):
        if order is None:  # default order
            order = range(len(self.generated_piece_dict.keys))

        for count, track in enumerate(self.generated_piece_dict.keys):
            inst = track.split("_")[-1]
            self.generated_piece_dict[
                f"TRACK_{order[count]}_{inst}"
            ] = self.generated_piece_dict.pop(track)

    def generate_one_more_bar(self, processed_prompt):
        """Generate one more bar from the input_prompt"""
        prompt_plus_bar = self.generate_one_track(
            input_prompt=processed_prompt,
            expected_length=1,
            verbose=False,
        )
        # remove the processed_prompt - but keeping "BAR_START " - and the TRACK_END
        added_bar = prompt_plus_bar[
            len(processed_prompt) - len("BAR_START ") : -len("TRACK_END")
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
    def process_prompt_for_next_bar(self, track_key):
        # preprompt: other tracks if already with + 1 bar
        # bar_index starts at 1 not 0 ; bar_0 is the track initialisation
        track_max_bar = self.generated_piece_bar_by_bar_dict[track_key]["max_bar_index"]

        pre_promt = ""
        processed_prompt = self.generated_piece_bar_by_bar_dict[track_key]["track_init"]

        for (
            current_track_key,
            current_track,
        ) in self.generated_piece_bar_by_bar_dict.items():
            if current_track_key != track_key:
                # if another track is longer it means that one bar was already added there
                # so it should be included in the prompt
                # iter: keep only the last (self.model_n_bar) bars
                if current_track["max_bar_index"] > track_max_bar:
                    pre_promt += current_track["track_init"]
                    iter = range(current_track["max_bar_index"] + 1)[
                        -(self.model_n_bar) :
                    ]
                    for bar in iter:
                        pre_promt += current_track[f"bar_{bar}"]

                    pre_promt += "TRACK_END "

            elif current_track_key == track_key:
                # iterc: keep only the last (self.model_n_bar - 1) bars
                iterc = range(track_max_bar + 1)[-(self.model_n_bar - 1) :]
                for bar in iterc:
                    processed_prompt += current_track[f"bar_{bar}"]
                processed_prompt += "BAR_START "

        return pre_promt + processed_prompt


if __name__ == "__main__":

    # worker
    DEVICE = "cpu"

    # define generation parameters
    N_FILES_TO_GENERATE = 4
    Temperatures_to_try = [0.75, 0.85]

    USE_FAMILIZED_MODEL = True
    force_sequence_length = True

    if USE_FAMILIZED_MODEL:
        # model_repo = "misnaej/the-jam-machine-elec-famil"
        # model_repo = "misnaej/the-jam-machine-elec-famil-ft32"
        model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
        instrument_promt_list = ["DRUMS", "3", "4", "6"]
        # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
        density_list = [2, 1, 2, 3]
    else:
        model_repo = "misnaej/the-jam-machine"
        instrument_promt_list = ["30", "DRUMS", "0", "83"]
        density_list = [3, 2, 3, 3]
        pass

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
            # input_prompt = generate_midi.generated_piece_dict["INST=DRUMS"]
            generate_midi.generate_n_more_bars(
                generate_midi.model_n_bar
            )  # let's double the length
            generate_midi.generated_piece = generate_midi.bar_dict_to_text()

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
                generate_midi.generated_piece, filename=filename.split(".")[0] + ".mid"
            )
            print("Et voilà! Your MIDI file is ready! But don't expect too much...")


"""TO DO
- add errror if density is not in tokenizer vocab -> TODO
- add a function to delete a track -> TO TEST
- add a function to reorder the tracks in a dictionary -> TO TEST
"""

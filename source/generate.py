from generation_utils import *
from utils import WriteTextMidiToFile, get_miditok
from load import LoadModel
from constants import INSTRUMENT_CLASSES

## import for execution
from decoder import TextDecoder


class GenerateMidiText:
    """Generating music with Class"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # default initialization
        self.initialize_default_parameters()
        self.initialize_dictionaries()

    """Setters"""

    def initialize_default_parameters(self):
        self.set_device()
        self.set_attention_length()
        self.generate_until = "TRACK_END"
        self.set_force_sequence_lenth()
        self.set_nb_bars_generated()
        self.set_intruments()
        self.set_densities()
        self.set_improvisation_level(0)
        self.set_temperatures()

    def initialize_dictionaries(self):
        self.piece_dict = {}
        self.generated_piece_bar_by_bar_dict = {}
        self.create_hyperparameter_dictionary()

    def set_device(self, device="cpu"):
        self.device = ("cpu",)

    def set_attention_length(self):
        self.max_length = self.model.config.n_positions
        print(
            f"Attention length set to {self.max_length} -> 'model.config.n_positions'"
        )

    def set_force_sequence_lenth(self, force_sequence_length=True):
        self.force_sequence_length = force_sequence_length

    def set_improvisation_level(self, improvisation_value):
        self.no_repeat_ngram_size = improvisation_value
        print("--------------------")
        print(f"no_repeat_ngram_size set to {improvisation_value}")
        print("--------------------")

    def set_intruments(self, instruments=["DRUMS", "4", "0", "3"]):
        self.instruments = instruments

    def set_densities(self, densities=[3, 2, 1, 2]):
        self.densities = densities

    def set_temperatures(self, temperature=0.75):
        if type(temperature) is not list:
            self.temperature = [temperature for _ in self.instruments]
        else:
            if len(temperature) == 1:
                self.temperature = [temperature[0] for _ in self.instruments]
            elif len(temperature) == len(self.instruments):
                self.temperature = temperature
            else:
                ValueError(
                    "temperature list must be of length 1 or the same length as the number of instruments"
                )

    def set_nb_bars_generated(self, n_bars=8):  # default is a 8 bar model
        self.model_n_bar = n_bars

    """ Generation Tools - Dictionnaries """

    def update_bar_dict__add_track(self, track):
        self.generated_piece_bar_by_bar_dict[track] = {}
        for index, bar in enumerate(self.piece_dict[track].split("BAR_START ")):
            if index == 0:
                dict_entry = f"track_init"
                self.generated_piece_bar_by_bar_dict[track][dict_entry] = bar
            elif index < len(self.piece_dict[track].split("BAR_START ")) - 1:
                dict_entry = f"bar_{index-1}"
                self.generated_piece_bar_by_bar_dict[track][
                    dict_entry
                ] = f"BAR_START {bar}"
            else:
                dict_entry = f"bar_{index-1}"
                self.generated_piece_bar_by_bar_dict[track][
                    dict_entry
                ] = f"BAR_START {bar}".strip("TRACK_END")

            self.update_hyperparameter_dictionnary_bar(
                track,
                dict_entry,
            )
        self.generated_piece_bar_by_bar_dict[track]["max_bar_index"] = index - 1

    @staticmethod
    def update_bar_dict__add_one_bar(self, track_key, new_bar):
        max_index = self.generated_piece_bar_by_bar_dict[track_key]["max_bar_index"]
        self.generated_piece_bar_by_bar_dict[track_key][f"bar_{max_index+1}"] = new_bar
        self.generated_piece_bar_by_bar_dict[track_key]["max_bar_index"] += 1
        self.piece_dict[track_key] += new_bar

    def bar_dict_to_text(self):
        text = ""
        for track in self.generated_piece_bar_by_bar_dict.keys():
            max_bar_index = self.generated_piece_bar_by_bar_dict[track]["max_bar_index"]
            text += self.generated_piece_bar_by_bar_dict[track][f"track_init"]
            for bar in range(max_bar_index + 1):
                text += self.generated_piece_bar_by_bar_dict[track][f"bar_{bar}"]

            text += "TRACK_END "

        return text

    def delete_one_track(self, track):  # TO BE TESTED
        self.piece_dict.pop(track)
        self.generated_piece_bar_by_bar_dict.pop(track)

    def reorder_tracks(self, order=None):  # TO BE TESTED
        if order is None:  # default order
            order = range(len(self.piece_dict.keys))

        for count, track in enumerate(self.piece_dict.keys):
            inst = track.split("_")[-1]
            self.piece_dict[f"TRACK_{order[count]}_{inst}"] = self.piece_dict.pop(track)
            self.generated_piece_bar_by_bar_dict[
                f"TRACK_{order[count]}_{inst}"
            ] = self.generated_piece_bar_by_bar_dict.pop(track)

    def create_hyperparameter_dictionary(self):
        self.hyperparameter_dictionary = {
            "model_identification": self.model.transformer.base_model.name_or_path,
            "max_seq_length": self.max_length,
            "generate_until": self.generate_until,
        }

    def update_hyperparameter_dictionnary_bar(self, track, bar_index):
        # get the track instrument index to get the density and temperature TO FIX
        self.create_track_entry_in_hyperparameter_dict(track)
        # for (inst_idx, intrument) in enumerate(self.instruments):
        #     if intrument == self.hyperparameter_dictionary[track]["instruments"]:
        #         idx = inst_idx

        # self.hyperparameter_dictionary[track][f"bar_{bar_index}"] = {
        #     "density": self.densities[idx],
        #     "temperature": self.temperature[idx],
        #     "improv_level": self.no_repeat_ngram_size,
        # }

    def update_hyperparameter_dictionnary__add_track(self, track, instrument):
        self.create_track_entry_in_hyperparameter_dict(track)
        self.hyperparameter_dictionary[track]["instruments"] = instrument

    def update_piece_dict__add_track(self, track_id, track):
        self.piece_dict[track_id] = track

    def create_track_entry_in_hyperparameter_dict(self, track):
        if track not in self.hyperparameter_dictionary.keys():
            self.hyperparameter_dictionary[track] = {}

    def update_all_dictionnaries__add_track(self, instrument, track_id, track):
        self.update_hyperparameter_dictionnary__add_track(track_id, instrument)
        self.update_piece_dict__add_track(track_id, track)
        self.update_bar_dict__add_track(track_id)

    def wrapping_piece_and_hyperparams():
        pass

    """Basic generation tools"""

    def tokenize_input_prompt(self, input_prompt, verbose=True):
        """Tokenizing prompt

        Args:
        - input_prompt (str): prompt to tokenize

        Returns:
        - input_prompt_ids (torch.tensor): tokenized prompt
        """
        if verbose:
            print("Tokenizing input_prompt...")

        return self.tokenizer.encode(input_prompt, return_tensors="pt")

    def generate_sequence_of_token_ids(
        self,
        input_prompt_ids,
        temperature,
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
            temperature=temperature,
            no_repeat_ngram_size=self.no_repeat_ngram_size,  # default = 0
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

    def get_new_track_id(self, instrument):
        track_id = len(self.generated_piece_bar_by_bar_dict)
        return f"TRACK_{track_id}_INST={instrument}"

    def get_last_generated_track(self, full_piece):
        track = "TRACK_START" + full_piece.split("TRACK_START")[-1]
        return track

    def generate_one_track(
        self,
        input_prompt="PIECE_START",
        instrument=None,
        density=None,
        temperature=None,
        verbose=True,
        expected_length=None,
    ):

        """generate a additional track:
        full_piece = input_prompt + generated"""
        if expected_length is None:
            expected_length = self.model_n_bar

        if instrument is not None:
            input_prompt = f"{input_prompt} TRACK_START INST={str(instrument)} "
            if density is not None:
                input_prompt = f"{input_prompt}DENSITY={str(density)} "

        if instrument is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if temperature is None:
            temperature = self.temperature[0]

        if verbose:
            print("--------------------")
            print(
                f"Generating {instrument} - Density {density} - temperature {temperature}"
            )
        bar_count_checks = False

        while not bar_count_checks:  # regenerate until right length
            input_prompt_ids = self.tokenize_input_prompt(input_prompt, verbose=verbose)
            generated_tokens = self.generate_sequence_of_token_ids(
                input_prompt_ids, temperature, verbose=verbose
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

            track_id = self.get_new_track_id(instrument)
            track = self.get_last_generated_track(full_piece)
            self.update_all_dictionnaries__add_track(instrument, track_id, track)

        return full_piece

    """ Piece generation - Basics """

    def generate_piece(self):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list
        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        'generated_piece' keeps track of the entire piece
        'generated_piece' is returned by self.generate_one_track
        # it is returned by self.generate_one_track"""

        generated_piece = "PIECE_START"
        for count, (instrument, density, temperature) in enumerate(
            zip(self.instruments, self.densities, self.temperature)
        ):
            generated_piece = self.generate_one_track(
                input_prompt=generated_piece,
                instrument=instrument,
                density=density,
                temperature=temperature,
            )

        return generated_piece

    """ Piece generation - Extra Bars """

    @staticmethod
    def process_prompt_for_next_bar(self, track_key):
        """Processing the prompt for the model to generate one more bar only.
        The prompt containts:
                if not the first bar: the previous, already processed, bars of the track
                the bar initialization (ex: "TRACK_START INST=DRUMS DENSITY=2 ")
                the last (self.model_n_bar)-1 bars of the track
        Args:
            track_key: the dictionnary of the track to be processed

        Returns:
            the processed prompt for generating the next bar
        """
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
                # iterc: keep only the last (self.model_n_bar - 2) bars
                iterc = range(track_max_bar + 1)[-(self.model_n_bar - 1) :]
                for bar in iterc:
                    processed_prompt += current_track[f"bar_{bar}"]
                processed_prompt += "BAR_START "

        return pre_promt + processed_prompt

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

    def generate_n_more_bars(self, n_bars, verbose=True):
        """Generate n more bars from the input_prompt"""
        print(f"================== ")
        print(f"Adding {n_bars} more bars to the piece ")
        for bar_id in range(n_bars):
            print(f"----- Extra bar #{bar_id+1}")
            for track_key in sorted(self.piece_dict.keys()):
                print(f"---- ----- {track_key}")
                # self.piece_dict[f"{track}_new_bars"] = ""
                bar_count_matches = False
                while bar_count_matches is False:
                    input_prompt = self.process_prompt_for_next_bar(self, track_key)
                    input_prompt, new_bar = self.generate_one_more_bar(input_prompt)
                    bar_count_matches, _ = bar_count_check(new_bar, 1)
                self.update_bar_dict__add_one_bar(self, track_key, new_bar)


if __name__ == "__main__":

    # worker
    DEVICE = "cpu"

    # define generation parameters
    N_FILES_TO_GENERATE = 4
    Temperatures_to_try = [0.75]

    USE_FAMILIZED_MODEL = True
    force_sequence_length = True

    if USE_FAMILIZED_MODEL:
        # model_repo = "misnaej/the-jam-machine-elec-famil"
        # model_repo = "misnaej/the-jam-machine-elec-famil-ft32"

        model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
        n_bar_generated = 8

        # model_repo = "JammyMachina/improved_4bars-mdl"
        # n_bar_generated = 4
        instrument_promt_list = ["4", "DRUMS", "3", "0"]
        # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
        density_list = [3, 3, 2, 3]
        # temperature_list = [0.7, 0.7, 0.75, 0.75]
    else:
        model_repo = "misnaej/the-jam-machine"
        instrument_promt_list = ["30", "DRUMS", "0", "83"]
        density_list = [3, 2, 3, 3]
        # temperature_list = [0.7, 0.5, 0.75, 0.75]
        pass

    # define generation directory
    generated_sequence_files_path = define_generation_dir(model_repo)

    # load model and tokenizer
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    # does the prompt make sense
    check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

    for temperature_list in Temperatures_to_try:
        print(
            f"================= TEMPERATURE {temperature_list} ======================="
        )
        for _ in range(N_FILES_TO_GENERATE):
            print(f"========================================")
            # 1 - instantiate
            generate_midi = GenerateMidiText(model, tokenizer)
            # 0 - set the n_bar for this model
            generate_midi.set_nb_bars_generated(n_bars=n_bar_generated)
            # 1 - defines the instruments, densities and temperatures
            generate_midi.set_intruments(instrument_promt_list)
            generate_midi.set_densities(density_list)
            generate_midi.set_temperatures(temperature_list)
            generate_midi.set_improvisation_level(0)
            # 2- generate the first 8 bars for each instrument
            generate_midi.generate_piece()
            # 3 - force the model to improvise
            generate_midi.set_improvisation_level(0)
            # 4 - generate the next 4 bars for each instrument
            generate_midi.generate_n_more_bars(4)
            # 5 - lower the improvisation level
            generate_midi.set_improvisation_level(0)
            # 6 - generate 8 more bars the improvisation level
            generate_midi.generate_n_more_bars(8)
            generate_midi.generated_piece = generate_midi.bar_dict_to_text()

            # print the generated sequence in terminal
            print("=========================================")
            for inst in generate_midi.piece_dict.items():
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
            print("Et voilà! Your MIDI file is ready! GO JAM!")


"""
- TODO: add improvisation level in bar dictionnary
- TODO: update hyperparameters dictionnary when adding new bars
- TODO: add errror if density is not in tokenizer vocab
- TODO: add a function to delete a track -> TO TEST
- TODO: add a function to reorder the tracks in a dictionary -> TO TEST
"""

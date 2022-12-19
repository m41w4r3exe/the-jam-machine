from generation_utils import *
from utils import WriteTextMidiToFile, get_miditok
from load import LoadModel
from constants import INSTRUMENT_CLASSES

## import for execution
from decoder import TextDecoder
from playback import get_music, show_piano_roll


class GenerateMidiText:
    """Generating music with Class

    LOGIC:

    FOR GENERATING FROM SCRATCH:
    - self.generate_one_new_track()
    it calls
        - self.generate_until_track_end()

    FOR GENERATING NEW BARS:
    - self.generate_one_more_bar()
    it calls
        - self.process_prompt_for_next_bar()
        - self.generate_until_track_end()"""

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
        self.set_improvisation_level(0)

    def initialize_dictionaries(self):
        self.piece_by_track = []

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

    def reset_temperatures(self, track_id, temperature):
        self.piece_by_track[track_id]["temperature"] = temperature

    def set_nb_bars_generated(self, n_bars=8):  # default is a 8 bar model
        self.model_n_bar = n_bars

    """ Generation Tools - Dictionnaries """

    def initiate_track_dict(self, instr, density, temperature):
        label = len(self.piece_by_track)
        self.piece_by_track.append(
            {
                "label": f"track_{label}",
                "instrument": instr,
                "density": density,
                "temperature": temperature,
                "bars": [],
            }
        )

    def update_track_dict__add_bars(self, bars, track_id):
        """Add bars to the track dictionnary"""
        for bar in self.striping_track_ends(bars).split("BAR_START "):
            if bar == "":  # happens is there is one bar only
                continue
            else:
                if "TRACK_START" in bar:
                    self.piece_by_track[track_id]["bars"].append(bar)
                else:
                    self.piece_by_track[track_id]["bars"].append("BAR_START " + bar)

    def get_all_instr_bars(self, track_id):
        return self.piece_by_track[track_id]["bars"]

    def striping_track_ends(self, text):
        if "TRACK_END" in text:
            # first get rid of extra space if any
            # then gets rid of "TRACK_END"
            text = text.rstrip(" ").rstrip("TRACK_END")
        return text

    def get_last_generated_track(self, full_piece):
        track = (
            "TRACK_START "
            + self.striping_track_ends(full_piece.split("TRACK_START ")[-1])
            + "TRACK_END "
        )  # forcing the space after track and
        return track

    def get_selected_track_as_text(self, track_id):
        text = ""
        for bar in self.piece_by_track[track_id]["bars"]:
            text += bar
        text += "TRACK_END "
        return text

    @staticmethod
    def get_newly_generated_text(input_prompt, full_piece):
        return full_piece[len(input_prompt) :]

    def get_whole_piece_from_bar_dict(self):
        text = "PIECE_START "
        for track_id, _ in enumerate(self.piece_by_track):
            text += self.get_selected_track_as_text(track_id)
        return text

    def delete_one_track(self, track):  # TO BE TESTED
        self.piece_by_track.pop(track)

    # def update_piece_dict__add_track(self, track_id, track):
    #     self.piece_dict[track_id] = track

    # def update_all_dictionnaries__add_track(self, track):
    # self.update_piece_dict__add_track(track_id, track)

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

    def generate_until_track_end(
        self,
        input_prompt="PIECE_START ",
        instrument=None,
        density=None,
        temperature=None,
        verbose=True,
        expected_length=None,
    ):

        """generate until the TRACK_END token is reached
        full_piece = input_prompt + generated"""
        if expected_length is None:
            expected_length = self.model_n_bar

        if instrument is not None:
            input_prompt = f"{input_prompt}TRACK_START INST={str(instrument)} "
            if density is not None:
                input_prompt = f"{input_prompt}DENSITY={str(density)} "

        if instrument is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if temperature is None:
            ValueError("Temperature must be defined")

        if verbose:
            print("--------------------")
            print(
                f"Generating {instrument} - Density {density} - temperature {temperature}"
            )
        bar_count_checks = False
        failed = 0
        while not bar_count_checks:  # regenerate until right length
            input_prompt_ids = self.tokenize_input_prompt(input_prompt, verbose=verbose)
            generated_tokens = self.generate_sequence_of_token_ids(
                input_prompt_ids, temperature, verbose=verbose
            )
            full_piece = self.convert_ids_to_text(generated_tokens, verbose=verbose)
            generated = self.get_newly_generated_text(input_prompt, full_piece)
            # bar_count_checks
            bar_count_checks, bar_count = bar_count_check(generated, expected_length)

            if not self.force_sequence_length:
                # set bar_count_checks to true to exist the while loop
                bar_count_checks = True

            if not bar_count_checks and self.force_sequence_length:
                # if the generated sequence is not the expected length
                if failed > 1:
                    full_piece, bar_count_checks = forcing_bar_count(
                        input_prompt,
                        generated,
                        bar_count,
                        expected_length,
                    )
                else:
                    print('"--- Wrong length - Regenerating ---')
            if not bar_count_checks:
                failed += 1
                if failed > 2:
                    bar_count_checks = True  # TOFIX exit the while loop

        return full_piece

    def generate_one_new_track(
        self,
        instrument,
        density,
        temperature,
        input_prompt="PIECE_START ",
    ):
        self.initiate_track_dict(instrument, density, temperature)
        full_piece = self.generate_until_track_end(
            input_prompt=input_prompt,
            instrument=instrument,
            density=density,
            temperature=temperature,
        )

        track = self.get_last_generated_track(full_piece)
        self.update_track_dict__add_bars(track, -1)
        full_piece = self.get_whole_piece_from_bar_dict()
        return full_piece

    """ Piece generation - Basics """

    def generate_piece(self, instrument_list, density_list, temperature_list):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list
        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        'generated_piece' keeps track of the entire piece
        'generated_piece' is returned by self.generate_until_track_end
        # it is returned by self.generate_until_track_end"""

        generated_piece = "PIECE_START "
        for instrument, density, temperature in zip(
            instrument_list, density_list, temperature_list
        ):
            generated_piece = self.generate_one_new_track(
                instrument,
                density,
                temperature,
                input_prompt=generated_piece,
            )

        # generated_piece = self.get_whole_piece_from_bar_dict()
        self.check_the_piece_for_errors()
        return generated_piece

    """ Piece generation - Extra Bars """

    @staticmethod
    def process_prompt_for_next_bar(self, track_idx):
        """Processing the prompt for the model to generate one more bar only.
        The prompt containts:
                if not the first bar: the previous, already processed, bars of the track
                the bar initialization (ex: "TRACK_START INST=DRUMS DENSITY=2 ")
                the last (self.model_n_bar)-1 bars of the track
        Args:
            track_idx (int): the index of the track to be processed

        Returns:
            the processed prompt for generating the next bar
        """
        track = self.piece_by_track[track_idx]
        # for bars which are not the bar to prolong
        pre_promt = "PIECE_START "
        for i, othertrack in enumerate(self.piece_by_track):
            if i != track_idx:
                len_diff = len(othertrack["bars"]) - len(track["bars"])
                if len_diff > 0:
                    # if other bars are longer, it mean that this one should catch up
                    pre_promt += othertrack["bars"][0]
                    for bar in track["bars"][-self.model_n_bar :]:
                        pre_promt += bar
                    pre_promt += "TRACK_END "
                elif False:  # len_diff <= 0: # THIS GENERATES EMPTINESS
                    # adding an empty bars at the end of the other tracks if they have not been processed yet
                    pre_promt += othertracks["bars"][0]
                    for bar in track["bars"][-(self.model_n_bar - 1) :]:
                        pre_promt += bar
                    for _ in range(abs(len_diff) + 1):
                        pre_promt += "BAR_START BAR_END "
                    pre_promt += "TRACK_END "

        # for the bar to prolong
        # initialization e.g TRACK_START INST=DRUMS DENSITY=2
        processed_prompt = track["bars"][0]
        for bar in track["bars"][-(self.model_n_bar - 1) :]:
            # adding the "last" bars of the track
            processed_prompt += bar

        processed_prompt += "BAR_START "
        print(
            f"--- prompt length = {len((pre_promt + processed_prompt).split(' '))} ---"
        )
        return pre_promt + processed_prompt

    def generate_one_more_bar(self, i):
        """Generate one more bar from the input_prompt"""
        processed_prompt = self.process_prompt_for_next_bar(self, i)
        prompt_plus_bar = self.generate_until_track_end(
            input_prompt=processed_prompt,
            temperature=self.piece_by_track[i]["temperature"],
            expected_length=1,
            verbose=False,
        )
        added_bar = self.get_newly_generated_bar(prompt_plus_bar)
        self.update_track_dict__add_bars(added_bar, i)

    def get_newly_generated_bar(self, prompt_plus_bar):
        return "BAR_START " + self.striping_track_ends(
            prompt_plus_bar.split("BAR_START ")[-1]
        )

    def generate_n_more_bars(self, n_bars, only_this_track=None, verbose=True):
        """Generate n more bars from the input_prompt"""
        if only_this_track is None:
            only_this_track

        print(f"================== ")
        print(f"Adding {n_bars} more bars to the piece ")
        for bar_id in range(n_bars):
            print(f"----- added bar #{bar_id+1} --")
            for i, track in enumerate(self.piece_by_track):
                if only_this_track is None or i == only_this_track:
                    print(f"--------- {track['label']}")
                    self.generate_one_more_bar(i)
        self.check_the_piece_for_errors()

    def check_the_piece_for_errors(self, piece: str = None):

        if piece is None:
            piece = generate_midi.get_whole_piece_from_bar_dict()
        errors = []
        errors.append(
            [
                (token, id)
                for id, token in enumerate(piece.split(" "))
                if token not in self.tokenizer.vocab or token == "UNK"
            ]
        )
        if len(errors) > 0:
            # print(piece)
            for er in errors:
                er
                print(f"Token not found in the piece at {er[0][1]}: {er[0][0]}")
                print(piece.split(" ")[er[0][1] - 5 : er[0][1] + 5])


if __name__ == "__main__":

    # worker
    DEVICE = "cpu"

    # define generation parameters
    N_FILES_TO_GENERATE = 2
    Temperatures_to_try = [0.7]

    USE_FAMILIZED_MODEL = True
    force_sequence_length = True

    if USE_FAMILIZED_MODEL:
        # model_repo = "misnaej/the-jam-machine-elec-famil"
        # model_repo = "misnaej/the-jam-machine-elec-famil-ft32"

        # model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
        # n_bar_generated = 8

        model_repo = "JammyMachina/improved_4bars-mdl"
        n_bar_generated = 4
        instrument_promt_list = ["4", "DRUMS", "3"]
        # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
        density_list = [3, 2, 2]
        # temperature_list = [0.7, 0.7, 0.75]
    else:
        model_repo = "misnaej/the-jam-machine"
        instrument_promt_list = ["30"]  # , "DRUMS", "0"]
        density_list = [3]  # , 2, 3]
        # temperature_list = [0.7, 0.5, 0.75]
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
            generate_midi = GenerateMidiText(model, tokenizer)
            # 0 - set the n_bar for this model
            generate_midi.set_nb_bars_generated(n_bars=n_bar_generated)
            # 1 - defines the instruments, densities and temperatures
            # 2- generate the first 8 bars for each instrument
            generate_midi.set_improvisation_level(30)
            generate_midi.generate_piece(
                instrument_promt_list,
                density_list,
                [temperature for _ in density_list],
            )
            # 3 - force the model to improvise
            # generate_midi.set_improvisation_level(20)
            # 4 - generate the next 4 bars for each instrument
            # generate_midi.generate_n_more_bars(n_bar_generated)
            # 5 - lower the improvisation level
            generate_midi.generated_piece = (
                generate_midi.get_whole_piece_from_bar_dict()
            )

            # print the generated sequence in terminal
            print("=========================================")
            print(generate_midi.generated_piece)
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
            inst_midi, mixed_audio = get_music(filename.split(".")[0] + ".mid")
            max_time = get_max_time(inst_midi)
            plot_piano_roll(inst_midi)

            print("Et voilà! Your MIDI file is ready! GO JAM!")

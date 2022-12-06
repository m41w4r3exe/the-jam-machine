from utils import WriteTextMidiToFile
from utils import define_generation_dir
from load import LoadModel


class GenerateMidiText:
    """
    # instantiate the class
    temperature = 0.75 # anything between 0 and +inf
    gen = GenerateMidiText(model, tokenizer, device, temperature=temperature)
    # generate a sequence:
    generated_sequence = gen.generate_one_sequence(input_prompt="PIECE_START")
    # generate a DRUM sequence:
    generated_sequence = gen.generate_one_sequence(inst="INST=DRUMS")
    # generate a multi track sequence
    generated_multi_track_sequence = gen.generate_multi_track_sequence(
        inst_list=["DRUMS", "38", "82", "51"],
        density_list=[2, 3, 3, 1])
    """

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        max_seq_length=None,
        temperature=0.75,
        generate_until="TRACK_END",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if max_seq_length is not None:
            self.max_length = max_seq_length
            print(f"Sequence length set to {self.max_length}")
        else:
            self.max_length = model.config.n_positions
            print(
                f"Sequence length set to {self.max_length} BASED ON 'model.config.n_positions'"
            )

        self.temperature = temperature
        self.generate_until = generate_until

    def tokenize_input_prompt(self, input_prompt, verbose=True):
        input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors="pt")
        if self.device == "cuda":  # TO CHECK - not sure if it works
            input_prompt_ids.cuda()

        if verbose:
            # print(f"input_prompts: {self.tokenizer.decode(input_prompt_ids[0])}")
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
            eos_token_id=self.tokenizer.encode(self.generate_until)[0],
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
    ):
        """generate a sequence based on
        - the input_prompt and/or inst and density parameters
        - the "final prompt" is converted into input_prompt_ids
        - input_prompt_ids are passed to generate_sequence_of_token_ids for generation
        - the generated toekn_ids are then converted to text
        """
        if inst is not None:
            if type(inst) is not str:  # inst should be str
                inst = str(inst)

            input_prompt = f"{input_prompt} TRACK_START INST={inst} "
            if density is not None:
                input_prompt = f"{input_prompt} DENSITY={density}"

        if inst is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if verbose:
            print("--------------------")
            print(
                f"Generating {inst} - Density {density} - Temperature {self.temperature}"
            )

        input_prompt_ids = self.tokenize_input_prompt(input_prompt)
        generated_ids = self.generate_sequence_of_token_ids(input_prompt_ids)
        generated_text = self.convert_ids_to_text(generated_ids)
        return generated_text

    def generate_multi_track_sequence(
        self, inst_list=["DRUMS", "84", 5], density_list=[3, 2, 1]
    ):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list

        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        """
        generate_features_dict = {
            "model_identification": self.model.transformer.base_model.name_or_path,
            "inst_list": inst_list,
            "density_list": density_list,
            "temperature": self.temperature,
            "max_seq_length": self.max_length,
            "generate_until": self.generate_until,
        }

        generated_multi_track_dict = {}
        generated_multi_track_sequence = "PIECE_START"
        for count, (inst, density) in enumerate(zip(inst_list, density_list)):
            generated_multi_track_sequence_length = len(generated_multi_track_sequence)
            generated_multi_track_sequence = self.generate_one_sequence(
                input_prompt=f"{generated_multi_track_sequence}",
                inst=inst,
                density=density,
            )
            if count > 0:  # not first iteration
                generated_track = generated_multi_track_sequence[
                    generated_multi_track_sequence_length + 1 :
                ]
            else:
                generated_track = generated_multi_track_sequence

            generated_multi_track_dict[f"INST={inst}"] = generated_track

        return (
            generated_multi_track_sequence,
            generated_multi_track_dict,
            generate_features_dict,
        )

    def process_prompt_for_next_bar(self, input_prompt, n_bars=2):
        """
        input_prompt should be at least a 8 bar sequence for one instrument
        """
        input_prompt_split = input_prompt.split(" ")
        prompt_processed_for_next_bar = ""
        bar_skipped = 0
        skipping_first_bar = False
        first_bar_skipped = False
        for token in input_prompt_split[:-1]:
            # input_prompt_split[:-1] should exclude TRACK_END
            if first_bar_skipped == False:
                if token == "BAR_START":
                    skipping_first_bar = True

                if skipping_first_bar is True:
                    # print(f"skipping token {token} for the first bar ")
                    if token != "BAR_END":
                        continue

                    else:
                        bar_skipped += 1
                        if bar_skipped == n_bars:
                            skipping_first_bar = False
                            first_bar_skipped = True

                        continue

                # if token == "TRACK_END":
                #     # print("removing the TRACK_END token")
                #     continue

            prompt_processed_for_next_bar += f"{token} "

        return prompt_processed_for_next_bar

    def generate_one_more_bar(self, input_prompt):
        prompt_processed_for_next_bar = self.process_prompt_for_next_bar(input_prompt)
        prompt_plus_bar = self.generate_one_sequence(
            input_prompt=prompt_processed_for_next_bar
        )
        added_bar = prompt_plus_bar[
            len(prompt_processed_for_next_bar) - 1 : -len("TRACK_END")
        ]
        return prompt_plus_bar, added_bar

    def generate_n_more_bars(self, input_prompt, n_bars=2):
        new_bars = ""
        for bar in range(n_bars):
            input_prompt, new_bar = self.generate_one_more_bar(input_prompt)
            new_bars += new_bar

        return new_bars

    def bar_count_check(sequence, n_bars=8):
        """check if the sequence contains the right number of bars"""
        pass


if __name__ == "__main__":

    device = "cpu"
    # model_repo = "misnaej/the-jam-machine-1024"
    # model_repo = "misnaej/the-jam-machine"
    # model, tokenizer = LoadModel(
    #     model_repo, from_huggingface=True
    # ).load_model_and_tokenizer()

    model_repo = "models/model_2048_fake_wholedataset"
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=False
    ).load_model_and_tokenizer()

    generated_sequence_files_path = define_generation_dir(model_repo)
    # set the temperature
    temperature = 0.3

    # instantiate the GenerateMidiText class
    gen = GenerateMidiText(
        model,
        tokenizer,
        device,
        temperature=temperature,
    )

    # generate a multi track sequence
    inst_list = ["DRUMS", "34"]
    density_list = [2, 3]
    (
        generated_multi_track_sequence,
        generated_multi_track_dict,
        generate_features_dict,
    ) = gen.generate_multi_track_sequence(
        inst_list=inst_list,
        density_list=density_list,
    )

    # write to file
    WriteTextMidiToFile(
        generated_multi_track_sequence,
        generated_sequence_files_path,
        feature_dict=generate_features_dict,
    ).text_midi_to_file()

    # generate 8 more bars for the drums
    input_prompt = generated_multi_track_dict["INST=DRUMS"]
    seq = gen.generate_n_more_bars(input_prompt, n_bars=2)
    whole_seq = f"{input_prompt}{seq}TRACK_END "
    # write to file
    WriteTextMidiToFile(
        whole_seq,
        generated_sequence_files_path,
        feature_dict=generate_features_dict,
    ).text_midi_to_file()

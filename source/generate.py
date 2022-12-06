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
        inst_list=["INST=DRUMS", "INST=38", "INST=82", "INST=51"],
        density_list=[2, 3, 3, 1])
    """

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        max_seq_length=2048,
        temperature=0.75,
        generate_until="TRACK_END",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_seq_length
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

    # generate from the tokenized input_prompt
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
            input_prompt = f"{input_prompt} TRACK_START {inst} "
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
        self, inst_list=["INST=DRUMS", "INST=38", "INST=82"], density_list=[3, 2, 1]
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

        generated_multi_track_sequence = []
        input_prompt = "PIECE_START"
        for inst, density in zip(inst_list, density_list):
            input_prompt = self.generate_one_sequence(
                input_prompt=f"{input_prompt}", inst=inst, density=density
            )
        generated_multi_track_sequence = input_prompt
        return generated_multi_track_sequence, generate_features_dict


if __name__ == "__main__":

    device = "cpu"
    # model_repo = "misnaej/the-jam-machine"
    model_repo = "misnaej/the-jam-machine"
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    generated_sequence_files_path = define_generation_dir(model_repo)
    # set the temperature
    temperature = 1

    # instantiate the GenerateMidiText class
    gen = GenerateMidiText(model, tokenizer, device, temperature=temperature)

    # generate a multi track sequence
    inst_list = ["INST=DRUMS", "INST=5", "INST=34", "INST=81"]
    density_list = [2, 3, 2, 2]
    (
        generated_multi_track_sequence,
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
    generated_multi_track_sequence

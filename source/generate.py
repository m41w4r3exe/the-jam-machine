from utils import WriteTextMidiToFile
from load import LoadFromHuggingFace
from load import LoadModelFromLocalFolder
import os


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
    def generate_the_next_8_bars(
        self,
        input_prompt_ids,
        verbose=True,
    ):
        generated_ids = self.model.generate(
            input_prompt_ids,
            max_length=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.encode(self.generate_until)[0],
        )
        if verbose:
            print("Generating a 8 bar token_id sequence...")
            # print(f"output: {generated_ids}")
        return generated_ids

    # convert the generated tokens to string
    def convert_ids_to_text(self, generated_ids, verbose=True):
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
        if inst is not None:  # ex: inst="INST=DRUMS"
            input_prompt = f"{input_prompt} TRACK_START {inst} "
            if density is not None:  # ex: inst="INST=DRUMS"
                input_prompt = f"{input_prompt} DENSITY={density}"

        if inst is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if verbose:
            print("--------------------")
            print(
                f"Generating {inst} - Density {density} - Temperature {self.temperature}"
            )

        input_prompt_ids = self.tokenize_input_prompt(input_prompt)
        generated_ids = self.generate_the_next_8_bars(input_prompt_ids)
        generated_text = self.convert_ids_to_text(generated_ids)
        return generated_text

    def generate_multi_track_sequence(
        self,
        inst_list=["INST=DRUMS", "INST=34", "INST=81", "INST=5"],
        density_list=[3, 2, 1],
        verbose=True,
    ):

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
    load_from_huggingface = False

    if load_from_huggingface:
        # load model and tokenizer from HuggingFace
        model_repo = "misnaej/the-jam-machine"
        model, tokenizer = LoadFromHuggingFace(model_repo).load_model_and_tokenizer()
    else:
        # load model and tokenizer from a local folder
        model_path = "models/model_2048_wholedataset"
        model, tokenizer = LoadModelFromLocalFolder(
            model_path
        ).load_model_and_tokenizer()

    # defined path to generate
    generated_sequence_files_path = "models/model_2048_wholedataset/generated_sequences"
    if not os.path.exists(generated_sequence_files_path):
        os.makedirs(generated_sequence_files_path)

    # set the temperature
    temperature = 1

    # instantiate the GenerateMidiText class
    gen = GenerateMidiText(model, tokenizer, device, temperature=temperature)

    # generate a multi track sequence
    inst_list = ["INST=DRUMS", "INST=34", "INST=73", "INST=30"]
    density_list = [3, 3, 3, 0]
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

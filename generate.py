import os
import json
from hashlib import sha256
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


class GenerateMidiText:
    """
    # instantiate the class
    temperature = 0.75 # anything between 0 and +inf
    gen = GenerateMidiText(model, tokenizer, device, temperature=temperature)
    # generate a sequence:
    generated_sequence = gen.generate_one_sequence(input="PIECE_START")
    # generate a DRUM sequence:
    generated_sequence = gen.generate_one_sequence(inst="INST=DRUMS")
    # generate a multi track sequence
    generated_multi_track_sequence = gen.generate_multi_track_sequence(
        inst_list=["INST=DRUMS", "INST=38", "INST=82", "INST=51"],
        density_list=[3, 6, 2, 1])
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

    def tokenize_input(self, input, verbose=False):
        input_ids = self.tokenizer.encode(input, return_tensors="pt")
        if self.device == "cuda":  # TO CHECK - not sure if it works
            input_ids.cuda()

        if verbose:
            print(f"inputs: {self.tokenizer.decode(input_ids[0])}")
        return input_ids

    # generate from the tokenized input
    def generate_the_next_8_bars(
        self,
        input_ids,
        verbose=False,
    ):
        generated_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.encode(self.generate_until)[0],
        )
        if verbose:
            print(f"output: {generated_ids}")
        return generated_ids

    # convert the generated tokens to string
    def convert_ids_to_text(self, generated_ids, verbose=False):
        generated_text = self.tokenizer.decode(generated_ids[0])
        if verbose:
            print(f"output: {generated_text}")
        return generated_text

    def generate_one_sequence(self, input="PIECE_START", inst=None, density=None):
        if inst is not None:  # ex: inst="INST=DRUMS"
            input = f"{input} TRACK_START {inst} "
            if density is not None:  # ex: inst="INST=DRUMS"
                input = f"{input} DENSITY={density}"
        if inst is None and density is not None:
            print("Density cannot be defined without an input instrument #TOFIX")

        input_ids = self.tokenize_input(input)
        generated_ids = self.generate_the_next_8_bars(input_ids)
        generated_text = self.convert_ids_to_text(generated_ids)
        return generated_text

    def generate_multi_track_sequence(
        self,
        inst_list=["INST=DRUMS", "INST=38", "INST=82"],
        density_list=[3, 6, 2],
    ):
        generate_features_dict = {
            "inst_list": inst_list,
            "density_list": density_list,
            "temperature": self.temperature,
            "max_seq_length": self.max_length,
            "generate_until": self.generate_until,
        }

        generated_multi_track_sequence = []
        input = "PIECE_START"
        for inst, density in zip(inst_list, density_list):
            input = self.generate_one_sequence(
                input=f"{input}", inst=inst, density=density
            )
        generated_multi_track_sequence = input
        return generated_multi_track_sequence, generate_features_dict


class WriteTextMidiToFile:  # utils saving to file
    def __init__(self, sequence, output_path, feature_dict=None):
        self.sequence = sequence
        self.output_path = output_path
        self.feature_dict = feature_dict

    def hashing_seq(self):
        self.filename = sha256(self.sequence.encode("utf-8")).hexdigest()
        self.output_path_filename = f"{self.output_path}/{self.filename}.txt"

    def writing_seq_to_file(self):
        file_object = open(f"{self.output_path_filename}", "w")
        assert type(self.sequence) is str, "sequence must be a string"
        file_object.writelines(self.sequence)
        file_object.close()
        print(f"Token sequence written: {self.output_path_filename}")

    def writing_feature_dict_to_file(self):
        with open(f"{self.output_path}{self.filename}_features.json", "w") as json_file:
            json.dump(self.feature_dict, json_file)

    def text_midi_to_file(self):
        self.hashing_seq()
        self.writing_seq_to_file()
        self.writing_feature_dict_to_file()


if __name__ == "__main__":

    device = "cpu"
    # model path
    model_path = "./models/model_2048_10kseq"
    tokenizer_path = "./models/model_2048_10kseq/tokenizer.json"
    generated_sequence_files_path = "models/model_2048_10kseq/generated_sequences/"

    # load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    temperature = 0.2

    # instantiate the GenerateMidiText class
    gen = GenerateMidiText(model, tokenizer, device, temperature=temperature)

    # generate a multi track sequence
    inst_list = ["INST=DRUMS", "INST=38", "INST=82", "INST=51"]
    density_list = [3, 6, 2, 1]
    (
        generated_multi_track_sequence,
        generate_features_dict,
    ) = gen.generate_multi_track_sequence(
        inst_list=inst_list,
        density_list=density_list,
    )

    generated_sequence_files_path = f"{generated_sequence_files_path}"
    if not os.path.exists(generated_sequence_files_path):
        os.makedirs(generated_sequence_files_path)

    # write to file
    WriteTextMidiToFile(
        generated_multi_track_sequence,
        generated_sequence_files_path,
        feature_dict=generate_features_dict,
    ).text_midi_to_file()
    generated_multi_track_sequence

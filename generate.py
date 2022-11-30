from hashlib import sha256
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


class GenerateMidiText:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # self.load_GPT2_model(model_path)
        # self.load_tokenizer(tokenizer_path)

    def tokenize_input(self, input, verbose=False):
        input_ids = self.tokenizer.encode(input, return_tensors="pt")
        if self.device == "cuda":  # TO CHECK - not sure if it works
            input_ids.cuda()

        if verbose:
            print(f"inputs: {self.tokenizer.decode(input_ids[0])}")
        return input_ids

    # generate from the tokenized input
    def generate_the_next_8_bars(self, input_ids, verbose=False):
        generated_ids = self.model.generate(
            input_ids,
            max_length=2048,
            do_sample=True,
            temperature=0.75,
            eos_token_id=self.tokenizer.encode("TRACK_END")[0],
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
        generated_multi_track_sequence = []
        input = "PIECE_START"
        for inst, density in zip(inst_list, density_list):
            input = self.generate_one_sequence(
                input=f"{input}", inst=inst, density=density
            )
        generated_multi_track_sequence = input
        return generated_multi_track_sequence


class WriteTextMidiToFile:  # utils saving to file
    def __init__(self, sequence, output_path):
        self.sequence = sequence
        self.output_path = output_path

    def hashing_seq(self):
        self.filename = sha256(self.sequence.encode("utf-8")).hexdigest()
        self.output_path_filename = f"{self.output_path}/{self.filename}.txt"

    def writing_seq_to_file(self):
        file_object = open(f"{self.output_path_filename}", "w")
        assert type(self.sequence) is str, "sequence must be a string"
        file_object.writelines(self.sequence)
        file_object.close()
        print(f"Token sequence written: {self.output_path_filename}")

    def text_midi_to_file(self):
        self.hashing_seq()
        self.writing_seq_to_file()


if __name__ == "__main__":

    device = "cpu"
    # model path
    model_path = "./models/model_2048_10kseq"
    tokenizer_path = "./models/model_2048_10kseq/tokenizer.json"
    generated_sequence_files_path = "models/model_2048_10kseq/generated_sequences/"

    # load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # instantiate the GenerateMidiText class
    gen = GenerateMidiText(model, tokenizer, device)
    # generate a sequence
    # generated_sequence = gen.generate_one_sequence(input="PIECE_START")
    # WriteTextMidiToFile(
    #     generated_sequence, generated_sequence_files_path
    # ).text_midi_to_file()

    # generate a DRUM sequence
    # generated_sequence = gen.generate_one_sequence(inst="INST=DRUMS")
    # WriteTextMidiToFile(
    #     generated_sequence, generated_sequence_files_path
    # ).text_midi_to_file()

    # generate a multi track sequence
    generated_multi_track_sequence = gen.generate_multi_track_sequence()

    # write to file
    WriteTextMidiToFile(
        generated_multi_track_sequence, generated_sequence_files_path
    ).text_midi_to_file()
    generated_multi_track_sequence

from hashlib import sha256
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


class GenerateMidiText:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # Load the pretrained GPT2 model.
    def load_GPT2_model(model_path, device="cpu"):
        print("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        return model

    # Load the tokenizer.
    def load_tokenizer(tokenizer_path, device="cpu"):
        print("Loading tokenizer...")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        return tokenizer

    # tokenize the prompt
    def tokenize_prompt(prompt, verbose=True, device="cpu"):
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if device == "cuda":  # TO CHECK - not sure if it works
            prompt_ids.cuda()

        if verbose:
            print(f"inputs: {self.tokenizer.decode(prompt_ids[0])}")
        return prompt_ids

    # generate from the tokenized prompt
    def generate_the_next_8_bars(prompt_ids, verbose=False):
        generated_ids = self.model.generate(
            prompt_ids,
            max_length=2048,
            do_sample=True,
            temperature=0.75,
            eos_token_id=self.tokenizer.encode("TRACK_END")[0],
        )
        if verbose:
            print(f"output: {generated_ids}")
        return generated_ids

    # convert the generated tokens to string
    def convert_ids_to_text(generated_ids, verbose=True):
        generated_text = self.tokenizer.decode(generated_ids[0])
        if verbose:
            print(f"output: {generated_text}")
        return generated_text

    def generate_one_sequence(prompt="PIECE_START", inst=None, density=None):
        if inst is not None:  # ex: inst="INST=DRUMS"
            prompt = f"{prompt} TRACK_START {inst} "
        if density is not None:  # ex: inst="INST=DRUMS"
            prompt = f"{prompt} TRACK_START {inst} DENSITY={density}"

        input_ids = tokenize_prompt(prompt)
        generated_ids = generate_the_next_8_bars(input_ids)
        generated_text = convert_ids_to_text(generated_ids)
        return generated_text

    def generate_multi_track_sequence(
        model,
        tokenizer,
        inst_list=["INST=DRUMS", "INST=38", "INST=82"],
        density_list=[3, 6, 2],
    ):
        generated_multi_track_sequence = []
        prompt = "PIECE_START"
        for inst, density in zip(inst_list, density_list):
            prompt = generate_one_sequence(
                model, tokenizer, prompt=f"{prompt}", inst=inst, density=density
            )
            generated_multi_track_sequence.append(prompt)
        return generated_multi_track_sequence

    # utils saving to file
    def hashing_seq(generated_sequence):
        seq_cats = ""
        for thisseq in generated_sequence:
            seq_cats += thisseq[0]
        filename = sha256(seq_cats.encode("utf-8")).hexdigest()
        return filename

    def writing_seq_to_file(generated_sequence, filepath_name):
        file_object = open(f"{filepath_name}.txt", "w")
        [file_object.writelines(f"{lines[0]} \n") for lines in generated_sequence]
        file_object.close()

        return print(f"Token sequence written: {filepath_name}.txt")


if __name__ == "__main__":

    device = "cpu"
    model_path = (
        "/content/drive/MyDrive/the_jam_machine/model_jean_2048/checkpoint-50000/"
    )
    tokenizer_path = (
        "/content/drive/MyDrive/the_jam_machine/model_jean_2048/tokenizer.json"
    )
    model = load_GPT2_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path, device)

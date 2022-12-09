from utils import WriteTextMidiToFile
from generation_utils import define_generation_dir, bar_count_check
from load import LoadModel
from tqdm import tqdm
from constants import INSTRUMENT_CLASSES
import numpy as np


class GenerateMidiText:
    """Generating music with Class"""

    def __init__(self, model, tokenizer, device="cpu", temperature=0.75):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = model.config.n_positions
        print(
            f"Attention length set to {self.max_length} -> 'model.config.n_positions'"
        )
        self.temperature = temperature
        self.generate_until = "TRACK_END"

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

    def generate_one_sequence(
        self,
        input_prompt="PIECE_START",
        inst=None,
        density=None,
        verbose=True,
        expected_length=8,
    ):
        """generate a sequence
        - input_prompt combined with inst and density parameters -> input_prompt
        - input_prompt is converted into input_prompt_ids
        - input_prompt_ids are passed to generate_sequence_of_token_ids for generation
        - the generated token_ids are then converted to text"""

        if inst is not None:
            if type(inst) is not str:
                inst = str(inst)
            input_prompt = f"{input_prompt} TRACK_START INST={inst} "
            if density is not None:
                input_prompt = f"{input_prompt}DENSITY={density} "

        if inst is None and density is not None:
            print("Density cannot be defined without an input_prompt instrument #TOFIX")

        if verbose:
            print("--------------------")
            print(
                f"Generating {inst} - Density {density} - temperature {self.temperature}"
            )
        bar_count_checks = False
        while not bar_count_checks:
            input_prompt_ids = self.tokenize_input_prompt(input_prompt)
            generated_tokens = self.generate_sequence_of_token_ids(input_prompt_ids)
            generated_text = self.convert_ids_to_text(generated_tokens)
            newly_generated_only = generated_text[len(input_prompt) :]
            bar_count_checks, bar_count = bar_count_check(
                newly_generated_only, expected_length
            )
            if bar_count_checks is False:
                """Cut the sequence if too long"""
                if bar_count - expected_length > 0:
                    regenerated_text = ""
                    splited = generated_text.split("BAR_END ")
                    for count, spl in enumerate(splited):
                        if count < expected_length:
                            regenerated_text += spl + "BAR_END "

                    regenerated_text += "TRACK_END"
                    generated_text = regenerated_text
                    print("Generated sequence trunkated at 8 bars")
                    bar_count_checks = True
                else:
                    pass

        return generated_text

    def generate_multi_track_sequence(
        self, inst_list=["4", "0", "DRUMS"], density_list=[3, 2, 3]
    ):
        """generate a sequence with mutiple tracks
        - inst_list sets the list of instruments of the order of generation
        - density is paired with inst_list

        Each track/intrument is generated on a prompt which contains the previously generated track/instrument
        This means that the first instrument is generated with less bias than the next one, and so on.

        """
        generate_features_dict = self.make_feature_dict(self, inst_list, density_list)
        generated_multi_track_dict = {}
        generated_multi_track_sequence = "PIECE_START"
        for count, (inst, density) in enumerate(zip(inst_list, density_list)):
            seq_len = len(generated_multi_track_sequence)
            generated_multi_track_sequence = self.generate_one_sequence(
                input_prompt=f"{generated_multi_track_sequence}",
                inst=inst,
                density=density,
            )
            if count > 0:  # not first iteration
                generated_track = generated_multi_track_sequence[seq_len + 1 :]
            else:
                generated_track = generated_multi_track_sequence

            generated_multi_track_dict[f"INST={inst}"] = generated_track

        return (
            generated_multi_track_sequence,
            generated_multi_track_dict,
            generate_features_dict,
        )

    ## MAKE STATIC
    def process_prompt_for_next_bar(self, input_prompt):
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

    def generate_one_more_bar(self, input_prompt):
        """Generate one more bar from the input_prompt"""
        processed_prompt = self.process_prompt_for_next_bar(input_prompt)
        prompt_plus_bar = self.generate_one_sequence(
            input_prompt=processed_prompt,
            expected_length=1,
        )
        # remove the processed_prompt - but keeping "BAR_START " - and the TRACK_END
        added_bar = prompt_plus_bar[
            len(processed_prompt) - len("BAR_START ") : -len("TRACK_END ")
        ]
        return prompt_plus_bar, added_bar

    def generate_n_more_bars(self, input_prompt, n_bars=8):
        """Generate n more bars from the input_prompt"""
        new_bars = ""
        for _ in range(n_bars):
            bar_count_matches = False
            while bar_count_matches is False:
                input_prompt, new_bar = self.generate_one_more_bar(input_prompt)
                bar_count_matches, _ = bar_count_check(new_bar, 1)
            new_bars += new_bar

        return new_bars

    @staticmethod
    def make_feature_dict(self, inst_list, density_list):
        return {
            "model_identification": self.model.transformer.base_model.name_or_path,
            "inst_list": inst_list,
            "density_list": density_list,
            "temperature": self.temperature,
            "max_seq_length": self.max_length,
            "generate_until": self.generate_until,
        }


def print_inst_classes(INSTRUMENT_CLASSES):
    """Print the instrument classes"""
    for classe in INSTRUMENT_CLASSES:
        print(f"{classe}")


def check_if_prompt_inst_in_tokenizer_vocab(tokenizer, inst_prompt_list):
    """Check if the prompt instrument are in the tokenizer vocab"""
    for inst in inst_prompt_list:
        if f"INST={inst}" not in tokenizer.vocab:
            instruments_in_dataset = np.sort(
                [tok.split("=")[-1] for tok in tokenizer.vocab if "INST" in tok]
            )
            print_inst_classes(INSTRUMENT_CLASSES)
            raise ValueError(
                f"""The instrument {inst} is not in the tokenizer vocabulary. 
                Available Instruments: {instruments_in_dataset}"""
            )


if __name__ == "__main__":

    DEVICE = "cpu"
    # model_repo = "misnaej/the-jam-machine-1024"
    # model_repo = "misnaej/the-jam-machine"
    # model, tokenizer = LoadModel(
    #     model_repo, from_huggingface=True
    # ).load_model_and_tokenizer()

    model_repo = "misnaej/the-jam-machine-elec-famil"
    generated_sequence_files_path = define_generation_dir(model_repo)

    temperature = 0.2
    instrument_promt_list = ["0", "4", "DRUMS"]
    density_list = [3, 2, 3]

    """" load model and tokenizer """
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    """" check if the prompt makes sense"""
    check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

    """" instantiate the class for generation """
    genesis = GenerateMidiText(
        model,
        tokenizer,
        DEVICE,
        temperature=temperature,
    )
    """" generate a multi-track sequence """
    (
        generated_multi_track_sequence,
        generated_multi_track_dict,
        generate_features_dict,
    ) = genesis.generate_multi_track_sequence(
        inst_list=instrument_promt_list,
        density_list=density_list,
    )
    """" write to JSON file """
    WriteTextMidiToFile(
        generated_multi_track_sequence,
        generated_sequence_files_path,
        feature_dict=generate_features_dict,
    ).text_midi_to_file()

    # """" Generate the next 8 bars """
    # input_prompt = generated_multi_track_dict["INST=DRUMS"]
    # added_sequence = genesis.generate_n_more_bars(input_prompt, n_bars=8)
    # added_sequence = f"{input_prompt}{added_sequence}TRACK_END "
    # """" Write to JSON file """
    # WriteTextMidiToFile(
    #     added_sequence,
    #     generated_sequence_files_path,
    #     feature_dict=generate_features_dict,
    # ).text_midi_to_file()

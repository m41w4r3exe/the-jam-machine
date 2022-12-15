import random
from joblib import Parallel, delayed
from pathlib import Path
from constants import INSTRUMENT_CLASSES
from utils import get_files, timeit, FileCompressor


class Familizer:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.reverse_family()

    def get_family_number(self, program_number):
        """
        Given a MIDI instrument number, return its associated instrument family number.
        """
        for instrument_class in INSTRUMENT_CLASSES:
            if program_number in instrument_class["program_range"]:
                return instrument_class["family_number"]

    def reverse_family(self):
        """
        Create a dictionary of family numbers to randomly assigned program numbers.
        This is used to reverse the family number tokens back to program number tokens.
        """
        self.reference_programs = {}
        for family in INSTRUMENT_CLASSES:
            self.reference_programs[family["family_number"]] = random.choice(
                family["program_range"]
            )

    def get_program_number(self, family_number):
        """
        Given given a family number return a random program number in the respective program_range.
        This is the reverse operation of get_family_number.
        """
        assert family_number in self.reference_programs
        return self.reference_programs[family_number]

    # Replace instruments in text files
    def replace_instrument_token(self, token):
        """
        Given a MIDI program number in a word token, replace it with the family or program
        number token depending on the operation.
        e.g. INST=86 -> INST=10
        """
        inst_number = int(token.split("=")[1])
        if self.operation == "family":
            return "INST=" + str(self.get_family_number(inst_number))
        elif self.operation == "program":
            return "INST=" + str(self.get_program_number(inst_number))

    def replace_instrument_in_text(self, text):
        """Given a text piece, replace all instrument tokens with family number tokens."""
        return " ".join(
            [
                self.replace_instrument_token(token)
                if token.startswith("INST=") and not token == "INST=DRUMS"
                else token
                for token in text.split(" ")
            ]
        )

    def replace_instruments_in_file(self, file):
        """Given a text file, replace all instrument tokens with family number tokens."""
        text = file.read_text()
        file.write_text(self.replace_instrument_in_text(text))

    @timeit
    def replace_instruments(self):
        """
        Given a directory of text files:
        Replace all instrument tokens with family number tokens.
        """
        files = get_files(self.output_directory, extension="txt")
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.replace_instruments_in_file)(file) for file in files
        )

    def replace_tokens(self, input_directory, output_directory, operation):
        """
        Given a directory and an operation, perform the operation on all text files in the directory.
        operation can be either 'family' or 'program'.
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.operation = operation

        # Uncompress files, replace tokens, compress files
        fc = FileCompressor(self.input_directory, self.output_directory, self.n_jobs)
        fc.unzip()
        self.replace_instruments()
        fc.zip()
        print(self.operation + " complete.")

    def to_family(self, input_directory, output_directory):
        """
        Given a directory containing zip files, replace all instrument tokens with
        family number tokens. The output is a directory of zip files.
        """
        self.replace_tokens(input_directory, output_directory, "family")

    def to_program(self, input_directory, output_directory):
        """
        Given a directory containing zip files, replace all instrument tokens with
        program number tokens. The output is a directory of zip files.
        """
        self.replace_tokens(input_directory, output_directory, "program")


if __name__ == "__main__":

    # Choose number of jobs for parallel processing
    n_jobs = -1

    # Instantiate Familizer
    familizer = Familizer(n_jobs)

    # Choose directory to process for program
    input_directory = Path("midi/dataset/first_selection/validate").resolve()  # fmt: skip
    output_directory = input_directory / "family"

    # familize files
    familizer.to_family(input_directory, output_directory)

    # Choose directory to process for family
    # input_directory = Path("../data/music_picks/encoded_samples/validate/family").resolve()  # fmt: skip
    # output_directory = input_directory.parent / "program"

    # # programize files
    # familizer.to_program(input_directory, output_directory)

import random
from joblib import Parallel, delayed
from pathlib import Path
from constants import INSTRUMENT_CLASSES
from utils import get_files, timeit, uncompress_files, compress_files

# DONE
# separate utils and constants
# fix path issues, offer output path as argument

# TO DO
# create Familizer class
# optimize directory paths
# create a Zip class
# fix random instrument issue with an instantiation of instrument number
# create a parallel util function


class Familizer:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def get_family_number(self, program_number):
        """
        Given a MIDI instrument number, return its associated instrument family number.
        """
        for instrument_class in INSTRUMENT_CLASSES:
            if program_number in instrument_class["program_range"]:
                return instrument_class["family_number"]

    def get_program_number(self, family_number):
        """
        Given given a family number return a random program number in the respective program_range.
        This is the reverse operation of get_family_number.
        """
        for instrument_class in INSTRUMENT_CLASSES:
            if family_number == instrument_class["family_number"]:
                return random.choice(instrument_class["program_range"])

    # Replace instruments in text files
    def replace_instrument_token(self, token, operation):
        """
        Given a MIDI program number in a word token, replace it with the family or program
        number token depending on the operation.
        e.g. INST=86 -> INST=10
        """
        program_number = int(token.split("=")[1])
        if operation == "family":
            return "INST=" + str(self.get_family_number(program_number))
        elif operation == "program":
            return "INST=" + str(self.get_program_number(program_number))

    def replace_instrument_in_text(self, text, operation):
        """Given a text file, replace all instrument tokens with family number tokens."""
        return " ".join(
            [
                self.replace_instrument_token(token, operation)
                if token.startswith("INST=") and not token == "INST=DRUMS"
                else token
                for token in text.split(" ")
            ]
        )

    def replace_instruments_in_file(self, file, operation):
        """Given a text file, replace all instrument tokens with family number tokens."""
        text = file.read_text()
        file.write_text(self.replace_instrument_in_text(text, operation))

    @timeit
    def replace_instruments(self, directory, operation, n_jobs):
        """
        Given a directory of text files:
        Replace all instrument tokens with family number tokens.
        """
        files = get_files(directory, extension="txt")
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.replace_instruments_in_file)(file, operation) for file in files
        )

    def replace_tokens(self, input_directory, output_directory, operation, n_jobs):
        """
        Given a directory and an operation, perform the operation on all text files in the directory.
        operation can be either 'family' or 'program'.
        """
        uncompress_files(input_directory, output_directory, n_jobs)
        self.replace_instruments(output_directory, operation, n_jobs)
        compress_files(output_directory, output_directory, n_jobs)
        print(operation + " complete.")

    def to_family(self, input_directory, output_directory):
        self.replace_tokens(input_directory, output_directory, "family", self.n_jobs)

    def to_program(self, input_directory, output_directory):
        self.replace_tokens(input_directory, output_directory, "program", self.n_jobs)


if __name__ == "__main__":

    # Choose number of jobs for parallel processing
    n_jobs = -1

    # Instantiate Familizer
    familizer = Familizer(n_jobs)

    # Choose directory to process for program
    input_directory = Path("../data/music_picks/encoded_samples/validate").resolve()  # fmt: skip
    output_directory = input_directory / 'family'

    # familize files
    familizer.to_family(input_directory, output_directory)

    # Choose directory to process for family
    input_directory = Path("../data/music_picks/encoded_samples/validate/family").resolve()  # fmt: skip
    output_directory = input_directory.parent / 'program'

    # programize files
    familizer.to_program(input_directory, output_directory)

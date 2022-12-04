import random
from joblib import Parallel, delayed
from pathlib import Path
from constants import INSTRUMENT_CLASSES
from utils import get_files, timeit, uncompress_files, compress_files

# DONE
# separate utils and constants

# TO DO
# fix path issues, offer output path as argument
# fix random instrument issue with an instantiation of instrument number
# create Familizer class
# create a Zip class
# create a parallel util function


def get_family_number(program_number):
    """
    Given a MIDI instrument number, return its associated instrument family number.
    """
    for instrument_class in INSTRUMENT_CLASSES:
        if program_number in instrument_class["program_range"]:
            return instrument_class["family_number"]


def get_program_number(family_number):
    """
    Given given a family number return a random program number in the respective program_range.
    This is the reverse operation of get_family_number.
    """
    for instrument_class in INSTRUMENT_CLASSES:
        if family_number == instrument_class["family_number"]:
            return random.choice(instrument_class["program_range"])


# Replace instruments in text files
def replace_instrument_token(token, operation):
    """
    Given a MIDI program number in a word token, replace it with the family or program
    number token depending on the operation.
    e.g. INST=86 -> INST=10
    """
    program_number = int(token.split("=")[1])
    if operation == "family":
        return "INST=" + str(get_family_number(program_number))
    elif operation == "program":
        return "INST=" + str(get_program_number(program_number))


def replace_instrument_in_text(text, operation):
    """Given a text file, replace all instrument tokens with family number tokens."""
    return " ".join(
        [
            replace_instrument_token(token, operation)
            if token.startswith("INST=") and not token == "INST=DRUMS"
            else token
            for token in text.split(" ")
        ]
    )


def replace_instruments_in_file(file, operation):
    """Given a text file, replace all instrument tokens with family number tokens."""
    text = file.read_text()
    file.write_text(replace_instrument_in_text(text, operation))


@timeit
def replace_instruments(directory, operation, n_jobs):
    """
    Given a directory of text files:
    Replace all instrument tokens with family number tokens.
    """
    files = get_files(directory, extension="txt")
    Parallel(n_jobs=n_jobs)(
        delayed(replace_instruments_in_file)(file, operation) for file in files
    )


def replace_tokens(input_directory, output_directory, operation, n_jobs):
    """
    Given a directory and an operation, perform the operation on all text files in the directory.
    operation can be either 'family' or 'program'.
    """
    assert operation in ["family", "program"]
    uncompress_files(input_directory, output_directory, n_jobs)
    replace_instruments(output_directory, operation, n_jobs)
    compress_files(output_directory, output_directory, n_jobs)
    print(operation + " complete.")


if __name__ == "__main__":

    # Choose between program and family operation
    operation = "program"

    # Choose number of jobs for parallel processing
    n_jobs = -1

    # Choose directory to process
    input_directory = Path(
        "../data/music_picks/encoded_samples/validate/family"
    ).resolve()
    output_directory = input_directory.parent / operation

    # Run operation
    replace_tokens(input_directory, output_directory, operation, n_jobs=n_jobs)

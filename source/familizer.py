from zipfile import ZipFile, ZIP_DEFLATED
import random
from joblib import Parallel, delayed
from time import perf_counter
from pathlib import Path


# Helper functions
def get_files(directory, extension):
    """Given a directory, get a list of the file paths of all files matching the
    specified file extension."""
    return directory.glob(f"*.{extension}")


def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds to run.")
        return result

    return wrapper


# fmt: off
# Instrument mapping and mapping functions
INSTRUMENT_CLASSES = [
    {"name": "Piano", "program_range": range(0, 8), "family_number": 0},
    {"name": "Chromatic Percussion", "program_range": range(8, 16), "family_number": 1},
    {"name": "Organ", "program_range": range(16, 24), "family_number": 2},
    {"name": "Guitar", "program_range": range(24, 32), "family_number": 3},
    {"name": "Bass", "program_range": range(32, 40), "family_number": 4},
    {"name": "Strings", "program_range": range(40, 48), "family_number": 5},
    {"name": "Ensemble", "program_range": range(48, 56), "family_number": 6},
    {"name": "Brass", "program_range": range(56, 64), "family_number": 7},
    {"name": "Reed", "program_range": range(64, 72), "family_number": 8},
    {"name": "Pipe", "program_range": range(72, 80), "family_number": 9},
    {"name": "Synth Lead", "program_range": range(80, 88), "family_number": 10},
    {"name": "Synth Pad", "program_range": range(88, 96), "family_number": 11},
    {"name": "Synth Effects", "program_range": range(96, 104), "family_number": 12},
    {"name": "Ethnic", "program_range": range(104, 112), "family_number": 13},
    {"name": "Percussive", "program_range": range(112, 120), "family_number": 14},
    {"name": "Sound Effects", "program_range": range(120, 128), "family_number": 15,},
]
# fmt: on


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


# File compression and decompression
def uncompress_single_file(file, operation):
    """uncompress single zip file"""
    with ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(file.parent / operation)


def compress_single_file(file, operation):
    """compress a single text file to a new zip file and delete the original"""
    output_file = file.parent / (file.stem + "_" + operation + ".zip")
    with ZipFile(output_file, "w") as zip_ref:
        zip_ref.write(file, arcname=file.name, compress_type=ZIP_DEFLATED)
        file.unlink()


@timeit
def uncompress_files(directory, operation, n_jobs):
    """uncompress all zip files in folder"""
    files = get_files(directory, extension="zip")
    Parallel(n_jobs=n_jobs)(
        delayed(uncompress_single_file)(file, operation) for file in files
    )
    return directory / operation


@timeit
def compress_files(directory, operation, n_jobs):
    """compress all text files in folder to new zip files and remove the text files"""
    files = get_files(directory, extension="txt")
    Parallel(n_jobs=n_jobs)(
        delayed(compress_single_file)(file, operation) for file in files
    )


# Main function
def replace_tokens(directory, operation, n_jobs):
    """
    Given a directory and an operation, perform the operation on all text files in the directory.
    operation can be either 'family' or 'program'.
    """
    assert operation in ["family", "program"]
    output_directory = uncompress_files(directory, operation, n_jobs)
    replace_instruments(output_directory, operation, n_jobs)
    compress_files(output_directory, operation, n_jobs)
    print(operation + " complete.")


if __name__ == "__main__":

    # Choose between program and family operation
    operation = "program"

    # Choose number of jobs for parallel processing
    n_jobs = -1

    # Choose directory to process
    directory = "../data/music_picks/encoded_samples/validate"
    directory = Path(directory).resolve()

    # Run operation
    replace_tokens(directory, operation, n_jobs=n_jobs)

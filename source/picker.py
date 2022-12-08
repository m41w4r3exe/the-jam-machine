"""
This script is used to pick out the midi files that we want to use for our project.
It takes a reference file as input, which contains the md5 hash of the tracks we want to copy.
This reference file is created by the script midi_stats.py and refined in mmd_metadata.py.
It then searches for all midi files in the input folder and subfolders and copies them to the output folder.
"""

from pathlib import Path
import pandas as pd
from utils import get_files, copy_file
from joblib import Parallel, delayed


def pick_midis(input_dir, output_dir, reference_file):

    # create output folder if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load file in which we have the md5 hash of the tracks we want to copy
    reference = pd.read_csv(reference_file)

    # get all midi files from the folder and subfolders that match the reference file
    file_paths = get_files(input_dir, "mid", recursive=True)
    file_paths = [f for f in file_paths if f.stem in list(reference.md5)]

    # copy all files from the file_paths list to the output folder
    for f in file_paths:
        copy_file(f, output_dir)

    print('All tracks copied faster than it takes to say "electronic music"')


if __name__ == "__main__":

    # Select paths
    input_dir = Path("data/lmd_full/").resolve()
    output_dir = Path("data/lmd_new/").resolve()
    reference_file = Path("data/electronic_artists.csv").resolve()

    # Run function
    pick_midis(input_dir, output_dir, reference_file)

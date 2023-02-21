from datetime import datetime
from miditok import Event, MIDILike
import os
import json
from time import perf_counter
from constants import DRUMS_BEAT_QUANTIZATION, NONE_DRUMS_BEAT_QUANTIZATION
from joblib import Parallel, delayed
from zipfile import ZipFile, ZIP_DEFLATED
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment
import shutil

""" Diverse utils"""


def index_has_substring(list, substring):
    for i, s in enumerate(list):
        if substring in s:
            return i
    return -1


# TODO: Make this singleton
def get_miditok():
    pitch_range = range(0, 127)  # was (21, 109)
    beat_res = {(0, 400): 8}
    return MIDILike(pitch_range, beat_res)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds to run.")
        return result

    return wrapper


def chain(input, funcs, *params):
    """Chain functions together, passing the output of one function as the input of the next."""
    res = input
    for func in funcs:
        try:
            res = func(res, *params)
        except TypeError:
            res = func(res)
    return res


def split_dots(value):
    """Splits a string separated by dots "a.b.c" into a list of integers [a, b, c]"""
    return list(map(int, value.split(".")))


def compute_list_average(l):
    return sum(l) / len(l)


def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


""" Encoding functions """


def int_dec_base_to_beat(beat_str):
    """
    Converts "integer.decimal.base" (str, from miditok) into beats
    e.g. "0.4.8" = 0 + 4/8 = 0.5
    Args:
        - beat_str: "integer.decimal.base"
    Returns:
        - beats: float
    """
    integer, decimal, base = split_dots(beat_str)
    return integer + decimal / base


def int_dec_base_to_delta(beat_str, instrument="drums"):
    """converts the time shift to time_delta according to Tristan's encoding scheme
    Drums TIME_DELTA are quantized according to DRUMS_BEAT_QUANTIZATION
    Other Instrument TIME_DELTA are quantized according to NONE_DRUMS_BEAT_QUANTIZATION"""

    beat_res = (
        DRUMS_BEAT_QUANTIZATION
        if instrument.lower() == "drums"
        else NONE_DRUMS_BEAT_QUANTIZATION
    )
    time_delta = int_dec_base_to_beat(beat_str) * beat_res
    return time_delta.__int__()


def get_text(event, instrument="drums"):
    """Converts an event into a string for the midi-text format"""
    match event.type:
        case "Piece-Start":
            return "PIECE_START "
        case "Track-Start":
            return "TRACK_START "
        case "Track-End":
            return "TRACK_END "
        case "Instrument":
            return f"INST={event.value} "
        case "Density":
            return f"DENSITY={event.value} "
        case "Bar-Start":
            return "BAR_START "
        case "Bar-End":
            return "BAR_END "
        case "Time-Shift":
            return f"TIME_DELTA={int_dec_base_to_delta(event.value, instrument)} "
        case "Note-On":
            return f"NOTE_ON={event.value} "
        case "Note-Off":
            return f"NOTE_OFF={event.value} "
        case _:
            return ""


""" Decoding functions """


def time_delta_to_beat(time_delta, instrument="drums"):
    """
    Converts TIME_DELTA (from midi-text) to beats according to Tristan's encoding scheme
    Args:
        - time_delta: int (TIME_DELTA)
        - instrument: str ("Drums" or other instrument): used to determine the quantization resolution defined on constants.py
    Returns:
        - beats: float
    """
    beat_res = (
        DRUMS_BEAT_QUANTIZATION
        if instrument.lower() == "drums"
        else NONE_DRUMS_BEAT_QUANTIZATION
    )
    beats = float(time_delta) / beat_res
    return beats


def beat_to_int_dec_base(beat, beat_res=8):
    """
    Converts beats into "integer.decimal.base" (str) for miditok
    Args:
        - beat_str: "integer.decimal.base"
    Returns:
        - beats: float (e.g. "0.4.8" = 0 + 4/8 = 0.5)
    """
    int_dec_base = [
        int((beat * beat_res) // beat_res),
        int((beat * beat_res) % beat_res),
        beat_res,
    ]
    return ".".join(map(str, int_dec_base))


def time_delta_to_int_dec_base(time_delta, instrument="drums"):
    return chain(
        time_delta,
        [
            time_delta_to_beat,
            beat_to_int_dec_base,
        ],
        instrument,
    )


def get_event(text, value=None, instrument="drums"):
    """Converts a midi-text like event into a miditok like event"""
    match text:
        case "PIECE_START":
            return Event("Piece-Start", value)
        case "TRACK_START":
            return Event("Track-Start", value)
        case "TRACK_END":
            return Event("Track-End", value)
        case "INST":
            return Event("Instrument", value)
        case "BAR_START":
            return Event("Bar-Start", value)
        case "BAR_END":
            return Event("Bar-End", value)
        case "TIME_SHIFT":
            return Event("Time-Shift", value)
        case "TIME_DELTA":
            return Event("Time-Shift", time_delta_to_int_dec_base(value, instrument))
            # return Event("Time-Shift", to_beat_str(int(value) / 4))
        case "NOTE_ON":
            return Event("Note-On", value)
        case "NOTE_OFF":
            return Event("Note-Off", value)
        case _:
            return None


""" File utils"""


def writeToFile(path, content):
    if type(content) is dict:
        with open(f"{path}", "w") as json_file:
            json.dump(content, json_file)
    else:
        if type(content) is not str:
            content = str(content)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def readFromFile(path, isJSON=False):
    with open(path, "r") as f:
        if isJSON:
            return json.load(f)
        else:
            return f.read()


def get_files(directory, extension, recursive=False):
    """
    Given a directory, get a list of the file paths of all files matching the
    specified file extension.
    directory: the directory to search as a Path object
    extension: the file extension to match as a string
    recursive: whether to search recursively in the directory or not
    """
    if recursive:
        return list(directory.rglob(f"*.{extension}"))
    else:
        return list(directory.glob(f"*.{extension}"))


def load_jsonl(filepath):
    """Load a jsonl file"""
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_mp3(waveform, output_path, bitrate="92k"):
    """
    Write a waveform to an mp3 file.
    output_path: Path object for the output mp3 file
    waveform: numpy array of the waveform
    bitrate: bitrate of the mp3 file (64k, 92k, 128k, 256k, 312k)
    """
    # write the wav file
    wav_path = output_path.with_suffix(".wav")
    write(wav_path, 44100, waveform.astype(np.float32))
    # compress the wav file as mp3
    AudioSegment.from_wav(wav_path).export(output_path, format="mp3", bitrate=bitrate)
    # remove the wav file
    wav_path.unlink()


def copy_file(input_file, output_dir):
    """Copy an input file to the output_dir"""
    output_file = output_dir / input_file.name
    shutil.copy(input_file, output_file)


class WriteTextMidiToFile:  # utils saving to file
    def __init__(self, generate_midi, output_path):
        self.generated_midi = generate_midi.generated_piece
        self.output_path = output_path
        self.hyperparameter_and_bars = generate_midi.piece_by_track

    def hashing_seq(self):
        self.current_time = get_datetime()
        self.output_path_filename = f"{self.output_path}/{self.current_time}.json"

    def wrapping_seq_hyperparameters_in_dict(self):
        # assert type(self.generated_midi) is str, "error: generate_midi must be a string"
        # assert (
        #     type(self.hyperparameter_dict) is dict
        # ), "error: feature_dict must be a dictionnary"
        return {
            "generated_midi": self.generated_midi,
            "hyperparameters_and_bars": self.hyperparameter_and_bars,
        }

    def text_midi_to_file(self):
        self.hashing_seq()
        output_dict = self.wrapping_seq_hyperparameters_in_dict()
        print(f"Token generate_midi written: {self.output_path_filename}")
        writeToFile(self.output_path_filename, output_dict)
        return self.output_path_filename


class FileCompressor:
    def __init__(self, input_directory, output_directory, n_jobs=-1):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.n_jobs = n_jobs

    # File compression and decompression
    def unzip_file(self, file):
        """uncompress single zip file"""
        with ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(self.output_directory)

    def zip_file(self, file):
        """compress a single text file to a new zip file and delete the original"""
        output_file = self.output_directory / (file.stem + ".zip")
        with ZipFile(output_file, "w") as zip_ref:
            zip_ref.write(file, arcname=file.name, compress_type=ZIP_DEFLATED)
            file.unlink()

    @timeit
    def unzip(self):
        """uncompress all zip files in folder"""
        files = get_files(self.input_directory, extension="zip")
        Parallel(n_jobs=self.n_jobs)(delayed(self.unzip_file)(file) for file in files)

    @timeit
    def zip(self):
        """compress all text files in folder to new zip files and remove the text files"""
        files = get_files(self.output_directory, extension="txt")
        Parallel(n_jobs=self.n_jobs)(delayed(self.zip_file)(file) for file in files)

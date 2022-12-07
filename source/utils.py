from datetime import datetime
from miditok import Event, MIDILike
import os
import json
from time import perf_counter
from joblib import Parallel, delayed
from zipfile import ZipFile, ZIP_DEFLATED


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


# Function to read from text from txt file:
def readFromFile(path, isJSON=False):
    with open(path, "r") as f:
        if isJSON:
            return json.load(f)
        else:
            return f.read()


def chain(input, funcs, *params):
    res = input
    for func in funcs:
        try:
            res = func(res, *params)
        except TypeError:
            res = func(res)
    return res


def to_beat_str(value, beat_res=8):
    values = [
        int(int(value * beat_res) / beat_res),
        int(int(value * beat_res) % beat_res),
        beat_res,
    ]
    return ".".join(map(str, values))


def to_base10(beat_str):
    integer, decimal, base = split_dots(beat_str)
    return integer + decimal / base


def split_dots(value):
    return list(map(int, value.split(".")))


def compute_list_average(l):
    return sum(l) / len(l)


def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def define_generation_dir(model_repo_path):
    #### to remove later ####
    if model_repo_path == "models/model_2048_fake_wholedataset":
        model_repo_path = "misnaej/the-jam-machine"
    #### to remove later ####
    generated_sequence_files_path = f"midi/generated/{model_repo_path}"
    if not os.path.exists(generated_sequence_files_path):
        os.makedirs(generated_sequence_files_path)
    return generated_sequence_files_path


def get_text(event):
    match event.type:
        case "Piece-Start":
            return "PIECE_START "
        case "Track-Start":
            return "TRACK_START "
        case "Track-End":
            return "TRACK_END "
        case "Instrument":
            return f"INST={event.value} "
        case "Bar-Start":
            return "BAR_START "
        case "Bar-End":
            return "BAR_END "
        case "Time-Shift":
            return f"TIME_SHIFT={event.value} "
        case "Note-On":
            return f"NOTE_ON={event.value} "
        case "Note-Off":
            return f"NOTE_OFF={event.value} "
        case _:
            return ""


def get_event(text, value=None):
    match text:
        case "PIECE_START":
            return Event("Piece-Start", value)
        case "TRACK_START":
            return None
        case "TRACK_END":
            return None
        case "INST":
            return Event("Instrument", value)
        case "BAR_START":
            return Event("Bar-Start", value)
        case "BAR_END":
            return Event("Bar-End", value)
        case "TIME_SHIFT":
            return Event("Time-Shift", value)
        case "TIME_DELTA":
            return Event("Time-Shift", to_beat_str(int(value) / 4))
        case "NOTE_ON":
            return Event("Note-On", value)
        case "NOTE_OFF":
            return Event("Note-Off", value)
        case _:
            return None


# TODO: Make this singleton
def get_tokenizer():
    pitch_range = range(21, 109)
    beat_res = {(0, 400): 8}
    return MIDILike(pitch_range, beat_res)


class WriteTextMidiToFile:  # utils saving to file
    def __init__(self, sequence, output_path, feature_dict=None):
        self.sequence = sequence
        self.output_path = output_path
        self.feature_dict = feature_dict

    def hashing_seq(self):
        self.current_time = get_datetime()
        self.output_path_filename = f"{self.output_path}/{self.current_time}.json"

    def wrapping_seq_feature_in_dict(self):
        assert type(self.sequence) is str, "error: sequence must be a string"
        assert (
            type(self.feature_dict) is dict
        ), "error: feature_dict must be a dictionnary"
        return {"sequence": self.sequence, "features": self.feature_dict}

    def text_midi_to_file(self):
        self.hashing_seq()
        output_dict = self.wrapping_seq_feature_in_dict()
        print(f"Token sequence written: {self.output_path_filename}")
        writeToFile(self.output_path_filename, output_dict)
        return self.output_path_filename


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


def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds to run.")
        return result

    return wrapper


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

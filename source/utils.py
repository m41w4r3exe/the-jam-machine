from datetime import datetime
from miditok import Event
import os
import json
from hashlib import sha256


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


def get_datetime_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


class WriteTextMidiToFile:  # utils saving to file
    def __init__(self, sequence, output_path, feature_dict=None):
        self.sequence = sequence
        self.output_path = output_path
        self.feature_dict = feature_dict

    def hashing_seq(self):
        # self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_time = get_datetime_filename()
        # self.filename = sha256(self.sequence.encode("utf-8")).hexdigest()
        self.output_path_filename = f"{self.output_path}/{self.current_time}.json"

    # def writing_seq_to_file(self):
    #     file_object = open(f"{self.output_path_filename}", "w")
    #     assert type(self.sequence) is str, "sequence must be a string"
    #     file_object.writelines(self.sequence)
    #     file_object.close()
    #     print(f"Token sequence written: {self.output_path_filename}")

    def wrapping_seq_feature_in_dict(self):
        assert type(self.sequence) is str, "error: sequence must be a string"
        assert (
            type(self.feature_dict) is dict
        ), "error: feature_dict must be a dictionnary"
        return {"sequence": self.sequence, "features": self.feature_dict}

    # def writing_feature_dict_to_file(feature_dict, output_path_filename):
    #     with open(f"{output_path_filename}_features.json", "w") as json_file:
    #         json.dump(feature_dict, json_file)

    def text_midi_to_file(self):
        self.hashing_seq()
        output_dict = self.wrapping_seq_feature_in_dict()
        print(f"Token sequence written: {self.output_path_filename}")
        writeToFile(self.output_path_filename, output_dict)
        # self.writing_feature_dict_to_file(self.feature_dict, self.output_path_filename)
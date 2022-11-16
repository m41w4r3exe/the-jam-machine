from miditok import MuMIDI
from miditoolkit import MidiFile
from utils import writeToFile

# TODO: Create EventToText methods

# Config for tokenizer
pitch_range = range(21, 109)
beat_res = {(0, 4000): 8}

tokenizer = MuMIDI(pitch_range, beat_res)

midi_filename = "the_strokes-reptilia"
midi = MidiFile(f"./midi/{midi_filename}.mid")

tokens = tokenizer.midi_to_tokens(midi)
events = tokenizer.tokens_to_events(tokens)

writeToFile(f"./midi/{midi_filename}_events_mumidi.txt", events)

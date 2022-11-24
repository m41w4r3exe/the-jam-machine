from miditoolkit import MidiFile
from miditok import OctupleMono
from utils import writeToFile

# TODO: Create EventToText methods

# Config for tokenizer
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 100
additional_tokens = {
    "Chord": False,
    "Rest": False,
    "Tempo": False,
    "Program": True,
    "TimeSignature": False,
}

tokenizer = OctupleMono(pitch_range, beat_res, nb_velocities, additional_tokens)

midi_filename = "the_strokes-reptilia"
midi = MidiFile(f"./midi/{midi_filename}.mid")

tokens = tokenizer.midi_to_tokens(midi)
# TODO: merge all instruments bar by bar
events = tokenizer.tokens_to_events(tokens[0])

writeToFile(f"./midi/{midi_filename}_events_oct.txt", events)

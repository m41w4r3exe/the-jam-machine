from miditoolkit import MidiFile
from miditok import MIDILike
from utils import writeToFile
from encoder import MIDIEncoder
import os

pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
tokenizer = MIDILike(pitch_range, beat_res)
encoder = MIDIEncoder(tokenizer)

midi_files = [f"midi/{f}" for f in os.listdir("midi/") if f.endswith(".mid")]

# midi_files = []

for file in midi_files:
    try:
        midi = MidiFile(file)
    except:
        print(f"Failed to load {file}")
        continue

    piece_text = encoder.get_piece_text(midi)

    midi_filename = os.path.splitext(os.path.basename(file))[0]
    dirname = os.path.dirname(file)
    writeToFile(f"{dirname}/encoded_txts/{midi_filename}.txt", piece_text)

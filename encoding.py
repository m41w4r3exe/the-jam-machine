from miditok import MIDILike, get_midi_programs, REMI
from miditoolkit import MidiFile

# Our parameters
pitch_range = range(0, 128)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {
    "Chord": True,
    "Rest": True,
    "Tempo": True,
    "Program": False,
    "TimeSignature": False,
    "rest_range": (2, 8),  # (half, 8 beats)
    "nb_tempos": 32,  # nb of tempo bins
    "tempo_range": (40, 250),
}  # (min, max)

# Creates the tokenizer and loads a MIDI
tokenizer = MIDILike(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
midi = MidiFile("lmd_full/0/0af5af6d9785c93d65215031077bead3.mid")

print(midi)

# # Converts MIDI to tokens, and back to a MIDI
tokens = tokenizer.midi_to_tokens(midi)
converted_back_midi = tokenizer.tokens_to_midi(tokens, get_midi_programs(midi))

# # Converts just a selected track
tokenizer.current_midi_metadata = {
    "time_division": midi.ticks_per_beat,
    "tempo_changes": midi.tempo_changes,
}
piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

# # And convert it back (the last arg stands for (program number, is drum))
converted_back_track, tempo_changes = tokenizer.tokens_to_track(
    piano_tokens, midi.ticks_per_beat, (0, False)
)

from miditoolkit import MidiFile
from miditok import MIDILike
from utils import writeToFile, EventToText

# TODO: Unfinished file! Bar calculations are wrong

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
nb_velocities = 5

additional_tokens = {
    "Chord": False,
    "Rest": False,
    "Tempo": False,
    "Program": True,
}

tokenizer = MIDILike(pitch_range, beat_res, nb_velocities, additional_tokens)

midi_filename = "the_strokes-reptilia"
midi = MidiFile(f"./midi/{midi_filename}.mid")

tokenizer.current_midi_metadata = {
    "time_division": midi.ticks_per_beat,
    "tempo_changes": midi.tempo_changes,
}

tokens = tokenizer.midi_to_tokens(midi)

event_to_text = EventToText()

piece_encoded_text = "PIECE_START"
for instrument in midi.instruments:
    piece_encoded_text += f" TRACK_START INST={instrument.program} BAR_START "
    inst_tokens = tokenizer.track_to_tokens(midi.instruments[0])
    tokens_events = tokenizer.tokens_to_events(inst_tokens)
    total_time_shift = 0

    for index, event in enumerate(tokens_events):

        if event.type == "Time-Shift":
            values = list(map(int, event.value.split(".")))
            total_time_shift += values[0] + (values[1] / 8)

            # TODO: Deal with notes clashing with bar finishings here
            if total_time_shift >= 4:
                piece_encoded_text += "BAR_END BAR_START "
                total_time_shift = 0

        piece_encoded_text += event_to_text.string(event)

writeToFile(f"./midi/{midi_filename}_text_mlike.txt", piece_encoded_text)

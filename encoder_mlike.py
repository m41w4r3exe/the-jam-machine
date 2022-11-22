from miditoolkit import MidiFile
from miditok import MIDILike
from utils import writeToFile, EventToText

# TODO: Unfinished file! Bar calculations are wrong

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
nb_velocities = 5
generate_by_N_bars = 4
time_signature = (4, 4)

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

track_length_in_bars = midi.max_tick / midi.ticks_per_beat / time_signature[0]

piece_encoded_text = "PIECE_START"

tokens_by_instruments_by_N_bars = list(range(len(midi.instruments)))
for ins, _ in enumerate(midi.instruments):
    tokens_by_instruments_by_N_bars[ins] = []

for inst_index, instrument in enumerate(midi.instruments):
    piece_encoded_text += f" TRACK_START INST={instrument.program} BAR_START "
    inst_tokens = tokenizer.track_to_tokens(midi.instruments[inst_index])
    tokens_events = tokenizer.tokens_to_events(inst_tokens)
    beat_count = 0
    bar_encoded_count = 0

    for event in tokens_events:
        if event.type == "Time-Shift":
            values = list(map(int, event.value.split(".")))
            beat_count += int(values[0] + (values[1] / 8))

            # # TODO: Deal with notes clashing with bar finishings here
            while beat_count > 4:
                bar_encoded_count += 1
                beat_count -= 4
                if bar_encoded_count == generate_by_N_bars:
                    break
                piece_encoded_text += "BAR_END "
                piece_encoded_text += "BAR_START "

            if beat_count == 4:
                bar_encoded_count += 1
                if bar_encoded_count == generate_by_N_bars:
                    break
                piece_encoded_text += "BAR_END "
                piece_encoded_text += "BAR_START "

            if beat_count < 4:
                event.value = (
                    str(beat_count) + "." + str(values[1]) + "." + str(values[2])
                )

        if beat_count < 4:
            piece_encoded_text += event_to_text.string(event)

        if bar_encoded_count == generate_by_N_bars:
            piece_encoded_text += "BAR_END TRACK_END "
            tokens_by_instruments_by_N_bars[inst_index].append([piece_encoded_text])
            piece_encoded_text = f"TRACK_START INST={instrument.program} BAR_START "
            bar_encoded_count = 0

# piece_encoded_text = "PIECE_END"
# writeToFile(f"./midi/{midi_filename}_text_mlike.txt", piece_encoded_text)

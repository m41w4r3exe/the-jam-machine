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
piece_events_in_bars = []
for instrument in midi.instruments:
    inst_tokens = tokenizer.track_to_tokens(instrument)
    midi_events = tokenizer.tokens_to_events(inst_tokens)

    track_encoded = f"TRACK_START INST={instrument.program} BAR_START "
    inst_events_in_bars = []
    bar_count = 0
    beat_count = 0
    for index, event in enumerate(midi_events):

        if event.type == "Time-Shift":
            values = list(map(int, event.value.split(".")))
            beat_count += values[0] + (values[1] / 8)

            while beat_count >= 4:
                beat_count -= 4
                values[0] = int(beat_count)
                event.value = ".".join(map(str, values))
                bar_count += 1
                if bar_count == 8:
                    break
                track_encoded += "BAR_END BAR_START "

            if bar_count == 8:
                bar_count = 0
                if beat_count == 0:
                    track_encoded += event_to_text.string(event)
                track_encoded += "BAR_END TRACK_END"
                inst_events_in_bars.append(track_encoded)
                track_encoded = f"TRACK_START INST={instrument.program} BAR_START "
                continue

        track_encoded += event_to_text.string(event)

    if bar_count != 0:
        track_encoded += "BAR_END BAR_START " * (8 - bar_count)
        track_encoded += "BAR_END TRACK_END"
        inst_events_in_bars.append(track_encoded)

    piece_events_in_bars.append(inst_events_in_bars)

print(piece_events_in_bars)


# writeToFile(f"./midi/{midi_filename}_text_mlike.txt", piece_encoded_text)

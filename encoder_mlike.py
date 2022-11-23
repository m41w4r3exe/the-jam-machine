from miditoolkit import MidiFile
from miditok import MIDILike
from utils import writeToFile, EventToText
from miditok import Event

midi_filename = "the_strokes-reptilia"
midi_filename = "1st Guitar"
midi = MidiFile(f"./midi/{midi_filename}.mid")

pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
tokenizer = MIDILike(pitch_range, beat_res)

midi_tokens = tokenizer.midi_to_tokens(midi)
midi_events = [tokenizer.tokens_to_events(inst_tokens) for inst_tokens in midi_tokens]


def remove_velocity(midi_events):
    return [
        [event for event in inst_events if event.type != "Velocity"]
        for inst_events in midi_events
    ]


def divide_timeshifts_by_bar(midi_events):
    new_midi_events = []
    for inst_events in midi_events:
        new_inst_events = []
        for event in inst_events:
            if event.type == "Time-Shift":
                values = list(map(int, event.value.split(".")))
                while values[0] > 4:
                    values[0] -= 4
                    new_inst_events.append(Event("Time-Shift", "4.0." + str(values[2])))
                values = ".".join(map(str, values))
                new_inst_events.append(Event("Time-Shift", values))
            else:
                new_inst_events.append(event)
        new_midi_events.append(new_inst_events)
    return new_midi_events


def add_bars(midi_events):
    new_midi_events = []
    for inst_events in midi_events:
        new_inst_events = [Event("Bar-Start", 1)]
        beat_count = 0
        bar_count = 1
        bar_end = False
        remainder_timeshift = None
        for i, event in enumerate(inst_events):

            if bar_end is True and event.type == "Note-Off":
                new_inst_events.append(event)
                continue

            if bar_end is True:
                bar_end = False
                new_inst_events.append(Event("Bar-End", bar_count))
                if i != len(inst_events) - 1:
                    bar_count += 1
                    new_inst_events.append(Event("Bar-Start", bar_count))
                    if remainder_timeshift is not None:
                        new_inst_events.append(remainder_timeshift)
                        remainder_timeshift = None

            if event.type == "Time-Shift":
                values = list(map(int, event.value.split(".")))
                beat_count += values[0] + (values[1] / values[2])

                if beat_count == 4:
                    beat_count = 0
                    bar_end = True

                if beat_count > 4:
                    beat_count -= 4
                    remainder_values = ".".join(
                        map(str, [int(beat_count), values[1], values[2]])
                    )
                    remainder_timeshift = Event("Time-Shift", remainder_values)
                    values[0] -= int(beat_count)
                    event.value = ".".join(map(str, values))
                    bar_end = True

                new_inst_events.append(event)

            else:
                new_inst_events.append(event)
        new_midi_events.append(new_inst_events)
    return new_midi_events


midi_events = remove_velocity(midi_events)
midi_events = divide_timeshifts_by_bar(midi_events)
midi_events = add_bars(midi_events)
# midi_events = get_section_texts(midi_events)

event_to_text = EventToText()
piece_encoded_text = "PIECE_START"
piece_text_in_bars = []

for instrument in midi.instruments:
    inst_tokens = tokenizer.track_to_tokens(instrument)
    midi_events = tokenizer.tokens_to_events(inst_tokens)

    track_encoded = f"TRACK_START INST={instrument.program} BAR_START "
    inst_events_in_bars = []
    bar_count = 0
    beat_count = 0
    bar_end = False
    for index, event in enumerate(midi_events):

        if event.type == "Time-Shift":
            values = list(map(int, event.value.split(".")))
            beat_count += values[0] + (values[1] / 8)

            # TODO: Deal with notes clashing with bar finishings here
            while beat_count >= 4:
                beat_count -= 4
                values[0] = int(beat_count)
                bar_count += 1
                if bar_count == 8:
                    break
                track_encoded += "BAR_END BAR_START "

            # Update beat count of next timeshift
            event.value = ".".join(map(str, values))

            if bar_count == 8:
                bar_count = 0
                bar_end = True
                if beat_count == 0:
                    track_encoded += event_to_text.string(event)
                # TODO: What happens when beat count > 4? problem: BAR_END is not added?
                if beat_count > 0:
                    remainder_timeshift = event_to_text.string(event)
                continue

        if remainder_timeshift is None:
            track_encoded += event_to_text.string(event)

        if bar_end is True and event.type == "Note-On":
            track_encoded += "BAR_END TRACK_END"
            inst_events_in_bars.append(track_encoded)
            bar_end = False
            track_encoded = f"TRACK_START INST={instrument.program} BAR_START "
            if remainder_timeshift is not None:
                track_encoded += remainder_timeshift + event_to_text.string(event)
                remainder_timeshift = None

    if bar_count != 0:
        track_encoded += "BAR_END BAR_START " * (8 - bar_count)
        track_encoded += "BAR_END TRACK_END "
        inst_events_in_bars.append(track_encoded)
    piece_text_in_bars.append(inst_events_in_bars)

print(piece_text_in_bars)


# writeToFile(f"./midi/{midi_filename}_text_mlike.txt", piece_encoded_text)

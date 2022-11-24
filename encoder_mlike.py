from miditoolkit import MidiFile
from miditok import MIDILike
from utils import writeToFile
from miditok import Event

midi_filename = "the_strokes-reptilia"
midi = MidiFile(f"./midi/{midi_filename}.mid")

pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
tokenizer = MIDILike(pitch_range, beat_res)

midi_tokens = tokenizer.midi_to_tokens(midi)
midi_events = [tokenizer.tokens_to_events(inst_tokens) for inst_tokens in midi_tokens]


def to_beat_str(value, base=8):
    values = [int(int(value * base) / base), int(int(value * base) % base), base]
    return ".".join(map(str, values))


def to_base10(value):
    integer, decimal, base = split_dots(value)
    return integer + decimal / base


def split_dots(value):
    return list(map(int, value.split(".")))


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
                values = split_dots(event.value)
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

            if (
                bar_end is True
                and event.type == "Note-Off"
                and remainder_timeshift is None
            ):
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
                timeshift_in_beats = to_base10(event.value)
                beat_count += timeshift_in_beats

                if beat_count == 4:
                    beat_count = 0
                    bar_end = True

                if beat_count > 4:
                    beat_count -= 4
                    event.value = to_beat_str(timeshift_in_beats - beat_count)
                    bar_end = True
                    remainder_timeshift = Event("Time-Shift", to_beat_str(beat_count))

            new_inst_events.append(event)
        new_midi_events.append(new_inst_events)
    return new_midi_events


def get_text(event):
    match event.type:
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


def make_sections(midi_events, instruments, n_bar=8):
    midi_sections = []
    for i, inst_events in enumerate(midi_events):
        inst_sections = []
        track_count = 1
        inst_sections += [
            Event("Track-Start", track_count),
            Event("Instrument", instruments[i].program),
        ]
        for event in inst_events:
            inst_sections.append(event)
            if event.type == "Bar-End" and int(event.value) % n_bar == 0:
                inst_sections += [
                    Event("Track-End", track_count),
                    Event("Track-Start", track_count + 1),
                    Event("Instrument", instruments[i].program),
                ]
                track_count += 1

        midi_sections.append(inst_sections)

    return midi_sections


def midi_to_text(midi_events):
    midi_section_texts = []
    for inst_events in midi_events:
        inst_sections = []
        track_text = ""
        for event in inst_events:

            if event.type == "Time-Shift" and event.value == "4.0.8":
                continue

            track_text += get_text(event)

            if event.type == "Track-End":
                inst_sections.append(track_text)
                track_text = ""

        midi_section_texts.append(inst_sections)

    return midi_section_texts


def get_piece_text(midi_text):
    piece_text = "PIECE_START "
    max_section_length = max(map(len, midi_text))
    for i in range(max_section_length):
        for inst_text in midi_text:
            if i < len(inst_text):
                piece_text += inst_text[i]

    return piece_text


def chain(start, *funcs):
    res = start
    for func in funcs:
        res = func(res)
    return res


# midi_events = remove_velocity(midi_events)
# midi_events = divide_timeshifts_by_bar(midi_events)
# midi_events = add_bars(midi_events)

midi_events = chain(midi_events, remove_velocity, divide_timeshifts_by_bar, add_bars)
midi_sections = make_sections(midi_events, midi.instruments)
midi_text = midi_to_text(midi_sections)
piece_text = get_piece_text(midi_text)

writeToFile(f"./midi/{midi_filename}_text_mlike.txt", piece_text)

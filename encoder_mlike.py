from miditoolkit import MidiFile
from miditok import MIDILike, Event
from utils import writeToFile, to_base10, to_beat_str, split_dots, chain


class MIDIEncoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def remove_velocity(midi_events):
        return [
            [event for event in inst_events if event.type != "Velocity"]
            for inst_events in midi_events
        ]

    @staticmethod
    def divide_timeshifts_by_bar(midi_events):
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = []
            for event in inst_events:
                if event.type == "Time-Shift":
                    values = split_dots(event.value)
                    while values[0] > 4:
                        values[0] -= 4
                        new_inst_events.append(
                            Event("Time-Shift", "4.0." + str(values[2]))
                        )
                    event.value = ".".join(map(str, values))
                new_inst_events.append(event)
            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def add_bars(midi_events):
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = [Event("Bar-Start", 1)]
            bar_count, beat_count = 1, 0
            bar_end, remainder_ts = False, None
            for i, event in enumerate(inst_events):

                if bar_end and event.type == "Note-Off" and remainder_ts is None:
                    new_inst_events.append(event)
                    continue

                if bar_end:
                    bar_end = False
                    new_inst_events.append(Event("Bar-End", bar_count))
                    if i != len(inst_events) - 1:
                        bar_count += 1
                        new_inst_events.append(Event("Bar-Start", bar_count))
                        if remainder_ts is not None:
                            new_inst_events.append(remainder_ts)
                            remainder_ts = None

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
                        remainder_ts = Event("Time-Shift", to_beat_str(beat_count))

                new_inst_events.append(event)
            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
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

    @staticmethod
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

    def events_to_text(self, midi_events):
        midi_section_texts = []
        for inst_events in midi_events:
            inst_sections = []
            track_text = ""
            for event in inst_events:

                if event.type == "Time-Shift" and event.value == "4.0.8":
                    continue

                track_text += self.get_text(event)

                if event.type == "Track-End":
                    inst_sections.append(track_text)
                    track_text = ""

            midi_section_texts.append(inst_sections)

        return midi_section_texts

    @staticmethod
    def sections_to_piece(midi_text):
        piece_text = "PIECE_START "
        max_section_length = max(map(len, midi_text))
        for i in range(max_section_length):
            for inst_text in midi_text:
                if i < len(inst_text):
                    piece_text += inst_text[i]

        return piece_text

    def get_midi_events(self, midi):
        return [
            self.tokenizer.tokens_to_events(inst_tokens)
            for inst_tokens in self.tokenizer.midi_to_tokens(midi)
        ]

    def get_piece_text(self, midi):
        midi_events = self.get_midi_events(midi)

        piece_text = chain(
            midi_events,
            [
                self.remove_velocity,
                self.divide_timeshifts_by_bar,
                self.add_bars,
                self.make_sections,
                self.events_to_text,
                self.sections_to_piece,
            ],
            midi.instruments,
        )

        return piece_text


midi_filename = "the_strokes-reptilia"
midi = MidiFile(f"./midi/{midi_filename}.mid")

pitch_range = range(21, 109)
beat_res = {(0, 400): 8}
tokenizer = MIDILike(pitch_range, beat_res)

piece_text = MIDIEncoder(tokenizer).get_piece_text(midi)

writeToFile(f"./midi/encoded_txts/{midi_filename}.txt", piece_text)

from miditoolkit import MidiFile
from miditok import MIDILike, Event
from utils import writeToFile, to_base10, to_beat_str, split_dots, chain, get_text

# TODO: Move remainder_ts logic to timeshift method
# TODO: Add comments
# TODO: Move midi read and text write to a seperate file


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
    def make_sections(midi_events, instruments, n_bar=8):
        midi_sections = []
        for i, inst_events in enumerate(midi_events):
            inst_sections = []
            track_count = 1
            section = [
                Event("Track-Start", track_count),
                Event("Instrument", instruments[i].program),
            ]
            for event in inst_events:
                section.append(event)
                if event.type == "Bar-End" and int(event.value) % n_bar == 0:
                    section.append(Event("Track-End", track_count))
                    inst_sections.append(section)
                    track_count += 1
                    section = [
                        Event("Track-Start", track_count),
                        Event("Instrument", instruments[i].program),
                    ]

            midi_sections.append(inst_sections)

        return midi_sections

    @staticmethod
    def sections_to_piece(midi_events):
        piece = [Event("Piece-Start", 1)]
        max_total_sections = max(map(len, midi_events))
        for i in range(max_total_sections):
            for inst_events in midi_events:
                if i < len(inst_events):
                    piece += inst_events[i]
        return piece

    @staticmethod
    def events_to_text(piece_events):
        piece_text = ""
        for event in piece_events:
            if event.type == "Time-Shift" and event.value == "4.0.8":
                continue

            piece_text += get_text(event)
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
                self.sections_to_piece,
                self.events_to_text,
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

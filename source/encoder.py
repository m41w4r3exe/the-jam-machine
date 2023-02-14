from miditoolkit import MidiFile
from miditok import Event
from utils import *
import numpy as np
from scipy import stats

# TODO: Move remainder_ts logic to divide_timeshift method
# TODO: Add method comments
# TODO: Fix beat resolution and its string representation
# TODO: Make instruments family while encoding
# TODO: Add density bins:
# Question: How to determine difference between 8 very long notes in 8 bar and 6 empty bar + 8 very short notes in last 2 bar?
# TODO: Data augmentation: hopping 1 bar and re-encode almost same notes
# TODO: Data augmentation: octave or pitch shift?
# TODO: Solve the one-instrument tracks problem


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
    def add_note_density_in_bar(midi_events):
        """
        For each bar:
        - calculate the note density as the number of note onset divided by the number of beats per bar
        - add the note density as a new event type "Bar-Density"
        """
        beats_per_bar = 4
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = []
            for event in inst_events:

                if event.type == "Bar-Start":
                    note_onset_count_in_bar = 0  # initialize not count
                    new_inst_events.append(event)  # append Bar-Start event
                    temp_event_list = []  # initialize the temporary event list
                else:
                    temp_event_list.append(event)

                if event.type == "Note-On":
                    note_onset_count_in_bar += 1

                if event.type == "Bar-End":
                    new_inst_events.append(
                        Event(
                            "Bar-Density",
                            round(note_onset_count_in_bar / beats_per_bar),
                        )
                    )
                    [
                        new_inst_events.append(temp_event)
                        for temp_event in temp_event_list
                    ]

            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def make_sections(midi_events, instruments, n_bar=8):
        """For each instrument, make sections of n_bar bars each"""
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
    def add_density_to_sections(midi_sections):
        """
        Add density to each section as the mode of bar density within that section
        """
        new_midi_sections = []
        note_count_distribution = []
        for inst_sections in midi_sections:
            new_inst_sections = []
            for section in inst_sections:
                for i, event in enumerate(section):
                    if event.type == "Bar-Density":
                        note_count_distribution.append(event.value)
                # add section density -> set to mode of bar density within that section
                density = stats.mode(
                    np.array(note_count_distribution).astype(np.int16)
                )[0][0]

                for i, event in enumerate(section):
                    if event.type == "Instrument":
                        section.insert(i + 1, Event("Density", density))
                        break
                new_inst_sections.append(section)
            new_midi_sections.append(new_inst_sections)
        return new_midi_sections

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

    @staticmethod
    def get_bar_density(bar):
        bar_density = []
        return bar_density

    @staticmethod
    def aggregate_density(piece_events):
        instrument_density = []
        return instrument_density

    @staticmethod
    def add_density_event(piece_events):

        return piece_events

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
                self.add_note_density_in_bar,
                self.make_sections,
                self.add_density_to_sections,
                self.sections_to_piece,
                self.events_to_text,
            ],
            midi.instruments,
        )

        return piece_text

    def get_piece_text_by_section(self, midi):
        midi_events = self.get_midi_events(midi)

        sectioned_instruments = chain(
            midi_events,
            [
                self.remove_velocity,
                self.divide_timeshifts_by_bar,
                self.add_bars,
                self.make_sections,
            ],
            midi.instruments,
        )

        # sectioned_intruments_as_text = [
        #     list(map(self.events_to_text, sections))
        #     for sections in sectioned_instruments
        # ]

        max_sections = max(list(map(len, sectioned_instruments)))
        sections_as_text = ["" for _ in range(max_sections)]

        for sections in sectioned_instruments:
            for idx in range(max_sections):
                try:
                    sections_as_text[idx] += self.events_to_text(sections[idx])
                except:
                    pass

        return sections_as_text


def from_MIDI_to_sectionned_text(midi_filename):
    """convert a MIDI file to a MidiText input prompt"""
    midi = MidiFile(f"{midi_filename}.mid")
    midi_like = get_miditok()
    piece_text = MIDIEncoder(midi_like).get_piece_text(midi)
    piece_text_split_by_section = MIDIEncoder(midi_like).get_piece_text_by_section(midi)
    return piece_text


if __name__ == "__main__":
    # Encode Strokes for debugging purposes:
    midi_filename = "the_strokes-reptilia"
    piece_text = from_MIDI_to_sectionned_text(f"midi/{midi_filename}")
    writeToFile(f"midi/encoded_txts/{midi_filename}.txt", piece_text)

from miditoolkit import MidiFile
from miditok import Event
from utils import *
import numpy as np
from scipy import stats

# TODO: Move remainder_ts logic to divide_timeshift method
# TODO: Add method comments
# TODO: Make instruments family while encoding
# TODO: Density Bins - (Done)
# Question: How to determine difference between 8 very long notes in 8 bar and 6 empty bar + 8 very short notes in last 2 bar?
# TODO: Data augmentation: hopping 4 bars and re-encode almost same notes
# TODO: Data augmentation: octave or pitch shift? both?
# TODO: Solve the one-instrument tracks problem - > needs a external function that converts the one track midi to multi-track midi based on the "channel information"
# TODO: Solve the one instrument spread to many channels problem -> it creates several intruments instead of one
# TODO: keep track of bar count and track count when encoding (at least in initial steps)
# TODO: Add seperate encoding methods:
#               - track by track, so each instrument is one after the other,
#               - section by section for training
# TODO: empty sections should be filled with bar start and bar end events


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
            new_inst_events = [Event("Bar-Start", 0)]
            bar_index, beat_count = 0, 0
            bar_end, remainder_ts = False, None
            for i, event in enumerate(inst_events):

                if bar_end:
                    if event.type == "Note-Off" and remainder_ts is None:
                        new_inst_events.append(event)
                        if (
                            i == len(inst_events) - 1
                        ):  # if is the last event, bar end needs to be added here
                            new_inst_events.append(Event("Bar-End", bar_index))
                        continue

                    else:
                        bar_end = False
                        new_inst_events.append(Event("Bar-End", bar_index))
                        if i != len(inst_events) - 1:  ## Why this condition here?
                            bar_index += 1
                            new_inst_events.append(Event("Bar-Start", bar_index))
                            if remainder_ts is not None:
                                # adding the previous bar remainder at the beginning of the new bar
                                new_inst_events.append(remainder_ts)
                                remainder_ts = None

                if event.type == "Time-Shift":
                    timeshift_in_beats = int_dec_base_to_beat(event.value)
                    beat_count += timeshift_in_beats

                    if beat_count == 4:
                        beat_count = 0
                        bar_end = True

                    if beat_count > 4:
                        beat_count -= 4
                        event.value = beat_to_int_dec_base(
                            timeshift_in_beats - beat_count
                        )
                        bar_end = True
                        # saving the remainder as an event for the next bar
                        remainder_ts = Event(
                            "Time-Shift", beat_to_int_dec_base(beat_count)
                        )

                new_inst_events.append(event)

            new_midi_events.append(new_inst_events)

        return new_midi_events

    @staticmethod
    def add_density_to_bar(midi_events):
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
        """For each instrument, make sections of n_bar bars each
        --> midi_sections[inst_sections][sections]
        because files can be encoded in many sections of n_bar"""

        midi_sections = []
        for i, inst_events in enumerate(midi_events):
            inst_section = []
            track_index = 0
            section = [
                Event("Track-Start", track_index),
                Event("Instrument", instruments[i].program),
            ]
            for event in inst_events:
                section.append(event)
                if event.type == "Bar-End" and int(event.value + 1) % n_bar == 0:
                    # finish the section with track-end event
                    section.append(Event("Track-End", track_index))
                    # append the section to the section list
                    inst_section.append(section)
                    track_index += 1
                    # start new section
                    section = [
                        Event("Track-Start", track_index),
                        Event("Instrument", instruments[i].program),
                    ]

            midi_sections.append(inst_section)

        return midi_sections

    @staticmethod
    def add_density_to_sections(midi_sections):
        """
        Add density to each section as the mode of bar density within that section
        """
        new_midi_sections = []
        for inst_sections in midi_sections:
            new_inst_sections = []
            for section in inst_sections:
                note_count_distribution = []
                for i, event in enumerate(section):
                    if event.type == "Instrument":
                        instrument_token_location = i
                    if event.type == "Bar-Density":
                        note_count_distribution.append(event.value)

                # add section density -> set to mode of bar density within that section
                density = stats.mode(
                    np.array(note_count_distribution).astype(np.int16)
                )[0][0]
                section.insert(instrument_token_location + 1, Event("Density", density))
                new_inst_sections.append(section)

            new_midi_sections.append(new_inst_sections)

        return new_midi_sections

    @staticmethod
    def sections_to_piece(midi_events):
        """Combine all sections into one piece
        Section are combined in a string as follows:
        'Piece_Start -
        Section 1 Instrument 1
        Section 1 Instrument 2
        Section 1 Instrument 3
        Section 2 Instrument 1
        ...'
        """
        piece = []
        max_total_sections = max(map(len, midi_events))
        for i in range(max_total_sections):
            # adding piece start event at the beggining of each section
            piece += [Event("Piece-Start", 1)]
            for inst_events in midi_events:
                nb_inst_section = len(inst_events)
                if i < nb_inst_section:
                    piece += inst_events[i]
        return piece

    @staticmethod
    def events_to_text(events):
        """Convert miditok events to text"""
        text = ""
        current_instrument = "undefined"
        for event in events:
            if event.type == "Time-Shift" and event.value == "4.0.8":
                # if event.value == "4.0.8": then it means that it is just an empty bar
                continue

            # keeping track of the instrument to set the quantization in get_text()
            if event.type == "Instrument":
                current_instrument = str(event.value)

            text += get_text(event, current_instrument)
        return text

    def get_midi_events(self, midi):
        return [
            self.tokenizer.tokens_to_events(inst_tokens)
            for inst_tokens in self.tokenizer.midi_to_tokens(midi)
        ]

    def get_piece_sections(self, midi):
        """Modifies the miditok events to our needs:
        Removes velocity, add bars, density and make sections for training and generation
        Args:
            - midi: miditok object
        Returns:
            - piece_sections: list (instruments) of lists (sections) of miditok events"""

        midi_events = self.get_midi_events(midi)

        piece_sections = chain(
            midi_events,
            [
                self.remove_velocity,
                self.divide_timeshifts_by_bar,
                self.add_bars,
                self.add_density_to_bar,
                self.make_sections,
                self.add_density_to_sections,
            ],
            midi.instruments,
        )

        return piece_sections

    def get_piece_text(self, midi):
        """Converts the miditok events to text,
        The text is organized in sections of 8 bars of instrument
        Args:
            - midi: miditok object
        Returns:
            - piece_text: string"""

        piece_text = chain(
            midi,
            [
                self.get_piece_sections,
                self.sections_to_piece,
                self.events_to_text,
            ],
            midi.instruments,
        )

        return piece_text

    def get_text_by_section(self, midi):
        """Returns a list of sections of text
        Args:
            midi: miditok object
        Returns:
            sections_as_text: list of sections of text
        """
        sectioned_instruments = self.get_piece_sections(midi)
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
    piece_text_split_by_section = MIDIEncoder(midi_like).get_text_by_section(midi)
    return piece_text


if __name__ == "__main__":
    # Encode Strokes for debugging purposes:
    midi_filename = "the_strokes-reptilia"
    piece_text = from_MIDI_to_sectionned_text(f"midi/{midi_filename}")
    writeToFile(f"midi/encoded_txts/{midi_filename}.txt", piece_text)

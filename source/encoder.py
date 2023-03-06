from miditoolkit import MidiFile
from miditok import Event
from utils import *
import numpy as np
from scipy import stats
from familizer import Familizer
from constants import BEATS_PER_BAR

# TODO HIGH PRIORITY
# TODO: Make instruments family while encoding
# TODO: Data augmentation:
#   - hopping K bars and re-encode almost same notes: needs to keep track of sequence length
#   - computing track key
#       - octave shifting
#       - pitch shifting
# TODO: Solve the one-instrument tracks problem - > needs a external function that converts the one track midi to multi-track midi based on the "channel information"
# TODO: Solve the one instrument spread to many channels problem -> it creates several intruments instead of one

# LOW PRIORITY
# TODO: Improve method comments
# TODO: Density Bins - Calculation Done - Not sure if it the best way - MMM paper uses a normalized density based on the entire instrument density in the dataset.
# They say that density for a given instrument does not mean the same for another. However, I am expecting that the instrument token is already implicitely taking care of that.
# Question: How to determine difference between 8 very long notes in 8 bar and 6 empty bar + 8 very short notes in last 2 bar?
# TODO: Should empty sections be filled with bar start and bar end events?
# TODO: changing the methods to avoid explicit loops and use the map function instead?

# NEW IDEAS
# TODO: Changing Generation approach : encoding all tracks in the same key and choose the key while generating, so we just shift the key after generation.


class MIDIEncoder:
    def __init__(self, tokenizer, familized=False):
        self.tokenizer = tokenizer
        self.familized = familized

    @staticmethod
    def remove_velocity(midi_events):
        return [
            [event for event in inst_events if event.type != "Velocity"]
            for inst_events in midi_events
        ]

    @staticmethod
    def set_timeshifts_to_min_length(midi_events):
        """convert existing time-shifts events to multiple time-shift events,
        which sum equals the original time shift event
        --> Simplifies the bar encoding process"""

        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = []
            for event in inst_events:
                if event.type == "Time-Shift":
                    values = split_dots(event.value)
                    # transfer values[0] to values[1]
                    values[1] += values[0] * values[2]
                    values[0] = 0
                    # generating and appending new time-shift events
                    while values[1] > 1:
                        values[1] -= 1
                        new_inst_events.append(
                            Event("Time-Shift", "0.1." + str(values[2]))
                        )
                    event.value = ".".join(map(str, values))

                new_inst_events.append(event)
            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def add_bars(midi_events):
        """Adding bar-start and bar-end events to the midi events
        Uses BEATS_PER_BAR constant to determine the bar length
        """
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = [Event("Bar-Start", 0)]
            bar_index, beat_count = 0, 0
            bar_end = False
            for i, event in enumerate(inst_events):

                # when bar_end reached, adding the remainder note-off events
                # adding bar end event and bar start event
                # only if event is not the last event of the track
                if bar_end and i != len(inst_events) - 1:
                    if event.type == "Note-Off":
                        new_inst_events.append(event)
                        continue

                    else:
                        new_inst_events.append(Event("Bar-End", bar_index))
                        bar_index += 1
                        new_inst_events.append(Event("Bar-Start", bar_index))
                        bar_end = False

                # keeping track of the beat count within the bar
                if event.type == "Time-Shift":
                    beat_count += int_dec_base_to_beat(event.value)
                    if beat_count == BEATS_PER_BAR:
                        beat_count = 0
                        bar_end = True
                # default
                new_inst_events.append(event)
                # adding the last bar-end event
                if i == len(inst_events) - 1:
                    new_inst_events.append(Event("Bar-End", bar_index))

            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def combine_timeshifts_in_bar(midi_events):
        """Combining adjacent time-shifts within the same bar"""
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = []
            aggregated_beats = 0
            for event in inst_events:
                # aggregating adjacent time-shifts and skipping them
                if event.type == "Time-Shift":
                    aggregated_beats += int_dec_base_to_beat(event.value)
                    continue
                # writting the aggregating time shift as a new event
                if aggregated_beats > 0:
                    new_inst_events.append(
                        Event("Time-Shift", beat_to_int_dec_base(aggregated_beats))
                    )
                    aggregated_beats = 0
                # default
                new_inst_events.append(event)
            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def remove_timeshifts_preceeding_bar_end(midi_events):
        """Useless time-shift removed, i.e. when bar are empty, or afgter the last event of a bar is there is a remainder time-shift
        This helps reducing the sequence length"""
        new_midi_events = []
        for inst_events in midi_events:
            new_inst_events = []
            for i, event in enumerate(inst_events):
                if (
                    i <= len(inst_events) - 1
                    and event.type == "Time-Shift"
                    and inst_events[i + 1].type == "Bar-End"
                ):
                    print(f"---- {i} - {event} ----")
                    [print(a) for a in inst_events[i - 3 : i + 3]]
                    continue

                new_inst_events.append(event)
            inst_events
            new_midi_events.append(new_inst_events)

        return new_midi_events

    @staticmethod
    def add_density_to_bar(midi_events):
        """
        For each bar:
        - calculate the note density as the number of note onset divided by the number of beats per bar
        - add the note density as a new event type "Bar-Density"
        """
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
                            round(note_onset_count_in_bar / BEATS_PER_BAR),
                        )
                    )
                    [
                        new_inst_events.append(temp_event)
                        for temp_event in temp_event_list
                    ]

            new_midi_events.append(new_inst_events)
        return new_midi_events

    @staticmethod
    def define_instrument(midi_tok_instrument, familize=False):
        familize_instrument = False
        """Define the instrument token from the midi token instrument and whether the instrument needs to be famnilized"""
        # get program number
        instrument = (
            midi_tok_instrument.program if not midi_tok_instrument.is_drum else "Drums"
        )
        # familize instrument
        if familize_instrument and not midi_tok_instrument.is_drum:
            familizer = Familizer()
            instrument = familizer.get_family_number(instrument)

        return instrument

    @staticmethod
    def initiate_track_in_section(instrument, track_index):
        section = [
            Event("Track-Start", track_index),
            Event("Instrument", instrument),
        ]
        return section

    @staticmethod
    def terminate_track_in_section(section, track_index):
        section.append(Event("Track-End", track_index))
        track_index += 1
        return section, track_index

    def make_sections(self, midi_events, instruments, n_bar=8):
        """For each instrument, make sections of n_bar bars each
        --> midi_sections[inst_sections][sections]
        because files can be encoded in many sections of n_bar"""

        midi_sections = []
        for i, inst_events in enumerate(midi_events):
            inst_section = []
            track_index = 0
            instrument = self.define_instrument(instruments[i], familize=self.familized)
            section = self.initiate_track_in_section(instrument, track_index)
            for ev_idx, event in enumerate(inst_events):
                section.append(event)
                if ev_idx == len(inst_events) - 1 or (
                    event.type == "Bar-End" and int(event.value + 1) % n_bar == 0
                ):
                    # finish the section with track-end event
                    section, track_index = self.terminate_track_in_section(
                        section, track_index
                    )
                    # append the section to the section list
                    inst_section.append(section)

                    # start new section if not the last event
                    if ev_idx < len(inst_events) - 1:
                        section = self.initiate_track_in_section(
                            instrument, track_index
                        )

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
                    np.array(note_count_distribution).astype(np.int16), keepdims=False
                )[0]
                section.insert(instrument_token_location + 1, Event("Density", density))
                new_inst_sections.append(section)

            new_midi_sections.append(new_inst_sections)

        return new_midi_sections

    @staticmethod
    def sections_to_piece(midi_events):
        """Combine all sections into one piece
        Section are combined in a string as follows:
        'Piece_Start
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
                self.set_timeshifts_to_min_length,
                self.add_bars,
                self.combine_timeshifts_in_bar,
                self.remove_timeshifts_preceeding_bar_end,
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
    piece_text = MIDIEncoder(midi_like, familized=True).get_piece_text(midi)
    piece_text_split_by_section = MIDIEncoder(midi_like).get_text_by_section(midi)
    return piece_text


if __name__ == "__main__":
    # Encode Strokes for debugging purposes:
    # midi_filename = "midi/the_strokes-reptilia"
    midi_filename = "source/tests/20230306_140430"
    piece_text = from_MIDI_to_sectionned_text(f"{midi_filename}")
    writeToFile(f"{midi_filename}.txt", piece_text)

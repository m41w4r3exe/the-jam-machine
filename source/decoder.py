from utils import *
from familizer import Familizer
from miditok import Event

# TODO: Decode quantized time properly
# TODO: how to decode 2 similar instrument sinto 2 tracks.


class TextDecoder:
    """Decodes text into:
    1- List of events
    2- Then converts these events to midi file via MidiTok and miditoolkit

    :param tokenizer: from MidiTok

    Usage with write_to_midi method:
        args: text(String) example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END
        returns: midi file from miditoolkit
    """

    def __init__(self, tokenizer, familized=True):
        self.tokenizer = tokenizer
        self.familized = familized

    def decode(self, text):
        r"""converts from text to instrument events
        Args:
            text (String): example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END

        Returns:
            Dict{inst_id: List[Events]}: List of events of Notes with velocities, aggregated Timeshifts, for each instrument
        """
        piece_events = self.text_to_events(text)
        piece_events = self.get_track_ids(piece_events)
        inst_events = self.piece_to_inst_events(piece_events)
        inst_events = self.get_bar_ids(inst_events)
        events = self.add_missing_timeshifts_in_a_bar(inst_events)
        events = self.remove_unwanted_tokens(events)
        events = self.aggregate_timeshifts(events)
        events = self.add_velocity(events)
        return events

    def tokenize(self, events):
        r"""converts from events to MidiTok tokens
        Args:
            events (Dict{inst_id: List[Events]}): List of events for each instrument

        Returns:
            List[List[Events]]: List of tokens for each instrument
        """
        tokens = []
        for inst in events:
            tokens.append(self.tokenizer.events_to_tokens(inst["events"]))
        return tokens

    def get_midi(self, text, filename=None):
        r"""converts from text to midi
        Args:
            text (String): example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END

        Returns:
            miditoolkit midi: Returns and writes to midi
        """
        events = self.decode(text)
        tokens = self.tokenize(events)
        instruments = self.get_instruments_tuple(events)
        midi = self.tokenizer.tokens_to_midi(tokens, instruments)

        if filename is not None:
            midi.dump(f"{filename}")
            print(f"midi file written: {filename}")

        return midi

    @staticmethod
    def text_to_events(text):
        events = []
        intrument = "Drums"
        cumul_time_delta = 0
        for word in text.split(" "):
            _event = word.split("=")
            value = _event[1] if len(_event) > 1 else None

            if _event[0] == "INST":
                intrument = get_event(_event[0], value).value

            event = get_event(_event[0], value, intrument)

            if event:
                events.append(event)

        return events

    @staticmethod
    def get_track_ids(events):
        """Adding tracking the track id for each track start and end event"""
        track_id = 0
        for i, event in enumerate(events):
            if event.type == "Track-Start":
                events[i].value = track_id
            if event.type == "Track-End":
                events[i].value = track_id
                track_id += 1
        return events

    @staticmethod
    def piece_to_inst_events(piece_events):
        """Converts piece events of 8 bars to instrument events for entire song

        Args:
            piece_events (List[Events]): List of events of Notes, Timeshifts, Bars, Tracks

        Returns:
            Dict{inst_id: List[Events]}: List of events for each instrument

        """
        inst_events = []
        current_track = -1  # so does not start before Track-Start is encountered
        for event in piece_events:
            # creates a new entry in the dictionnary when "Track-Start" event is encountered
            if event.type == "Track-Start":
                current_track = event.value
                if len(inst_events) <= event.value:
                    inst_events.append({})
                    inst_events[current_track]["channel"] = current_track
                    inst_events[current_track]["events"] = []
            # append event to the track
            if current_track != -1:
                inst_events[current_track]["events"].append(event)

            if event.type == "Instrument":
                inst_events[current_track]["Instrument"] = event.value
        # TODO: needs cleaning Track-start and track end
        return inst_events

    @staticmethod
    def get_bar_ids(inst_events):
        """tracking bar index for each intrument and saving them in the miditok Events"""
        for inst_index, inst_event in enumerate(inst_events):
            bar_idx = 0
            for event_index, event in enumerate(inst_event["events"]):
                if event.type == "Bar-Start" or event.type == "Bar-End":
                    inst_events[inst_index]["events"][event_index].value = bar_idx
                if event.type == "Bar-End":
                    bar_idx += 1
        return inst_events

    @staticmethod
    def add_missing_timeshifts_in_a_bar(inst_events, beat_per_bar=4, verbose=True):
        """Add missing time shifts in bar to make sure that each bar has 4 beats
        takes care of the problem of a missing time shift if notes do not last until the end of the bar
        takes care of the problem of empty bars that are only defined by "BAR_START BAR END"""
        new_inst_events = []
        for index, inst_event in enumerate(inst_events):
            new_inst_events.append({})
            new_inst_events[index]["Instrument"] = inst_event["Instrument"]
            new_inst_events[index]["channel"] = index
            new_inst_events[index]["events"] = []

            for event in inst_event["events"]:
                if event.type == "Bar-Start":
                    beat_count = 0

                if event.type == "Time-Shift":
                    beat_count += int_dec_base_to_beat(event.value)

                if event.type == "Bar-End" and beat_count < beat_per_bar:
                    time_shift_to_add = beat_to_int_dec_base(beat_per_bar - beat_count)
                    new_inst_events[index]["events"].append(
                        Event("Time-Shift", time_shift_to_add)
                    )
                    beat_count += int_dec_base_to_beat(time_shift_to_add)

                if event.type == "Bar-End" and verbose == True:
                    print(
                        f"Instrument {index} - {inst_event['Instrument']} - Bar {event.value} - beat_count = {beat_count}"
                    )
                if event.type == "Bar-End" and beat_count > beat_per_bar:
                    print(
                        f"Instrument {index} - {inst_event['Instrument']} - Bar {event.value} - Beat count exceeded "
                    )
                new_inst_events[index]["events"].append(event)

        return new_inst_events

    @staticmethod
    def remove_unwanted_tokens(events):
        for inst_index, inst_event in enumerate(events):
            new_inst_event = []
            for event in inst_event["events"]:
                if not (
                    event.type == "Bar-Start"
                    or event.type == "Bar-End"
                    or event.type == "Track-Start"
                    or event.type == "Track-End"
                    or event.type == "Piece-Start"
                    or event.type == "Instrument"
                ):
                    new_inst_event.append(event)
            # replace the events list with the new one
            events[inst_index]["events"] = new_inst_event
        return events

    @staticmethod
    def add_timeshifts(beat_values1, beat_values2):
        """Adds two beat values

        Args:
            beat_values1 (String): like 0.3.8
            beat_values2 (String): like 1.7.8

        Returns:
            beat_str (String): added beats like 2.2.8 for example values
        """
        value1 = int_dec_base_to_beat(beat_values1)
        value2 = int_dec_base_to_beat(beat_values2)
        return beat_to_int_dec_base(value1 + value2)

    def aggregate_timeshifts(self, events):
        """Aggregates consecutive time shift events bigger than a bar
        -> like Timeshift 4.0.8

        Args:
            events (_type_): _description_

        Returns:
            _type_: _description_
        """
        for inst_index, inst_event in enumerate(events):
            new_inst_event = []
            for event in inst_event["events"]:
                if (
                    event.type == "Time-Shift"
                    and len(new_inst_event) > 0
                    and new_inst_event[-1].type == "Time-Shift"
                ):
                    new_inst_event[-1].value = self.add_timeshifts(
                        new_inst_event[-1].value, event.value
                    )
                else:
                    new_inst_event.append(event)

            events[inst_index]["events"] = new_inst_event
        return events

    @staticmethod
    def add_velocity(events):
        """Adds default velocity 99 to note events since they are removed from text, needed to generate midi"""
        for inst_index, inst_event in enumerate(events):
            new_inst_event = []
            for inst_event in inst_event["events"]:
                new_inst_event.append(inst_event)
                if inst_event.type == "Note-On":
                    new_inst_event.append(Event("Velocity", 99))
            events[inst_index]["events"] = new_inst_event
        return events

    def get_instruments_tuple(self, events):
        """Returns instruments tuple for midi generation"""
        instruments = []
        for track in events:
            is_drum = 0
            if track["Instrument"] == "DRUMS":
                track["Instrument"] = 0
                is_drum = 1
            if self.familized:
                track["Instrument"] = Familizer(arbitrary=True).get_program_number(
                    int(track["Instrument"])
                )
            instruments.append((int(track["Instrument"]), is_drum))
        return tuple(instruments)


if __name__ == "__main__":

    filename = "midi/generated/JammyMachina/elec-gmusic-familized-model-13-12__17-35-53/20221218_164700"
    encoded_json = readFromFile(
        f"{filename}.json",
        True,
    )
    encoded_text = encoded_json["generate_midi"]
    # encoded_text = "PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=69 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=69 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=57 TIME_DELTA=1 NOTE_OFF=57 NOTE_ON=56 TIME_DELTA=1 NOTE_OFF=56 NOTE_ON=64 NOTE_ON=60 NOTE_ON=55 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=59 NOTE_ON=55 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=59 NOTE_OFF=50 NOTE_OFF=55 NOTE_OFF=50 BAR_END BAR_START BAR_END TRACK_END"

    miditok = get_miditok()
    TextDecoder(miditok).get_midi(encoded_text, filename=filename)

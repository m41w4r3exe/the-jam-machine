from utils import (
    readFromFile,
    get_event,
    to_base10,
    to_beat_str,
    get_datetime,
    get_tokenizer,
)
from miditok import Event


class TextDecoder:
    """Decodes text into:
    1- List of events
    2- Then converts these events to midi file via MidiTok and miditoolkit

    :param tokenizer: from MidiTok

    Usage with write_to_midi method:
        args: text(String) example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END
        returns: midi file from miditoolkit
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, text):
        r"""converts from text to instrument events
        Args:
            text (String): example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END

        Returns:
            Dict{inst_id: List[Events]}: List of events of Notes with velocities, aggregated Timeshifts, for each instrument
        """
        piece_events = self.text_to_events(text)
        inst_events = self.piece_to_inst_events(piece_events)
        events = self.add_timeshifts_for_empty_bars(inst_events)
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
        for inst in events.keys():
            tokens.append(self.tokenizer.events_to_tokens(events[inst]))
        return tokens

    def write_to_midi(self, text, filename=None):
        r"""writes text to midi file
        Args:
            text (String): example ->  PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50...BAR_END TRACK_END

        Returns:
            miditoolkit midi: Returns and writes to midi
        """
        if filename is None:
            raise Exception("path_filename required")
        events = self.decode(text)
        tokens = self.tokenize(events)
        instruments = self.get_instruments_tuple(events)
        midi = self.tokenizer.tokens_to_midi(tokens, instruments)
        midi.dump(f"{filename}.mid")
        print(f"midi file written: {filename}.mid")

    @staticmethod
    def text_to_events(text):
        events = []
        for word in text.split(" "):
            # TODO: Handle bar and track values with a counter
            _event = word.split("=")
            value = _event[1] if len(_event) > 1 else None
            event = get_event(_event[0], value)
            if event:
                events.append(event)
        return events

    @staticmethod
    def piece_to_inst_events(piece_events):
        """Converts piece events of 8 bars to instrument events for entire song

        Args:
            piece_events (List[Events]): List of events of Notes, Timeshifts, Bars, Tracks

        Returns:
            Dict{inst_id: List[Events]}: List of events for each instrument

        """
        inst_events = {}
        current_instrument = -1
        for event in piece_events:
            if event.type == "Instrument":
                current_instrument = event.value
                if current_instrument not in inst_events:
                    inst_events[current_instrument] = []
            elif current_instrument != -1:
                inst_events[current_instrument].append(event)
        return inst_events

    @staticmethod
    def add_timeshifts_for_empty_bars(inst_events):
        """Adds time shift events instead of consecutive [BAR_START BAR_END] events"""
        new_inst_events = {}
        for inst, events in inst_events.items():
            new_inst_events[inst] = []
            for index, event in enumerate(events):
                if event.type == "Bar-End" or event.type == "Bar-Start":
                    if events[index - 1].type == "Bar-Start":
                        new_inst_events[inst].append(Event("Time-Shift", "4.0.8"))
                else:
                    new_inst_events[inst].append(event)
        return new_inst_events

    @staticmethod
    def add_timeshifts(beat_values1, beat_values2):
        """Adds two beat values

        Args:
            beat_values1 (String): like 0.3.8
            beat_values2 (String): like 1.7.8

        Returns:
            beat_str (String): added beats like 2.2.8 for example values
        """
        value1 = to_base10(beat_values1)
        value2 = to_base10(beat_values2)
        return to_beat_str(value1 + value2)

    def aggregate_timeshifts(self, events):
        """Aggregates consecutive time shift events bigger than a bar
        -> like Timeshift 4.0.8

        Args:
            events (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_events = {}
        for inst, events in events.items():
            inst_events = []
            for i, event in enumerate(events):
                if (
                    event.type == "Time-Shift"
                    and len(inst_events) > 0
                    and inst_events[-1].type == "Time-Shift"
                ):
                    inst_events[-1].value = self.add_timeshifts(
                        inst_events[-1].value, event.value
                    )
                else:
                    inst_events.append(event)
            new_events[inst] = inst_events
        return new_events

    @staticmethod
    def add_velocity(events):
        """Adds default velocity 99 to note events since they are removed from text, needed to generate midi"""
        new_events = {}
        for inst, events in events.items():
            inst_events = []
            for event in events:
                inst_events.append(event)
                if event.type == "Note-On":
                    inst_events.append(Event("Velocity", 99))
            new_events[inst] = inst_events
        return new_events

    @staticmethod
    def get_instruments_tuple(events):
        """Returns instruments tuple for midi generation"""
        instruments = []
        for inst in events.keys():
            is_drum = 0
            if inst == "DRUMS":
                inst = 0
                is_drum = 1
            instruments.append((int(inst), is_drum))
        return tuple(instruments)


if __name__ == "__main__":

    path_filename = "midi/generated/misnaej/the-jam-machine/20221206_170556"
    encoded_json = readFromFile(
        f"{path_filename}.json",
        True,
    )
    encoded_text = encoded_json["sequence"]
    # encoded_text = "PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=69 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=69 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=57 TIME_DELTA=1 NOTE_OFF=57 NOTE_ON=56 TIME_DELTA=1 NOTE_OFF=56 NOTE_ON=64 NOTE_ON=60 NOTE_ON=55 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=59 NOTE_ON=55 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=59 NOTE_OFF=50 NOTE_OFF=55 NOTE_OFF=50 BAR_END BAR_START BAR_END TRACK_END"

    tokenizer = get_tokenizer()
    TextDecoder(tokenizer).write_to_midi(encoded_text, path_filename=path_filename)

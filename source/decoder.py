# from encoder_mlike import tokenizer
from tokenizer import get_tokenizer
from utils import readFromFile, get_event, to_base10, to_beat_str, get_datetime_filename
from miditok import Event

# TODO: Add method comments


class TextDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, text):
        piece_events = self.text_to_events(text)
        inst_events = self.piece_to_inst_events(piece_events)
        events = self.add_timeshifts_for_empty_bars(inst_events)
        events = self.aggregate_timeshifts(events)
        events = self.add_velocity(events)
        return events

    def tokenize(self, events):
        tokens = []
        for inst in events.keys():
            tokens.append(self.tokenizer.events_to_tokens(events[inst]))
        return tokens

    def write_to_midi(self, text, filename=get_datetime_filename()):
        events = self.decode(text)
        tokens = self.tokenize(events)
        instruments = self.get_instruments_tuple(events)
        midi = tokenizer.tokens_to_midi(tokens, instruments)
        midi.dump(f"midi/decoded/generated_{filename}.mid")

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
        value1 = to_base10(beat_values1)
        value2 = to_base10(beat_values2)
        return to_beat_str(value1 + value2)

    def aggregate_timeshifts(self, events):
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
        instruments = []
        for inst in events.keys():
            is_drum = 1 if inst == "DRUM" else 0
            instruments.append((int(inst), is_drum))
        return tuple(instruments)


if __name__ == "__main__":

    filename = "20221202_183506_def4c02615dffd4be3579b5f7595459c288c392b31675f52577452521299e90c.json"
    encoded_json = readFromFile(
        f"models/model_2048_wholedataset/generated_sequences/{filename}",
        True,
    )
    encoded_text = encoded_json["sequence"]
    # encoded_text = "PIECE_START TRACK_START INST=25 DENSITY=2 BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=69 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=69 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=57 TIME_DELTA=1 NOTE_OFF=57 NOTE_ON=56 TIME_DELTA=1 NOTE_OFF=56 NOTE_ON=64 NOTE_ON=60 NOTE_ON=55 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=66 NOTE_ON=62 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=66 NOTE_OFF=62 NOTE_OFF=50 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=67 NOTE_ON=64 TIME_DELTA=1 NOTE_OFF=67 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=50 NOTE_ON=64 NOTE_ON=60 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=64 NOTE_OFF=60 NOTE_OFF=50 NOTE_ON=59 NOTE_ON=55 NOTE_ON=50 TIME_DELTA=1 NOTE_OFF=59 NOTE_OFF=50 NOTE_OFF=55 NOTE_OFF=50 BAR_END BAR_START BAR_END TRACK_END"

    tokenizer = get_tokenizer()
    TextDecoder(tokenizer).write_to_midi(encoded_text)

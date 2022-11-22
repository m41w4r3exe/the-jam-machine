# from encoder_mlike import tokenizer
from utils import readFromFile, TextToEvent

text_to_event = TextToEvent()


def decode_mlike(encoded_text: str):
    tracks = encoded_text.split("TRACK_START ")
    for trck in tracks[1:]:
        bars = trck.split("BAR_START ")
        instrument_id = bars[0].strip().split("INST=")[1]
        decoded_events = []
        empty_bar_count = 0

        for bar in bars[1:]:
            events = bar.strip().split(" ")
            for event in events:

                # Keep count of empty bars to add them later to Time-shift
                if event == "BAR_END":
                    if len(events) == 1:
                        empty_bar_count += 1
                    continue

                type, value = event.split("=")
                if empty_bar_count != 0 and type == "TIME_SHIFT":
                    value = add_emtpy_bars(value, empty_bar_count)
                midi_events = text_to_event.getlist(type, value)

                decoded_events.extend(midi_events)


def add_emtpy_bars(beat_values, empty_bar_count):
    beat_values = list(map(int, beat_values.split(".")))
    beat_values[0] += empty_bar_count * 4
    return ".".join(map(str, beat_values))


encoded_text = readFromFile("midi/the_strokes-reptilia_text_mlike.txt")
decode_mlike(encoded_text)

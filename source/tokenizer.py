from miditok import MIDILike

# TODO: Make this singleton
def get_tokenizer():
    pitch_range = range(21, 109)
    beat_res = {(0, 400): 8}
    return MIDILike(pitch_range, beat_res)

import matplotlib.pyplot as plt
import librosa.display
from pretty_midi import PrettyMIDI


# Note: these functions are meant to be played within an interactive Python shell
# Please refer to the synth.ipynb for an example of how to use them


def get_music(midi_file):
    """
    Load a midi file and return the PrettyMIDI object and the audio signal
    """
    music = PrettyMIDI(midi_file=str(midi_file))
    waveform = music.fluidsynth()
    return music, waveform


def show_piano_roll(music_notes, fs=100):
    """
    Show the piano roll of a music piece, with all instruments squashed onto a single 128xN matrix
    :param music_notes: PrettyMIDI object
    :param fs: sampling frequency
    """
    # get the piano roll
    piano_roll = music_notes.get_piano_roll(fs)
    print("Piano roll shape: {}".format(piano_roll.shape))

    # plot the piano roll
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(piano_roll, sr=100, x_axis="time", y_axis="cqt_note")
    plt.colorbar()
    plt.title("Piano roll")
    plt.tight_layout()
    plt.show()

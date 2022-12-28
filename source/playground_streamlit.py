import uuid
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_TRANSFER_CLASSES
from decoder import TextDecoder
from utils import get_miditok
from playback import get_music
from matplotlib import pylab
import sys
import matplotlib
from generation_utils import plot_piano_roll
import streamlit as st
from familizer import Familizer

# TODO: simplify state and piece_by_track to just one variable
# TODO: remove track_index and work with track names instead
# TODO: add a button to export the generated midi
# TODO: add a button to export the generated audio
# TODO: add a button to clear the state
# TODO: cleanup imports

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
sys.modules["pylab"] = pylab

# Load and initialize model
model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
n_bar_generated = 8
model, tokenizer = LoadModel(
    model_repo,
    from_huggingface=True,
).load_model_and_tokenizer()

# Initialize MIDI decoder
miditok = get_miditok()
decoder = TextDecoder(miditok)


def get_prompt(state, composer):
    """If something has already been generated, use this as a prompt,
    else use PIECE_START"""
    if len(state) == 0:
        input_prompt = "PIECE_START "
    else:
        input_prompt = composer.get_whole_piece_from_bar_dict()
    return input_prompt


def generator(
    label,
    temp,
    density,
    instrument_name,
    state,  # list of generated tracks
    piece_by_track,  # dictionary of generated tracks
):
    # initialize composer with model and generated tracks
    composer = GenerateMidiText(model, tokenizer, piece_by_track)
    new_track = {"label": label}
    if instrument_name == "Drums":
        instrument_family = "DRUMS"
    else:
        instrument_family = Familizer(arbitrary=True).get_family_number(
            instrument_name, program_type="transfer_name"
        )

    # Check if instrument already exists
    track_index = -1  # default to last generated
    regenerate = False
    if state != []:
        for i, current_track in enumerate(state):
            if current_track["label"] == new_track["label"]:
                track_index = i  # changing if exists
                regenerate = True

    # If instrument track already exists, delete it
    if regenerate:
        # Delete track
        state.pop(track_index)  # redundant
        composer.delete_one_track(track_index)
        track_index = -1  # reset to last generated

    # Generate new track
    input_prompt = get_prompt(state, composer)
    generated_text = composer.generate_one_new_track(
        instrument_family, density, temp, input_prompt=input_prompt
    )

    # convert generated text to midi and audio and save files locally
    decoder.get_midi(generated_text, "mixed.mid")
    mixed_inst_midi, mixed_audio = get_music("mixed.mid")

    # get last generated track and save it locally
    new_track_text = composer.get_selected_track_as_text(track_index)
    inst_midi_name = f"{instrument_name}.mid"
    decoder.get_midi(new_track_text, inst_midi_name)
    _, inst_audio = get_music(inst_midi_name)

    # get piano roll figure
    piano_roll = plot_piano_roll(mixed_inst_midi)

    # update state with new track
    new_track["text"] = new_track_text
    state.append(new_track)

    return (
        new_track_text,
        inst_audio,
        piano_roll,
        state,
        mixed_audio,
        composer.piece_by_track,
    )


def instrument_row(default_inst, row_id, col):
    # create dropdown for choice of instruments
    instrument = col.selectbox(
        "Instrument 🎹",
        sorted(
            [inst["transfer_to"] for inst in INSTRUMENT_TRANSFER_CLASSES] + ["Drums"]
        ),
        index=default_inst,
        key=row_id + 1,
    )

    # create slider for choice of temperature
    temp = col.slider(
        "Creativity (temperature) 🎨",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key=row_id + 2,
    )

    # create slider for choice of note density
    density = col.slider(
        "Note Density 🎼",
        min_value=0,
        max_value=3,
        value=3,
        step=1,
        key=row_id + 3,
    )

    # create button to generate track
    if col.button("Generate", key=row_id + 4):
        # generate track
        (
            output_txt,
            inst_audio,
            piano_roll,
            state,
            mixed_audio,
            piece_by_track,
        ) = generator(
            row_id,
            temp,
            density,
            instrument,
            st.session_state["state"],
            st.session_state["piece_by_track"],
        )

        # display generated track audio
        col.subheader(f"Track {int(str(row_id)[0]) + 1} Audio")
        col.audio(inst_audio, sample_rate=44100)

        # store state variables
        st.session_state.mixed_audio = mixed_audio
        st.session_state.piano_roll = piano_roll


def display_global_output():

    # display mixed audio and download button on the same row
    st.subheader("Mixed Audio")
    col1, col2 = st.columns([9, 1])
    col1.audio(st.session_state.mixed_audio, sample_rate=44100)

    # download generated track midi
    hash = str(uuid.uuid4())[:8]
    binary_midi = open("mixed.mid", "rb")  # .read()
    col2.download_button(
        label="Download MIDI",
        data=binary_midi,
        file_name=f"jam_gen_{hash}.mid",
        mime="audio/midi",
    )

    # display generated piano roll
    st.subheader("Piano Roll")
    st.pyplot(st.session_state.piano_roll)


def main():
    # initialize state variables
    if "mixed_audio" not in st.session_state:
        st.session_state.mixed_audio = None
    if "piano_roll" not in st.session_state:
        st.session_state.piano_roll = None
    if "state" not in st.session_state:
        st.session_state.state = []
    if "piece_by_track" not in st.session_state:
        st.session_state.piece_by_track = []


    # setup page layout
    st.set_page_config(layout="wide")

    # Generate each track config column
    default_instruments = [0, 6, 1]  # index of drums, synth bass, electric piano
    for instrument, i, col in zip(default_instruments, range(3), st.columns(3)):
        # Display track number
        col.subheader(f"Track {i+1}")
        # each track has its own row, with id in increments of 10
        instrument_row(default_inst=instrument, row_id=i * 10, col=col)

    # Display generated tracks
    if st.session_state.mixed_audio is not None:
        display_global_output()


if __name__ == "__main__":
    main()

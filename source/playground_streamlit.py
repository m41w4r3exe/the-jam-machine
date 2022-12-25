import matplotlib.pyplot as plt
import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_CLASSES
from decoder import TextDecoder
from utils import get_miditok, index_has_substring
from playback import get_music
from matplotlib import pylab
import sys
import matplotlib
from generation_utils import plot_piano_roll
import numpy as np
import streamlit as st

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


def define_prompt(state, genesis):
    """If somethikng has already been generated, use this as a prompt,
    else use PIECE_START"""
    if len(state) == 0:
        input_prompt = "PIECE_START "
    else:
        input_prompt = genesis.get_whole_piece_from_bar_dict()
    return input_prompt


def generator(
    label,
    regenerate,
    temp,
    density,
    instrument,
    state,
    piece_by_track,  # dictionary of generated tracks
    add_bars=False,
    add_bar_count=1,
):
    # initialize composer with model and generated tracks
    composer = GenerateMidiText(model, tokenizer, piece_by_track)
    track = {"label": label}
    inst = next(
        (inst for inst in INSTRUMENT_CLASSES if inst["transfer_to"] == instrument),
        {"family_number": "DRUMS"},
    )["family_number"]

    inst_index = -1  # default to last generated
    if state != []:
        for index, instrum in enumerate(state):
            if instrum["label"] == track["label"]:
                inst_index = index  # changing if exists

    # Generate
    if not add_bars:
        # Regenerate
        if regenerate:
            state.pop(inst_index)
            composer.delete_one_track(inst_index)

            generated_text = (
                composer.get_whole_piece_from_bar_dict()
            )  # maybe not useful here
            inst_index = -1  # reset to last generated

        # NEW TRACK
        input_prompt = define_prompt(state, composer)
        generated_text = composer.generate_one_new_track(
            inst, density, temp, input_prompt=input_prompt
        )

        regenerate = True  # set generate to true
    else:
        # NEW BARS
        composer.generate_n_more_bars(add_bar_count)  # for all instruments
        generated_text = composer.get_whole_piece_from_bar_dict()

    decoder.get_midi(generated_text, "mixed.mid")
    mixed_inst_midi, mixed_audio = get_music("mixed.mid")

    inst_text = composer.get_selected_track_as_text(inst_index)
    inst_midi_name = f"{instrument}.mid"
    decoder.get_midi(inst_text, inst_midi_name)
    _, inst_audio = get_music(inst_midi_name)
    piano_roll = plot_piano_roll(mixed_inst_midi)
    track["text"] = inst_text
    state.append(track)

    return (
        inst_text,
        (44100, inst_audio),
        piano_roll,
        state,
        (44100, mixed_audio),
        regenerate,
        composer.piece_by_track,
    )


def instrument_row(default_inst, row_id, col):
    # create dropdown for choice of instruments
    instrument = col.selectbox(
        "Instrument 🎹",
        sorted([inst["name"] for inst in INSTRUMENT_CLASSES] + ["Drums"]),
        index=default_inst,
        key=row_id + 1,
    )

    # create slider for choice of temperature
    temp = col.slider(
        "Creativity (temperature) 🎨",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
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
            regenerate,
            piece_by_track,
        ) = generator(
            row_id, regenerate, temp, density, instrument, state, piece_by_track
        )

        # display generated track audio
        col.subheader("Track Audio")
        col.audio(inst_audio)

        # display generated piano roll
        col.subheader("Piano Roll")
        col.pyplot(piano_roll)


def main():
    # initialize state variables
    piece_by_track = {}
    state = []
    regenerate = False

    # setup page layout
    st.set_page_config(layout="wide")

    # display mixed audio and piano roll
    # st.audio(mixed_audio)
    # st.pyplot(piano_roll)

    # Generate each track config column
    default_instruments = [3, 0, 6]  # index of drums, bass, guitar
    for instrument, i, col in zip(default_instruments, range(3), st.columns(3)):
        # Display track number
        col.subheader(f"Track {i+1}")
        # each track has its own row, with id in increments of 10
        instrument_row(default_inst=instrument, row_id=i * 10, col=col)


if __name__ == "__main__":
    main()

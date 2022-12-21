import matplotlib.pyplot as plt
import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_CLASSES, INSTRUMENT_TRANSFER_CLASSES
from decoder import TextDecoder
from utils import get_miditok, index_has_substring
from playback import get_music
from matplotlib import pylab
import sys
import matplotlib
from generation_utils import plot_piano_roll
import numpy as np

matplotlib.use("Agg")

sys.modules["pylab"] = pylab

model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
n_bar_generated = 8
# model_repo = "JammyMachina/improved_4bars-mdl"
# n_bar_generated = 4

model, tokenizer = LoadModel(
    model_repo,
    from_huggingface=True,
).load_model_and_tokenizer()

miditok = get_miditok()
decoder = TextDecoder(miditok)


def define_prompt(state, genesis):
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
    piece_by_track,
    add_bars=False,
    add_bar_count=1,
):

    genesis = GenerateMidiText(model, tokenizer, piece_by_track)
    track = {"label": label}
    inst = next(
        (
            inst
            for inst in INSTRUMENT_TRANSFER_CLASSES
            if inst["transfer_to"] == instrument
        ),
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
            genesis.delete_one_track(inst_index)

            generated_text = (
                genesis.get_whole_piece_from_bar_dict()
            )  # maybe not useful here
            inst_index = -1  # reset to last generated

        # NEW TRACK
        input_prompt = define_prompt(state, genesis)
        generated_text = genesis.generate_one_new_track(
            inst, density, temp, input_prompt=input_prompt
        )

        regenerate = True  # set generate to true
    else:
        # NEW BARS
        genesis.generate_n_more_bars(add_bar_count)  # for all instruments
        generated_text = genesis.get_whole_piece_from_bar_dict()

    decoder.get_midi(generated_text, "mixed.mid")
    mixed_inst_midi, mixed_audio = get_music("mixed.mid")

    inst_text = genesis.get_selected_track_as_text(inst_index)
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
        genesis.piece_by_track,
    )


def instrument_row(default_inst, row_id):
    with gr.Row():
        row = gr.Variable(row_id)
        with gr.Column(scale=1, min_width=100):
            inst = gr.Dropdown(
                sorted([inst["transfer_to"] for inst in INSTRUMENT_TRANSFER_CLASSES])
                + ["Drums"],
                value=default_inst,
                label="Instrument",
            )
            temp = gr.Dropdown(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                value=0.7,
                label="Creativity",
            )
            density = gr.Dropdown([1, 2, 3], value=3, label="Note Density")

        with gr.Column(scale=3):
            output_txt = gr.Textbox(
                label="output", lines=10, max_lines=10, show_label=False
            )
        with gr.Column(scale=1, min_width=100):
            inst_audio = gr.Audio(label="TRACK Audio", show_label=True)
            regenerate = gr.Checkbox(value=False, label="Regenerate", visible=False)
            # add_bars = gr.Checkbox(value=False, label="Add Bars")
            # add_bar_count = gr.Dropdown([1, 2, 4, 8], value=1, label="Add Bars")
            gen_btn = gr.Button("Generate")
            gen_btn.click(
                fn=generator,
                inputs=[row, regenerate, temp, density, inst, state, piece_by_track],
                outputs=[
                    output_txt,
                    inst_audio,
                    piano_roll,
                    state,
                    mixed_audio,
                    regenerate,
                    piece_by_track,
                ],
            )


with gr.Blocks() as demo:
    piece_by_track = gr.State([])
    state = gr.State([])
    title = gr.Markdown(
        """ # Demo-App of The-Jam-Machine
    A Generative AI trained on text transcription of MIDI music """
    )
    track1_md = gr.Markdown(""" ## Mixed Audio and Piano Roll """)
    mixed_audio = gr.Audio(label="Mixed Audio")
    piano_roll = gr.Plot(label="Piano Roll", show_label=False)
    description = gr.Markdown(
        """
        For each **TRACK**, choose your **instrument** along with **creativity** (temperature) and **note density**. Then, hit the **Generate** Button!
        You can have a look at the generated text; but most importantly, check the **piano roll** and listen to the TRACK audio!
        If you don't like the track, hit the generate button to regenerate it! Generate more tracks and listen to the **mixed audio**!
        """
    )
    track1_md = gr.Markdown(""" ## TRACK 1 """)
    instrument_row("Drums", 0)
    track1_md = gr.Markdown(""" ## TRACK 2 """)
    instrument_row("Synth Bass 1", 1)
    track1_md = gr.Markdown(""" ## TRACK 3 """)
    instrument_row("Synth Lead Square", 2)
    # instrument_row("Piano")

demo.launch(debug=True)

"""
TODO: add a button to save the generated midi
TODO: add improvise button
TODO: set values for temperature as it is done for density
"""

import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_CLASSES
from encoder import MIDIEncoder
from decoder import TextDecoder
from utils import get_miditok, index_has_substring
from playback import get_music
from matplotlib import pylab
import sys
import matplotlib
from generation_utils import plot_piano_roll
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules["pylab"] = pylab

model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
revision = "ddf00f90d6d27e4cc0cb99c04a22a8f0a16c933e"
n_bar_generated = 8
# model_repo = "JammyMachina/improved_4bars-mdl"
# n_bar_generated = 4

model, tokenizer = LoadModel(
    model_repo, from_huggingface=True, revision=revision
).load_model_and_tokenizer()
genesis = GenerateMidiText(
    model,
    tokenizer,
)
genesis.set_nb_bars_generated(n_bars=n_bar_generated)

miditok = get_miditok()
decoder = TextDecoder(miditok)


def define_prompt(state, genesis):
    if len(state) == 0:
        input_prompt = "PIECE_START "
    else:
        input_prompt = genesis.get_whole_piece_from_bar_dict()
    return input_prompt


def generator(regenerate, add_bars, temp, density, instrument, add_bar_count, state):

    inst = next(
        (inst for inst in INSTRUMENT_CLASSES if inst["name"] == instrument),
        {"family_number": "DRUMS"},
    )["family_number"]

    inst_index = index_has_substring(state, "INST=" + str(inst))

    # Regenerate
    if regenerate:
        state.pop(inst_index)
        genesis.delete_one_track(inst_index)
        generated_text = (
            genesis.get_whole_piece_from_bar_dict()
        )  # maybe not useful here
        inst_index = -1  # reset to last generated

    # Generate
    if not add_bars:
        # NEW TRACK
        input_prompt = define_prompt(state, genesis)
        generated_text = genesis.generate_one_new_track(
            inst, density, temp, input_prompt=input_prompt
        )
    else:
        # NEW BARS
        genesis.generate_n_more_bars(add_bar_count)  # for all instruments
        generated_text = genesis.get_whole_piece_from_bar_dict()

    decoder.get_midi(generated_text, "tmp/mixed.mid")
    mixed_inst_midi, mixed_audio = get_music("tmp/mixed.mid")

    inst_text = genesis.get_selected_track_as_text(inst_index)
    inst_midi_name = f"tmp/{instrument}.mid"
    decoder.get_midi(inst_text, inst_midi_name)
    inst_midi, inst_audio = get_music(inst_midi_name)
    piano_roll = plot_piano_roll(mixed_inst_midi)
    state.append(inst_text)

    return inst_text, (44100, inst_audio), piano_roll, state, (44100, mixed_audio)


def instrument_row(default_inst):

    with gr.Row():
        with gr.Column(scale=1, min_width=50):
            inst = gr.Dropdown(
                [inst["name"] for inst in INSTRUMENT_CLASSES] + ["Drums"],
                value=default_inst,
                label="Instrument",
            )
            temp = gr.Number(value=0.7, label="Temperature")
            density = gr.Dropdown([0, 1, 2, 3], value=3, label="Density")

        with gr.Column(scale=3):
            output_txt = gr.Textbox(label="output", lines=10, max_lines=10)
        with gr.Column(scale=1, min_width=100):
            inst_audio = gr.Audio(label="Audio")
        with gr.Column(scale=1, min_width=50):
            regenerate = gr.Checkbox(value=False, label="Regenerate")
            add_bars = gr.Checkbox(value=False, label="Add Bars")
            add_bar_count = gr.Dropdown([1, 2, 4, 8], value=1, label="Add Bars")
            gen_btn = gr.Button("Generate")
            gen_btn.click(
                fn=generator,
                inputs=[
                    regenerate,
                    add_bars,
                    temp,
                    density,
                    inst,
                    add_bar_count,
                    state,
                ],
                outputs=[output_txt, inst_audio, piano_roll, state, mixed_audio],
            )


with gr.Blocks() as demo:
    state = gr.State([])
    mixed_audio = gr.Audio(label="Mixed Audio")
    piano_roll = gr.Plot(label="Piano Roll")
    instrument_row("Drums")
    instrument_row("Bass")
    instrument_row("Synth Lead")
    # instrument_row("Piano")

demo.launch(debug=True)

""" 
TODO: add improvise button
TODO: regenerate and add bars button should not be activatblae together
TODO: make a global add bar button/tick box
TODO: row fuckikng height to fix
TODO: add a button to save the generated midi
TODO: improve the piano roll - maybe using librosa to check if it works/looks good already by default
TODO: adding a reset button to reload the model
TODO: update all piano_rolls, audio and text of OTHER INSTRUMENTS when adding bars with one instrument. Or changing the way to add bars
TODO: mapping instrument names to specific instrument and not
TODO: reset state of tick boxes after used maybe (regenerate, add bars) ; 
TODO: block regenerate if add bar on
"""

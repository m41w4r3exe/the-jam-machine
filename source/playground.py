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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules["pylab"] = pylab

model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
model, tokenizer = LoadModel(
    model_repo, from_huggingface=True
).load_model_and_tokenizer()
genesis = GenerateMidiText(
    model,
    tokenizer,
)
miditok = get_miditok()
decoder = TextDecoder(miditok)


def plot_piano_roll(p_midi_note_list):
    piano_roll_fig = plt.figure()
    piano_roll_fig.tight_layout()
    piano_roll_fig.patch.set_alpha(0)
    note_time = []
    note_pitch = []
    for note in p_midi_note_list:
        note_time.append([note.start, note.end])
        note_pitch.append([note.pitch, note.pitch])

    plt.plot(
        np.array(note_time).T,
        np.array(note_pitch).T,
        color="orange",
        linewidth=1,
    )
    plt.xlabel("ticks")
    plt.axis("off")
    return piano_roll_fig


def generator(regenerate, temp, density, instrument, state):
    genesis.set_temperatures([temp])

    inst = next(
        (inst for inst in INSTRUMENT_CLASSES if inst["name"] == instrument),
        {"family_number": "DRUMS"},
    )["family_number"]

    if regenerate:
        i = index_has_substring(state, "INST=" + str(inst))
        state.pop(i)
        genesis.delete_one_track()

    state_text = "".join(state)
    if state_text == "":
        state_text = "PIECE_START"

    generated_text = genesis.generate_one_track(
        input_prompt=state_text, instrument=inst, density=density
    )
    decoder.get_midi(generated_text, "tmp/mixed.mid")
    _, mixed_audio = get_music("tmp/mixed.mid")

    inst_text = generated_text
    for each in state:
        inst_text = inst_text.replace(each, "")
    inst_midi_name = f"tmp/{instrument}.mid"
    decoder.get_midi(inst_text, inst_midi_name)
    inst_midi, inst_audio = get_music(inst_midi_name)
    piano_roll = plot_piano_roll(inst_midi.instruments[0].notes)

    state.append(inst_text)

    return inst_text, (44100, inst_audio), piano_roll, state, (44100, mixed_audio)


def instrument_row(default_inst):
    with gr.Row():
        with gr.Column(scale=1, min_width=10):
            inst = gr.Dropdown(
                [inst["name"] for inst in INSTRUMENT_CLASSES] + ["Drums"],
                value=default_inst,
                label="Instrument",
            )
            temp = gr.Number(value=0.75, label="Temperature")
            density = gr.Dropdown([0, 1, 2, 3], value=2, label="Density")
        with gr.Column(scale=3, min_width=100):
            with gr.Tab("Piano Roll"):
                piano_roll = gr.Plot(label="Piano Roll")
            with gr.Tab("Music text tokens"):
                output_txt = gr.Textbox(label="output", lines=6, max_lines=6)
        with gr.Column(scale=1, min_width=100):
            inst_audio = gr.Audio(label="Audio")
            regenerate = gr.Checkbox(value=False, label="Regenerate")
            gen_btn = gr.Button("Generate")
            gen_btn.click(
                fn=generator,
                inputs=[regenerate, temp, density, inst, state],
                outputs=[output_txt, inst_audio, piano_roll, state, mixed_audio],
            )


with gr.Blocks() as demo:
    state = gr.State([])
    mixed_audio = gr.Audio(label="Mixed Audio")
    instrument_row("Drums")
    instrument_row("Bass")
    instrument_row("Guitar")
    instrument_row("Piano")

demo.launch(debug=True)

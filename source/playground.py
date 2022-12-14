import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_CLASSES
from decoder import TextDecoder
from utils import get_miditok
from playback import get_music

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


def generator(temp, density, instrument, state):
    genesis.set_temperature(temp)

    inst = next(
        (inst for inst in INSTRUMENT_CLASSES if inst["name"] == instrument),
        {"family_number": "DRUMS"},
    )["family_number"]

    generated_text = genesis.generate_one_track(
        input_prompt=state, instrument=inst, density=density
    )
    midi = decoder.get_midi(
        generated_text, "temporary.mid"
    )  # Returns miditoolkit, not compatible with pretty_midi
    p_midi, audio = get_music("temporary.mid")
    state = generated_text
    return generated_text, (44100, audio), state


with gr.Blocks() as demo:
    state = gr.State("PIECE_START")
    with gr.Row():
        with gr.Column(scale=1, min_width=10):
            inst = gr.Dropdown(
                ["Drums", "Bass", "Piano", "Synth Lead", "Synth Pad"],
                value="Drums",
                label="Instrument",
            )
            with gr.Row():
                temp = gr.Number(value=0.75, label="Temperature")
                density = gr.Dropdown([0, 1, 2, 3], value=2, label="Density")
        with gr.Column(scale=1, min_width=100):
            output_txt = gr.Textbox(label="output", lines=6, max_lines=6)
        with gr.Column(scale=1, min_width=100):
            audio = gr.Audio(label="Audio")
            gen_btn = gr.Button("Generate")
            gen_btn.click(
                fn=generator,
                inputs=[temp, density, inst, state],
                outputs=[output_txt, audio, state],
            )
    with gr.Row():
        temp = gr.Number(value=0.75, label="Temperature")

demo.launch()

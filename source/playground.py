import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_CLASSES
from decoder import TextDecoder
from utils import get_miditok
from playback import get_music
import librosa.display


model_repo = "misnaej/the-jam-machine"
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

    generated_text = genesis.generate_one_sequence(
        input_prompt=state, instrument=inst, density=density
    )
    midi = decoder.get_midi(
        generated_text, "temporary.mid"
    )  # Returns miditoolkit, not compatible with pretty_midi
    p_midi, audio = get_music("temporary.mid")
    piano_roll = p_midi.get_piano_roll(fs=4410)  # Not working
    img = librosa.display.specshow(piano_roll, sr=100, x_axis="time", y_axis="cqt_note")
    state = generated_text

    return generated_text, (44100, audio), img, state


with gr.Blocks() as demo:
    state = gr.State("PIECE_START")
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            temp = gr.Number(value=0.75, label="Temperature")
            inst = gr.Dropdown(
                ["Drums", "Bass", "Piano", "Synth Lead", "Synth Pad"],
                value="Drums",
                label="Instrument",
            )
            density = gr.Dropdown([0, 1, 2, 3], value=2, label="Density")
            output_txt = gr.Textbox(label="output")
            piano_roll = gr.Image(label="Piano Roll", image_mode="L")
            audio = gr.Audio(label="Audio")
            gen_btn = gr.Button("Generate")

            gen_btn.click(
                fn=generator,
                inputs=[temp, density, inst, state],
                outputs=[output_txt, audio, piano_roll, state],
            )
    with gr.Row():
        temp = gr.Number(value=0.75, label="Temperature")

demo.launch()

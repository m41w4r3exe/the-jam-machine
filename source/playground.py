import gradio as gr
from load import LoadModel
from generate import GenerateMidiText
from constants import INSTRUMENT_TRANSFER_CLASSES
from decoder import TextDecoder
from utils import get_miditok
from playback import get_music
from matplotlib import pylab
import sys
import os
import matplotlib
from generation_utils import plot_piano_roll

matplotlib.use("Agg")

sys.modules["pylab"] = pylab

model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
n_bar_generated = 8

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
    track = {
        "label": label,
        "instrument": instrument,
        "temperature": temp,
        "density": density,
    }
    inst = next(
        (
            inst
            for inst in INSTRUMENT_TRANSFER_CLASSES
            if inst["transfer_to"] == instrument
        ),
        {"family_number": "DRUMS"},
    )["family_number"]

    inst_index = -1  # default to last generated
    if piece_by_track != []:
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

    # save the mix midi and get the mix audio
    decoder.get_midi(generated_text, "mixed.mid")
    mixed_inst_midi, mixed_audio = get_music("mixed.mid")
    # get the instrument text MIDI
    inst_text = genesis.get_whole_track_from_bar_dict(inst_index)
    # save the instrument midi and get the instrument audio
    decoder.get_midi(inst_text, f"{instrument}.mid")
    _, inst_audio = get_music(f"{instrument}.mid")
    # generate the piano roll
    piano_roll = plot_piano_roll(mixed_inst_midi)
    track["text"] = inst_text
    state.append(track)
    output_file = "./mixed.mid"
    return (
        inst_text,
        (44100, inst_audio),
        piano_roll,
        state,
        (44100, mixed_audio),
        regenerate,
        genesis.piece_by_track,
        output_file,
    )


def generated_text_from_state(state):
    generated_text_from_state = "PIECE_START "
    for track in state:
        generated_text_from_state += track["text"]
    return generated_text_from_state


def instrument_col(default_inst, col_id):
    inst_label = gr.Variable(col_id)
    with gr.Column(scale=1, min_width=100):
        track_md = gr.Markdown(f"""## TRACK {col_id+1}""")
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
        regenerate = gr.State(
            value=False
        )  # initial state should be to generate (not regenerate)
        gen_btn = gr.Button("Generate")
        inst_audio = gr.Audio(label="TRACK Audio", show_label=True)
        output_txt = gr.Textbox(
            label="output", lines=10, max_lines=10, show_label=False, visible=False
        )

    gen_btn.click(
        fn=generator,
        inputs=[inst_label, regenerate, temp, density, inst, state, piece_by_track],
        outputs=[
            output_txt,
            inst_audio,
            piano_roll,
            state,
            mixed_audio,
            regenerate,
            piece_by_track,
            output_file,
        ],
    )


with gr.Blocks() as demo:
    piece_by_track = gr.State([])
    state = gr.State([])
    title = gr.Markdown(
        """ # Demo-App of The-Jam-Machine
    ## A Generative AI trained on text transcription of MIDI music """
    )

    description = gr.Markdown(
        """
        For each **TRACK**, choose your **instrument** along with **creativity** (temperature) and **note density**. 
        Then, hit the **Generate** Button, and after a few seconds a track should have been generated. 
        Check the **piano roll** and listen to the TRACK! If you don't like it, hit the generate button to regenerate it. 
        You can then generate more tracks and listen to the **mixed audio**! \n
        Does it sound nice? Maybe a little robotic and laking some depth... Well, you can download the MIDI file and import it in your favorite DAW to edit the instruments and add some effects!\
        Note: Do not try to generate several tracks simultaneously as it will crash the app; wait for one track to be generated before generating another one.    
        """
    )

    aud_md = gr.Markdown(f""" ## Mixed Audio, Piano Roll and MIDI Download """)
    with gr.Row(variant="default"):
        mixed_audio = gr.Audio(label="Mixed Audio", show_label=False)
        output_file = gr.File(
            label="Download",
            show_label=False,
        )
    with gr.Row(variant="compact"):
        piano_roll = gr.Plot(label="Piano Roll", show_label=False)

    with gr.Row(variant="default"):
        instrument_col("Drums", 0)
        instrument_col("Synth Bass 1", 1)
        instrument_col("Synth Lead Square", 2)

demo.launch(debug=True, server_name="0.0.0.0", share=False)
"""
TODO: add improvise button
TODO: cleanup input output of generator
TODO: add a way to add bars
"""

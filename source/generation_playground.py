from generate import *
from utils import WriteTextMidiToFile, get_miditok
from load import LoadModel
from decoder import TextDecoder
from playback import get_music

# worker
DEVICE = "cpu"

# define generation parameters
N_FILES_TO_GENERATE = 2
Temperatures_to_try = [0.7]

USE_FAMILIZED_MODEL = True
force_sequence_length = True

if USE_FAMILIZED_MODEL:
    # model_repo = "misnaej/the-jam-machine-elec-famil"
    # model_repo = "misnaej/the-jam-machine-elec-famil-ft32"

    model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
    n_bar_generated = 8

    # model_repo = "JammyMachina/improved_4bars-mdl"
    # n_bar_generated = 4
    instrument_promt_list = ["4", "DRUMS", "3"]
    # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
    density_list = [3, 2, 2]
    # temperature_list = [0.7, 0.7, 0.75]
else:
    model_repo = "misnaej/the-jam-machine"
    instrument_promt_list = ["30"]  # , "DRUMS", "0"]
    density_list = [3]  # , 2, 3]
    # temperature_list = [0.7, 0.5, 0.75]
    pass

# define generation directory
generated_sequence_files_path = define_generation_dir(model_repo)

# load model and tokenizer
model, tokenizer = LoadModel(
    model_repo, from_huggingface=True
).load_model_and_tokenizer()

# does the prompt make sense
check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

for temperature in Temperatures_to_try:
    print(f"================= TEMPERATURE {temperature} =======================")
    for _ in range(N_FILES_TO_GENERATE):
        print(f"========================================")
        # 1 - instantiate
        piece_by_track = []  # reset the piece by track
        generate_midi = GenerateMidiText(model, tokenizer, piece_by_track)
        # 0 - set the n_bar for this model
        generate_midi.set_nb_bars_generated(n_bars=n_bar_generated)
        # 1 - defines the instruments, densities and temperatures
        # 2 - generate the first 8 bars for each instrument
        # generate_midi.set_improvisation_level(0)
        generate_midi.generate_piece(
            instrument_promt_list,
            density_list,
            [temperature for _ in density_list],
        )
        # 3 - force the model to improvise
        generate_midi.set_improvisation_level(8)
        # 4 - generate the next 4 bars for each instrument
        generate_midi.generate_n_more_bars(4)
        generate_midi.set_improvisation_level(20)
        generate_midi.generate_n_more_bars(4)
        generate_midi.set_improvisation_level(4)
        generate_midi.generate_n_more_bars(2)
        generate_midi.set_improvisation_level(40)
        generate_midi.generate_n_more_bars(8)

        generate_midi.generated_piece = generate_midi.get_whole_piece_from_bar_dict()

        # print the generated sequence in terminal
        print("=========================================")
        print(generate_midi.generated_piece)
        print("=========================================")

        # write to JSON file
        filename = WriteTextMidiToFile(
            generate_midi,
            generated_sequence_files_path,
        ).text_midi_to_file()

        # decode the sequence to MIDI """
        decode_tokenizer = get_miditok()
        TextDecoder(decode_tokenizer, USE_FAMILIZED_MODEL).get_midi(
            generate_midi.generated_piece, filename=filename.split(".")[0] + ".mid"
        )
        inst_midi, mixed_audio = get_music(filename.split(".")[0] + ".mid")
        max_time = get_max_time(inst_midi)
        piano_roll_fig = plot_piano_roll(inst_midi)
        piano_roll_fig.savefig(
            filename.split(".")[0] + "_piano_roll.png", bbox_inches="tight"
        )
        piano_roll_fig.clear()

        print("Et voilà! Your MIDI file is ready! GO JAM!")

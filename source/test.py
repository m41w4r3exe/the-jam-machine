from generate import *
from generation_utils import WriteTextMidiToFile
from utils import get_miditok
from load import LoadModel
from decoder import TextDecoder
from encoder import *
from playback import get_music

USE_FAMILIZED_MODEL = True
force_sequence_length = True

DEVICE = "cpu"
model_repo = "JammyMachina/elec-gmusic-familized-model-13-12__17-35-53"
n_bar_generated = 8

# define test directory
test_dir = define_generation_dir("source/tests")


def test_generate():
    """Generate a MIDI_text sequence"""
    # define generation parameters
    temperature = 0.7

    instrument_promt_list = ["DRUMS", "4", "3"]
    # DRUMS = drums, 0 = piano, 1 = chromatic percussion, 2 = organ, 3 = guitar, 4 = bass, 5 = strings, 6 = ensemble, 7 = brass, 8 = reed, 9 = pipe, 10 = synth lead, 11 = synth pad, 12 = synth effects, 13 = ethnic, 14 = percussive, 15 = sound effects
    density_list = [3, 2, 2]

    # load model and tokenizer
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    # does the prompt make sense
    check_if_prompt_inst_in_tokenizer_vocab(tokenizer, instrument_promt_list)

    print(f"================= TEMPERATURE {temperature} =======================")

    # 1 - instantiate
    piece_by_track = []  # reset the piece by track
    generate_midi = GenerateMidiText(model, tokenizer, piece_by_track)
    # 0 - set the n_bar for this model
    generate_midi.set_nb_bars_generated(n_bars=n_bar_generated)
    # 1 - defines the instruments, densities and temperatures
    # 2 - generate the first 8 bars for each instrument
    generate_midi.generate_piece(
        instrument_promt_list,
        density_list,
        [temperature for _ in density_list],
    )
    generate_midi.generated_piece = generate_midi.get_whole_piece_from_bar_dict()

    # print the generated sequence in terminal
    print("=========================================")
    print(generate_midi.generated_piece)
    print("=========================================")
    return generate_midi


def test_decode(generate_midi):
    """Write the generated MIDI_text sequence to a file and decode it to MIDI"""
    filename = WriteTextMidiToFile(
        generate_midi,
        test_dir,
    ).text_midi_to_file()

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

    return filename.split(".")[0]


def test_encode(midi_filename):
    piece_text = from_MIDI_to_sectionned_text(f"{midi_filename}")
    writeToFile(f"{midi_filename}_from_midi.txt", piece_text)
    return piece_text


def check_encoder_decoder_consistency():
    midi_text_generated = test_generate()
    midi_file = test_decode(midi_text_generated)
    midi_text_from_file = test_encode(midi_file)
    midi_text_from_file


if __name__ == "__main__":
    """ " Test Run : 1 generate, 2 decode, 3 encode, compare 1 generated and 3 encoded"""
    check_encoder_decoder_consistency()
    try:
        check_encoder_decoder_consistency()
    except:
        print("Error in encoder-decoder consistency test")

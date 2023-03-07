import logging
import gradio
import subprocess
from generate import *
from generation_utils import WriteTextMidiToFile
from utils import get_miditok
from load import LoadModel
from decoder import TextDecoder
from encoder import *
from playback import get_music

# TODO Use logging to create a log file

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
    density_list = [3, 2, 3]

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

    filename = WriteTextMidiToFile(
        generate_midi,
        test_dir,
    ).text_midi_to_file()

    return generate_midi, filename


def check_for_duplicated_subsequent_tokens(generated_text):
    """Check if there are duplicated tokens in the generated sequence"""
    for i, text in enumerate(generated_text.split(" ")):
        if (
            i < len(generated_text.split(" ")) - 1
            and text == generated_text.split(" ")[i + 1]
        ):
            print(f"DUPLICATED TOKENS {text} in position {i}")
            print(f"{generated_text.split(' ')[i - min([i,3]): i + 3]}")


def test_decode(filename=None):
    if filename is None:
        filename = "source/tests/test_decode.json"

    filename = filename.split(".")[0]  # remove extension

    generated_piece = readFromFile(f"{filename}.json", isJSON=True)["generated_midi"]

    decode_tokenizer = get_miditok()
    TextDecoder(decode_tokenizer, USE_FAMILIZED_MODEL).get_midi(
        generated_piece, filename=f"{filename}.mid"
    )
    inst_midi, _ = get_music(f"{filename}.mid")
    piano_roll_fig = plot_piano_roll(inst_midi)
    piano_roll_fig.savefig(f"{filename}_piano_roll.png", bbox_inches="tight")
    piano_roll_fig.clear()

    return filename


def test_encode(midi_filename=None):
    if midi_filename is None:
        midi_filename = "midi/the_strokes-reptilia"

    piece_text = from_MIDI_to_sectionned_text(
        f"{midi_filename}", familized=USE_FAMILIZED_MODEL
    )
    print("=========================================")
    print(piece_text)
    writeToFile(f"{midi_filename}_from_midi.txt", piece_text)
    return piece_text


def simplify_events_for_comparison(generated_event, encoded_event):
    """Simplifies the 'NOTE' events, byt getting rid of the level of the note (e.g. 'NOTE=60' becomes 'NOTE')
    Why: because when the sequence is encoded as midi with pretty midi, the order of sequences of NOTE_OFF or NOTE_ON can be changed
    This does not changes the music, but then the text sequences will be different and won't match"""
    if "NOTE" in generated_event:
        generated_word = generated_event.split("=")[0]
    else:
        generated_word = generated_event

    if "NOTE" in encoded_event:
        encoded_word = encoded_event.split("=")[0]
    else:
        encoded_word = encoded_event

    return generated_word, encoded_word


def check_sequence_word_by_word(generated_text, encoded_text):
    """Check if the generated MIDI_text sequence and the encoded MIDI_text sequence are the same word by word"""
    generated_text = generated_text.split(" ")
    encoded_text = encoded_text.split(" ")
    absolutely_similar = True
    for i in range(len(generated_text)):
        generated_word, encoded_word = simplify_events_for_comparison(
            generated_text[i], encoded_text[i]
        )

        if generated_word != encoded_word:
            absolutely_similar = False
            print(
                f"Word {i} is different - Generated: {generated_text[i]} vs Encoded: {encoded_text[i]}"
            )
            print(f"generated: {generated_text[i - min([i, 4]) : i + 3]}")
            print(f"encoded: {encoded_text[i - min([i, 4]) : i + 3]}")
            print("------------------")
            # raise ValueError("Generated and encoded MIDI_text sequences are different")

    return absolutely_similar


def test_compare_generated_encoded(generated_text, encoded_text):
    """Compare the generated MIDI_text sequence and the encoded MIDI_text sequence"""
    absolutely_similar = False
    if generated_text == encoded_text:
        absolutely_similar = True
    else:
        similar_length = len(generated_text.split(" ")) == len(encoded_text.split(" "))

    if not absolutely_similar and similar_length:
        print(
            f"Lengths of generated and encoded sequences are the same: {len(encoded_text.split(' '))} words"
        )
        absolutely_similar = check_sequence_word_by_word(generated_text, encoded_text)

    if not absolutely_similar and not similar_length:
        print(
            f"Lengths of generated and encoded sequences are different: {len(generated_text.split(' '))} vs {len(encoded_text.split(' '))} words"
        )
        print("generated:")
        print(f"beginning: {generated_text[:120]}")
        print(f"end: {generated_text[-120:]}")
        print("encoded:")
        print(f"beginning: {encoded_text[:120]}")
        print(f"end: {encoded_text[-120:]}")
        print("------------------")
        # raise ValueError("Generated and encoded MIDI_text sequences are different")

    if absolutely_similar:
        print("Generated and encoded MIDI_text sequences are the same")


def check_encoder_decoder_consistency():
    midi_text_generated, filename = test_generate()
    check_for_duplicated_subsequent_tokens(midi_text_generated.generated_piece)
    midi_file = test_decode(filename)
    midi_text_from_file = test_encode(midi_file)

    test_compare_generated_encoded(
        midi_text_generated.generated_piece, midi_text_from_file
    )


def test_gradio():
    current_wd = os.getcwd()
    os.chdir("./source")
    # start subprocess
    p = subprocess.run(["gradio playground.py"], shell=True)
    # os.chdir(current_wd)


def run_test(func, *args, **kwargs):
    for kwarg in kwargs:
        match kwarg:
            case "tested_process":
                testing_process = kwargs[kwarg]
    # func(*args)
    try:
        print("==================================")
        print(f"Testing: {testing_process}")
        func(*args)
        print("----------------------------------")
        print(f"{testing_process} successful")
    except:
        print("----------------------------------")
        print(f"{testing_process} failed")

    print("----------------------------------")


if __name__ == "__main__":

    # test encoding
    run_test(test_encode, "midi/the_strokes-reptilia", tested_process="Encoding")

    # test generation
    run_test(test_generate, tested_process="Generation")

    # test decoding
    run_test(test_decode, tested_process="Decoding")

    # " Test Run : 1 generate, 2 decode, 3 encode, compare 1 generated and 3 encoded
    run_test(
        check_encoder_decoder_consistency, tested_process="Encoder-Decoder Consistency"
    )

    # test launch gradio app
    run_test(test_gradio, tested_process="Launching Gradio App")
    # Here the radio app will be started and needs to be tested manually, then closed
    # The message will report that the test failed because the process was killed, but this can be ignored

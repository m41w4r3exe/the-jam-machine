# The-Jam-Machine - a Generative AI composing MIDI music

*This project is still under development*.

## Summary

The-Jam-Machine is a source of inspiration for beginner or more proficient musicians. Based on a GPT (Generative Pretrained-Transformer) architecture, and trained on the text transcriptions of about 5000 MIDI songs, it can generate harmonious MIDI sequences.

You can check the App on [HuggingFace](https://huggingface.co/spaces/JammyMachina/the-jam-machine-app), with which you can generate 8 bars of downloadable MIDI music with up to 3 instruments playing in harmony.
_______________

## [Presentation](https://pitch.com/public/417162a8-88b0-4472-a651-c66bb89428be)

_______________

## Contributors

Jean Simonnet: [Github](https://github.com/misnaej) / [Linkedin](https://www.linkedin.com/in/jeansimonnet/) \
Louis Demetz:  [Github](https://github.com/louis-demetz) / [Linkedin](https://www.linkedin.com/in/ldemetz/) \
Halid Bayram:  [Github](https://github.com/m41w4r3exe) / [Linkedin](https://www.linkedin.com/in/halid-bayram-6b9ba861/)

_______________

## Setting up the Jam-Machine on your computer

This works for MacOS 12.6
### 1. Install fluidsynth
The Jam-Machine requires Fluidsynth, a software synthetizer.\
Make sure to install it on your system, and fopr this please check the github repo [here](https://github.com/FluidSynth/fluidsynth/wiki/Download).\
E.g.: with Mac OS X and Homebrew, run `brew install fluidsynth`

### 2. Clone the repository
`git clone git@github.com:m41w4r3exe/the-jam-machine.git`
### 3. Install the dependencies
[pipenv](https://pypi.org/project/pipenv/) was chosen to manage the dependencies.
Make sure that pipenv in install in your main python environment.\
If not, run `pip install pipenv` in your terminal.\
Then, from `the-jam-machine` root folder, run `pipenv install --ignore-pipfile` to install the dependencies.
To activate the virtual environment, run `pipenv shell`.



### 4. Test the Jam-Machine
Run the `test.py` script from `the-jam-machine/source` to check that everything is working fine.\
It test the encoding, generation, decoding and the consistency between the encoding and decoding, as well as the gradio app.\ 
It returns a lot of messages in the terminal including:\
- `Encoding successful`
- `Generation successful`
- `Decoding successful`
- `Encoder-Decoder Consistency successful`

Testing the gradio app requires to open the URL displayed in the terminal and manually test the app.\
Then when closed by keyboard interuption (CTL+C), it returns `Launching Gradio App failed` but it can be ignored.

## Making Music with the Jam-Machine
### 1. With the gradio app.

From `the-jam-machine/source`, run `gradio playground.py` and then open the URL displayed in your terminal. You will be able to generate 8 bars of MIDI music with up to 3 instruments playing in harmony. More instructions are displayed in the app. 
Try it on [HuggingFace](https://huggingface.co/spaces/JammyMachina/the-jam-machine-app) first, maybe this is enough for you, and you don't need to install and run anything locally.

### 2. With `source/generation_playground.py`
The gradio app has been made pretty simple so it works without breaking. The problem is that is quite limited in what it can do.
If you want to get more experimental with generation you can check out `source/generation_playground.py` and try to get inspired by the code. You will be able to generate longer tracks, but it is less interactive than the app.

# Have Fun!

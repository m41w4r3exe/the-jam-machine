[Github](https://github.com/m41w4r3exe/the-jam-machine)

# Description
We aim to provide the creative spark to help artists overcome the writer's block at the beginning of a project

The jam machine dreams up techno music compositions through deep neural hallucinations. Just import the raw inspiration into your favorite DAW using the generated MIDI file to finish producing your original tracks.

The jam machine is the copilot for artists.

# Plan

## Objectives
1. Create new music genres by cross-breeding techo samples with other genre tracks
2. Create pure techno tracks

## Steps
- Encoder / decoder for the midi data, use data exactly like tristan
- statistics on dataset
	- e.g. number of tracks, instruments for each feature
	- calculate note density
	- finding similarity between genres / midi
	- statistics time signature (also per genre)
- have a first pipeline ready
	- gpt-2 train on low number of songs and try to generate one instrument, one bar of one genre (e.g. famous rock songs)
- data augmentation / feature engineering
	- include note density, velocity, 

## Features
- 2 models
	- continuation
	- track-level inpainting
- polyphony
- attribute control
	- instruments
	- mood (valence)
	- section (chorus, bridge)
	- genre

## encoding
- midi-like first
- try rest vs time-shift

## overall
- Create a model
	- select compute platform
	- use existing lakh and other available large datasets
	- figure out encoding/tokenizer situation
	- plug into tristan notebook & train model
	- figure out generation with multiple parameters
- Gather a techno specific dataset --> to figure out later

## next steps
- halid
	- figure out encoding with miditok
- louis
	- playback midi files in lakh & other datasets to get a feel fore the quality
	- read/find papers about this business
	- read/run tristan notebookw

## statistics on data
- statistics on number of instruments, bars, tracks, voices per file
- how many tokens needed for data?
	- mmm used 128 note_on and note_off, 48 time_shift tokens

# Resources
## datasets
- [Lakh MIDI](https://colinraffel.com/projects/lmd/) full midi tracks from anonymous artits
- [Meta MIDI](https://github.com/jeffreyjohnens/MetaMIDIDataset) very large full midi tracks datasets
	- [paper on how dataset was created](https://archives.ismir.net/ismir2021/paper/000022.pdf)
- [Bitmidi](https://bitmidi.com/) full midi tracks with metadata from spotify, msg etc
- [Million Song dataset](http://millionsongdataset.com/) Metadata for tracks
- [nonstop2k](https://www.nonstop2k.com/) has lots of house midis

## midi tokenizer
- [MidiTok](https://github.com/Natooz/MidiTok)

## other resources
- [ismir](https://ismir.net/) conference for everything music technology
	- Music Information Retrieval
	- organise music demixing challenge
- [Magenta](https://github.com/magenta/magenta-js/tree/master/music) lots of models for music by google
- [Demucs Music Source Separation](https://github.com/facebookresearch/demucs) Facebook, won ismir 2021
- Tristan Behrens
	- [Composer](https://huggingface.co/spaces/ai-guru/composer/tree/main)
	- Jupyter notebook on music generation
- [Amazon DeepComposer](https://aws.amazon.com/deepcomposer/) Tool to learn ML with a music generation project

## papers
- [DiffRoll](https://arxiv.org/abs/2210.05148) 
	- paper from october 2022 shared by Tristan
	- transcription with unsupervised learning
	- looks promising despite not doing onset offset framing
- [Musika!](https://arxiv.org/abs/2208.08706) techno / experimental music generator
	- [model on huggingface](https://huggingface.co/spaces/marcop/musika)
	- [code on github]()
	- Uses
		- [Adversarial autoencoders](https://arxiv.org/abs/1511.05644)
		- [GANs](https://www.semanticscholar.org/paper/Generative-Adversarial-Nets-Goodfellow-Pouget-Abadie/54e325aee6b2d476bbbb88615ac15e251c6e8214)
- [MusicVAE](https://arxiv.org/abs/1803.05428) A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music
	- [summary page with all resources](https://magenta.tensorflow.org/music-vae)
- [MMM : Multi-Track Music Machine](https://metacreation.net/mmm-multi-track-music-machine/) used by tristan for generative work
	- [[MMM - Multi-Track Music Machine]]
	- [Transformer-XL](https://arxiv.org/abs/1901.02860) The architecture recommended by MMM
	- [Calliope](https://metacreation.net/calliope/) music generation interface based on mmm paper
	- [PreGLAM-MMM: Using MMM in video games](https://metacreation.net/preglam-mmm-using-mmm-in-video-games/) Applied in a video game for live music generation based on a model evaluating valence, arousal and tension
- [MidiTok introduction paper at ismir](https://archives.ismir.net/ismir2021/latebreaking/000005.pdf)
- [Piano inpainting application](https://ghadjeres.github.io/piano-inpainting-application/)
- [Multi-instrument Music Synthesis with Spectrogram Diffusion](https://github.com/magenta/music-spectrogram-diffusion) shared by Tristan
- [ismir 2021 papers](https://ismir2021.ismir.net/papers/)

## music analysis
- Spotify APIs
	- [Track Audio Analysis](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-analysis) Low-level audio analysis
		- [Tool based on this API](https://spotify-audio-analysis.glitch.me/analysis.html) Looks at sections part ! 👀 
	- [Track Audio Features](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features) danceability, valence etc

## Music generation companies
- https://splashhq.com
- https://www.ampermusic.com/
- magenta
	- https://experiments.withgoogle.com/nsynth-super
- https://www.evabeat.com/ looks kinda pro
- https://www.loudly.com/aimusicstudio advanced ?
- https://soundraw.io/ has a cheap plan, website looks broken
- https://www.bandlab.com/songstarter ai powered daw
- https://www.orbplugins.com/orb-producer-suite/ ai plugin
- https://soundful.com/ cheap, working product
 for musicians

# Mentorship
## 11.04 Meeting
### Questions
- talk about vision: aim is to make midi gen for techno music
- challenge: putting dataset together
- steps:
	- tokenizer issue
	- how to do parameters
	- musenet prompt midi generation?
		- velocity changes are less but more important
		- how do you make song coherent
	- fine-tuning / training
		- should train on whole music corpus then fine tune?
		- what about subgenres

### Answers
- start with 500 songs to train on colab
	- good results within reach of free tier for first baseline
- 16 gpus, 1.5 weeks to train on whole metamidi
	- gpt is already well trained 
	- training big model then fine tuning is out of scope for this project
	- just use the full notebook with simply replacing input data to get first baseline
- midi packs with no song structure are problematic: model infers symphony only with full structure
- trouble with transformers is sequence length: compromise between bars and number of instruments
	- normal length is 4 bars for training
	- try adding tokens before generation (to tell genre apart)
	- most of his work based on mmm paper - [[the-jam-machine#papers|See papers section]]
		- limited number of bars is the start of paper
		- tristan did different: bars are fed non sequentially with a bar number
		- try 4 instruments, then see if possible to increase size
			- maybe techno has generally easier song structure, allowing longer seq length
		- velocity doubles the midi size, halving seq length
		- bach composer is 512 seq length
		- standard tristan is 2048 seq length
- to get started: get midi files, do a bit of statistics and midi analysis
- midi tok has 2 alternative encodings, so try out different encodings
	- has seq to midi functionality: makes life easier
	- start with 1 midi file to get pipeline running
	- 1.5k songs is even better


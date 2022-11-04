# Objective
We aim to provide the creative spark to help artists overcome the writer's block at the beginning of a project

The jam machine dreams up techno music compositions through deep neural hallucinations. Just import the raw inspiration into your favorite DAW using the generated MIDI file to finish producing your original tracks.

# Plan
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
- midi packs with no song structure are problematic: model infers symphomy only with full structure
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


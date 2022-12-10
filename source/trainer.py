# RUN 3 lines below in a separate cell in Google Colab
# !pip install transformers tokenizers wandb huggingface_hub datasets datetime nvidia-ml-py3
# from huggingface_hub import notebook_login
# notebook_login()

"""TO DO:
- separating this file
     -> we can just clone the repo and run it on the machine wherever it is at
- dataset on Huggingface
- pushing checkpoints and model to HuggingFace
- Get rid of google drive/local seperation stuff
- forget about google collab for ever
"""

import os
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import wandb
from datasets import load_dataset

# from pynvml import *
from trainer_utils import *

# CONFIG:
# The Checkpoint path folder will here be in a distinct folder
# Because a new name is given for every model using formattedtime
TRAIN_FROM_CHECKPOINT = None  # Example: fullpath/model/checkpoint-80000
ADDITIONAL_TRAIN_EPOCHS = 0  # only used if TRAIN_FROM_CHECKPOINT is not None
EVAL_STEPS = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 3
TRAIN_EPOCHS = 5
# MODEL_RUN_IN = "gdrive"
MODEL_RUN_IN = "local"
# wandb.init(project=f"the-jammy-machine")
HF_REPO = "misnaej/the-jam-machine"

# set paths
base_path = "/Users/jean/WORK/DSR_2022_b32/music_portfolio/the_jam_machine_github/the-jam-machine"
model_path = "models/model_elec_familiarised"
data_path = "midi/dataset/electronic/familiarised"
paths = set_paths(base_path, model_path, data_path, model_run_in=MODEL_RUN_IN)
tokenizer_path = paths["tokenizer_path"]
dataset_path = paths["dataset_path"]
model_path = paths["model_path"]

# Load dataset from gzip files
train_data = load_dataset(dataset_path, data_files={"train": "train/*.zip"})["train"]
validate_data = load_dataset(dataset_path, data_files={"val": "validate/*.zip"})["val"]

# Tokenizer
tokenizer = train_tokenizer(tokenizer_path, train_data)

# Tokenize Dateset
print("=======Tokenizing Train dataset========")
train_data_tokenized = TokenizeDataset(tokenizer).batch_tokenization(train_data)
check_tokenized_data(train_data, train_data_tokenized)

print("=======Tokenizing Validation dataset========")
validate_data_tokenized = TokenizeDataset(tokenizer).batch_tokenization(validate_data)
check_tokenized_data(validate_data, validate_data_tokenized)

# Model, Data collator and Trainer
model = GPT2LMHeadModel(
    GPT2Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        n_embd=512,
        n_head=8,
        n_layer=6,
        n_positions=2048,
    )
)
training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCHS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    learning_rate=5e-4,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    fp16=False,
    save_strategy="steps",
    save_steps=EVAL_STEPS * 4,
    save_total_limit=5,
    logging_steps=EVAL_STEPS,
    logging_dir=os.path.join(model_path, "logs"),
    report_to="wandb",
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data_tokenized,
    eval_dataset=validate_data_tokenized,
)
if MODEL_RUN_IN != "gdrive":  # it does not work from gdrive
    trainer.args.push_to_hub = True
    trainer.args.hub_strategy = "end"
    trainer.args.hub_model_id = HF_REPO

# Train the model
if TRAIN_FROM_CHECKPOINT is not None:
    trainer.args.num_train_epochs += ADDITIONAL_TRAIN_EPOCHS
    result = trainer.train(TRAIN_FROM_CHECKPOINT)
else:
    result = trainer.train()

print("Training finished")
print(result)

# Save the tokenizer, latest status of trained model
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

# wandb finish
wandb.finish()

# Save the trainer state
trainer.state.save_to_json(f"{model_path}/trainer_state.json")

# Ploting the history of the training
history = get_history(trainer)
plot_history(history)

if MODEL_RUN_IN != "gdrive":  # not sure it works from gdrive # TOCHECK
    trainer.push_to_hub()

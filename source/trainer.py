# RUN 3 lines below in a seperate cell in Google Colab
# !pip install transformers tokenizers wandb huggingface_hub datasets datetime
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
from pathlib import Path
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from datetime import datetime
import wandb
from datasets import load_dataset
from trainer_utils import get_history, plot_history


# CONFIG:
""" 
The Checkpoint path folder will here be in a distinct folder
Because a new name is given for every model using formattedtime
"""
TRAIN_FROM_CHECKPOINT = None  # Example: fullpath/model/checkpoint-80000
ADDITIONAL_TRAIN_EPOCHS = 0  # only used if TRAIN_FROM_CHECKPOINT is not None
EVAL_STEPS = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
TRAIN_EPOCHS = 5

"""Set paths either from Google Drive or locally"""
formattedtime = datetime.now().strftime("%d-%m__%H-%M-%S")
model_name = f"model_{formattedtime}"
try:
    from google.colab import drive  # type: ignore

    wandb.init(project=f"the-jammy-machine")
    drive.mount("/content/gdrive")
    drive_path = "/content/gdrive/MyDrive/the_jam_machine"
    dataset_path = f"{drive_path}/midi_encoded"
    model_path = f"{drive_path}/model_{formattedtime}"
    MODEL_RUN_IN = "gdrive"
except:
    dataset_path = "./midi_encoded"
    model_path = f"./models/model_{formattedtime}"
    MODEL_RUN_IN = "local"
    HF_REPO = "misnaej/the-jam-machine"
tokenizer_path = f"{model_path}/tokenizer.json"

if not os.path.exists(model_path):
    os.mkdir(model_path)

"""Load dataset from gzip files"""
train_data = load_dataset(dataset_path, data_files={"train": "train/*.zip"})["train"]
validate_data = load_dataset(dataset_path, data_files={"val": "validate/*.zip"})["val"]

# TODO: Move tokenizer logic to encoder and use its json here only.
"""Get tokenizer from scratch or saved tokenizer.json"""
if not os.path.isfile(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(train_data["text"], trainer=tokenizer_trainer)
    tokenizer.save(tokenizer_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print("Vocabulary size: ", tokenizer.vocab_size)


def tokenize(data):
    return tokenizer(
        data["text"],
        truncation=True,
        padding=True,
        max_length=2048,
    )


train_data_tokenized = train_data.map(tokenize, batched=True, remove_columns=["text"])
validate_data_tokenized = validate_data.map(
    tokenize, batched=True, remove_columns=["text"]
)

"""Make sure the tokenized dataset structure is correct and check a few examples"""
assert "input_ids" in list(train_data_tokenized[0]), list(train_data_tokenized[0])
for i, data in enumerate(train_data["text"][:3]):
    print("----")
    print(data)
    print(train_data_tokenized[i]["input_ids"])


"""Create model and trainer"""
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
    fp16=True,
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
"""Train the model from scratch or from checkpoint"""
if TRAIN_FROM_CHECKPOINT is not None:
    trainer.args.num_train_epochs += ADDITIONAL_TRAIN_EPOCHS
    result = trainer.train(TRAIN_FROM_CHECKPOINT)
else:
    result = trainer.train()

print("Training finished")
print(result)


"""Save the tokenizer, latest status of trained model and push it to hugging face."""
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

"""wandb finish"""
wandb.finish()

"""Save the trainer state which contains the metrics to json"""
trainer.state.save_to_json(f"{model_path}/trainer_state.json")

"""Ploting the history of the training"""
history = get_history(trainer)
plot_history(history)

if MODEL_RUN_IN != "gdrive":  # not sure it works from gdrive # TOCHECK
    trainer.push_to_hub()

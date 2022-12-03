# RUN 3 lines below in a seperate cell in Google Colab
# !pip install transformers tokenizers wandb huggingface_hub
# from huggingface_hub import notebook_login
# notebook_login()

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

# CONFIG:
TRAIN_FROM_CHECKPOINT = None  # Example: checkpoint-80000
EVAL_STEPS = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 1
TRAIN_EPOCHS = 5
WANDB_KEY = "156af33a7166789bdccefbe9d465fe87b82f2e5e"

formattedtime = datetime.now().strftime("%d-%m__%H-%M-%S")
wandb.login(key=WANDB_KEY)
wandb.init(project=f"the-jam-machine-{formattedtime}")

try:
    from google.colab import drive

    drive.mount("/content/gdrive")
    drive_path = Path("/content/gdrive/MyDrive/the_jam_machine")
    dataset_path = Path(f"{drive_path}/midi_encoded")
    model_path = Path(f"{drive_path}/model_{formattedtime}")
except:
    dataset_path = "./midi_encoded"
    model_path = f"./models/model_{formattedtime}"

if not os.path.exists(model_path):
    os.mkdir(model_path)

tokenizer_path = f"{model_path}/tokenizer.json"
dataset_path_train = os.path.join(dataset_path, "train")
dataset_path_valid = os.path.join(dataset_path, "validate")


def get_dataset(dataset_path):
    all_files_paths = [f"{dataset_path}/{file}" for file in os.listdir(dataset_path)]
    dataset = []
    for file_path in all_files_paths:
        with open(file_path) as file:
            for line in file:
                dataset.append(line.rstrip("\n"))
    return dataset


dataset_train = get_dataset(dataset_path_train)
dataset_validate = get_dataset(dataset_path_valid)


if TRAIN_FROM_CHECKPOINT is None:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(dataset_train, trainer=tokenizer_trainer)
    tokenizer.save(tokenizer_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print("Vocabulary size: ", tokenizer.vocab_size)


def tokenize_function(data_to_tokenize):
    tokenized_data = tokenizer(
        data_to_tokenize,
        truncation=True,
        padding=True,
        max_length=2048,
    )
    return {"input_ids": tokenized_data["input_ids"]}


dataset_train_tokenized = list(map(tokenize_function, dataset_train))
dataset_val_tokenized = list(map(tokenize_function, dataset_validate))


assert list(dataset_train_tokenized[0]) == ["input_ids"], list(
    dataset_train_tokenized[0]
)
# Check a few samples
for i, data in enumerate(dataset_train[:3]):
    print("----")
    print(data)
    print(dataset_train_tokenized[i])

model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    n_embd=512,
    n_head=8,
    n_layer=10,
    n_positions=2048,
)
model = GPT2LMHeadModel(model_config)

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
    save_strategy="steps",
    save_steps=EVAL_STEPS * 10,
    save_total_limit=10,
    logging_steps=EVAL_STEPS * 10,
    logging_dir=os.path.join(model_path, "logs"),
    report_to="wandb",
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train_tokenized,
    eval_dataset=dataset_val_tokenized,
)

if TRAIN_FROM_CHECKPOINT is not None:
    trainer.train(f"{model_path}/{TRAIN_FROM_CHECKPOINT}")
else:
    trainer.train()


# Save the tokenizer.
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
trainer.push_to_hub()

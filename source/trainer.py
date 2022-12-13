# MUST: Run 'huggingface-cli login' and 'wanddb login'
# sudo apt install git-lfs
# !pip install transformers tokenizers wandb huggingface_hub datasets datetime nvidia-ml-py3

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
from trainer_utils import *
from datetime import datetime
from huggingface_hub import create_repo

formattedtime = datetime.now().strftime("%d-%m__%H-%M-%S")

# CONFIG:
DATASET_NAME = "elec-gmusic-familized"
HF_DATASET_REPO = f"JammyMachina/{DATASET_NAME}"
HF_MODEL_REPO = f"{HF_DATASET_REPO}-model-{formattedtime}"
TRAIN_FROM_CHECKPOINT = None  # Must be full path: {HF_MODEL_REPO}/checkpoint-80000
EVAL_STEPS = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_EPOCHS = 5
MODEL_PATH = f"models/{DATASET_NAME}"

if not os.path.exists(MODEL_PATH):
    print(f"Creating model path: {MODEL_PATH}")
    os.makedirs(MODEL_PATH, exist_ok=True)

wandb.init(project="the-jammy-machine")
create_repo(HF_MODEL_REPO, exist_ok=True)

data = load_dataset(
    HF_DATASET_REPO, data_files={"train": "train/*.zip", "eval": "validate/*.zip"}
)
tokenizer = train_tokenizer(MODEL_PATH, data["train"])
print("=======Tokenizing dataset========")
data_tokenized = TokenizeDataset(tokenizer).batch_tokenization(data)
check_tokenized_data(data["train"], data_tokenized["train"], plot_path=MODEL_PATH)
check_tokenized_data(data["eval"], data_tokenized["eval"])

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
    output_dir=MODEL_PATH,
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
    logging_dir=os.path.join(MODEL_PATH, "logs"),
    report_to="wandb",
    seed=42,
    push_to_hub=True,
    hub_model_id=HF_MODEL_REPO,
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data_tokenized["train"],
    eval_dataset=data_tokenized["eval"],
)


result = trainer.train(TRAIN_FROM_CHECKPOINT)
print("Training finished")
print(result)

# Save the tokenizer, latest status of trained model
tokenizer.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)
wandb.finish()
trainer.state.save_to_json(f"{MODEL_PATH}/trainer_state.json")

# Ploting the history of the training
history = get_history(trainer)
plot_history(history)

trainer.push_to_hub()

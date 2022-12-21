# Commands to run on a new machine:
# $ git clone https://github.com/m41w4r3exe/the-jam-machine
# $ sudo apt install git-lfs
# $ pip install transformers tokenizers wandb huggingface_hub datasets

import os
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb
from datasets import load_dataset
from trainer_utils import *
from huggingface_hub import create_repo, HfApi
from passwords import HF_READ_TOKEN, HF_WRITE_TOKEN

# CONFIG:
DATASET_NAME = "improved_4bars"
HF_DATASET_REPO = f"JammyMachina/{DATASET_NAME}"
HF_MODEL_REPO = f"{HF_DATASET_REPO}-mdl"
MODEL_PATH = f"models/{DATASET_NAME}"
TRAIN_FROM_CHECKPOINT = True
EVAL_STEPS = 1024
TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 7
GRADIENT_ACCUMULATION_STEPS = 1


if not os.path.exists(MODEL_PATH):
    print(f"Creating model path: {MODEL_PATH}")
    os.makedirs(MODEL_PATH, exist_ok=True)

os.environ["WANDB_API_KEY"] = "156af33a7166789bdccefbe9d465fe87b82f2e5e"
wandb.init(project="the-jammy-machine")
create_repo(HF_MODEL_REPO, token=HF_WRITE_TOKEN, exist_ok=True)

data = load_dataset(
    HF_DATASET_REPO,
    data_files={"train": "train/*.zip", "eval": "validate/*.zip"},
    use_auth_token=HF_READ_TOKEN,
)

if TRAIN_FROM_CHECKPOINT:
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_REPO, use_auth_token=HF_READ_TOKEN
    )
else:
    tokenizer = train_tokenizer(MODEL_PATH, data["train"])

print("=======Tokenizing dataset========")
data_tokenized = TokenizeDataset(tokenizer).batch_tokenization(data)
# check_tokenized_data(data["train"], data_tokenized["train"], plot_path=MODEL_PATH)
# check_tokenized_data(data["eval"], data_tokenized["eval"])

if TRAIN_FROM_CHECKPOINT:
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_REPO,
        use_auth_token=HF_READ_TOKEN,
        revision="d1262472162d86c420ed8bf6a8e54270a6186993",
    )
else:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            n_embd=512,
            n_head=8,
            n_layer=8,
            n_positions=2048,
        )
    )

training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCHS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    learning_rate=5e-5,
    # weight_decay=0.1,
    # warmup_steps=5000,
    # lr_scheduler_type="cosine",
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
    hub_token=HF_WRITE_TOKEN,
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data_tokenized["train"],
    eval_dataset=data_tokenized["eval"],
)

with open(f"{MODEL_PATH}/training_args.json", "w") as f:
    f.write(training_args.to_json_string())

result = trainer.train()
print("Training finished")
print(result)


tokenizer.save_pretrained(MODEL_PATH, push_to_hub=True, use_auth_token=HF_WRITE_TOKEN)
trainer.state.save_to_json(f"{MODEL_PATH}/trainer_state.json")
model.save_pretrained(MODEL_PATH, push_to_hub=True, use_auth_token=HF_WRITE_TOKEN)
trainer.push_to_hub()
wandb.finish()

HfApi().upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_MODEL_REPO,
    ignore_patterns=[".git/*", "**/.git/*"],
    token=HF_WRITE_TOKEN,
)

# Ploting the history of the training
history = get_history(trainer)
plot_history(history, MODEL_PATH, HF_MODEL_REPO)

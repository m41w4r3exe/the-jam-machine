from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
import os


class LoadModelFromLocalFolder:
    def __init__(self, path, device="cpu"):
        if not os.path.exists(path):
            raise Exception("Model path does not exist")

        self.path = path
        self.device = device

    def load_model_and_tokenizer(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(self.path).to(self.device)
        return model

    def load_tokenizer(self):
        if not os.path.exists(f"{self.path}/tokenizer.json"):
            raise Exception(
                f"There is no 'tokenizer.json'file in the defined {self.path}"
            )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{self.path}/tokenizer.json"
        )
        return tokenizer


class LoadFromHuggingFace:
    def __init__(self, path, device="cpu"):
        self.path = path
        self.device = device

    def load_model_and_tokenizer(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(self.path).to(self.device)
        return model

    def load_tokenizer(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.path)
        return tokenizer

from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
import os


class LoadModel:
    def __init__(self, path, from_huggingface=True, device="cpu"):
        # path is either a relative path on a local/remote machine or a model repo on HuggingFace
        if not from_huggingface:
            if not os.path.exists(path):
                raise Exception("Model path does not exist")
        self.from_huggingface = from_huggingface
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
        if self.from_huggingface:
            pass
        else:
            if not os.path.exists(f"{self.path}/tokenizer.json"):
                raise Exception(
                    f"There is no 'tokenizer.json'file in the defined {self.path}"
                )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.path)
        return tokenizer

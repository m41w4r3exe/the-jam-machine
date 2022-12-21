from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
import os
import torch


class LoadModel:
    """
    Example usage:

    # if loading model and tokenizer from Huggingface
    model_repo = "misnaej/the-jam-machine"
    model, tokenizer = LoadModel(
        model_repo, from_huggingface=True
    ).load_model_and_tokenizer()

    # if loading model and tokenizer from a local folder
    model_path = "models/model_2048_wholedataset"
    model, tokenizer = LoadModel(
        model_path, from_huggingface=False
    ).load_model_and_tokenizer()

    """

    def __init__(self, path, from_huggingface=True, device="cpu", revision=None):
        # path is either a relative path on a local/remote machine or a model repo on HuggingFace
        if not from_huggingface:
            if not os.path.exists(path):
                print(path)
                raise Exception("Model path does not exist")
        self.from_huggingface = from_huggingface
        self.path = path
        self.device = device
        self.revision = revision
        if torch.cuda.is_available():
            self.device = "cuda"

    def load_model_and_tokenizer(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()

        return model, tokenizer

    def load_model(self):
        if self.revision is None:
            model = GPT2LMHeadModel.from_pretrained(self.path)  # .to(self.device)
        else:
            model = GPT2LMHeadModel.from_pretrained(
                self.path, revision=self.revision
            )  # .to(self.device)

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

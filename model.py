from data import HF_TOKEN, TOPICS

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn

class TagPredictor(nn.Module):
    def __init__(self, tokenizer, base_model, classes: list[str],
                 embedding_size: int, hidden_size: int = None, ident_on_eval: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.classes = np.array(classes)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, hidden_size or embedding_size),
            nn.ReLU(),
            nn.Linear(hidden_size or embedding_size, len(classes)),
            nn.Sigmoid()
        )
        self.ident_on_eval = ident_on_eval
    
    def forward(self, return_loss=True, **input):
        if self.ident_on_eval and not self.training:
            size = len(input['index'])
            result = torch.zeros((size, len(self.classes)), requires_grad=False, device=input['index'].device)
            result[:, 0] += input['index'] / 1024
            return result
        embeddings = self.base_model.forward(**input).last_hidden_state[:, 0]
        return self.classifier.forward(embeddings)
    
    def get_sample_proba(self, title: str, summary: str):
        with torch.no_grad():
            tokens = self.tokenizer(title + " [SEP] " + summary, return_tensors='pt')
            result = self.forward(**tokens)
            return result[0]
    
    def get_sample_pred(self, title: str, summary: str, threshold: float):
        return self.classes[self.get_sample_proba(title, summary) >= threshold]

def create_model():
    from huggingface_hub import login, hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    login(HF_TOKEN)
    model_tensors_path = hf_hub_download(repo_id="Ajno/ArxivArticleTagPredictor", filename="model.safetensors")

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased", token=HF_TOKEN)
    model = AutoModel.from_pretrained("distilbert/distilbert-base-cased", token=HF_TOKEN)

    embedding_dim = model.embeddings.word_embeddings.embedding_dim
    
    classifier_model = TagPredictor(tokenizer, model, TOPICS["List"], embedding_dim)
    classifier_model.load_state_dict(load_file(model_tensors_path))

    return classifier_model

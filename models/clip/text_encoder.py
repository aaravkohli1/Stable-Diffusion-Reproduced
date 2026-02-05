from __future__ import annotations
from typing import List, Union
import torch
import torch.nn as nn
from .tokenizer import CLIPTokenizerWrapper

class CLIPTextEncoder(nn.Module):
    def __init__(self, hf_text_model: nn.Module, tokenizer: CLIPTokenizerWrapper, max_length: int = 77):
        super().__init__()
        self.hf = hf_text_model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained_hf(cls, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        from transformers import CLIPTextModel
        tokenizer = CLIPTokenizerWrapper.from_pretrained(model_name, max_length=max_length)
        model = CLIPTextModel.from_pretrained(model_name)
        return cls(model, tokenizer, max_length=max_length)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts, device=device)
        out = self.hf(**tokens)
        return out.last_hidden_state  # [B, T, D]

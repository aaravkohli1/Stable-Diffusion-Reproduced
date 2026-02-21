from __future__ import annotations

from typing import List, Union, Optional, Dict
import torch
import torch.nn as nn

from .tokenizer import CLIPTokenizerWrapper
from .clip_text_model import MyCLIPTextModel


class CLIPTextEncoder(nn.Module):
    """
    - from_pretrained_hf(): downloads HF weights, builds our architecture, loads those weights into it.
    - encode(): runs our model forward and returns last_hidden_state [B, T, D].
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CLIPTokenizerWrapper,
        max_length: int = 77,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained_hf(
        cls,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        *,
        strict: bool = False,
        verbose: bool = False,
    ) -> "CLIPTextEncoder":
        
        from transformers import CLIPTextModel

        tokenizer = CLIPTokenizerWrapper.from_pretrained(model_name, max_length=max_length)

        # Load HF model 
        hf = CLIPTextModel.from_pretrained(model_name)
        hf.eval()
        cfg = hf.config

        # Build our model using HF config
        my = MyCLIPTextModel(
            vocab_size=cfg.vocab_size,                          
            hidden_size=cfg.hidden_size,                      
            intermediate_size=cfg.intermediate_size,          
            num_hidden_layers=cfg.num_hidden_layers,           
            num_attention_heads=cfg.num_attention_heads,        
            max_position_embeddings=cfg.max_position_embeddings,
            layer_norm_eps=cfg.layer_norm_eps,   
        )
        my.eval()

        # Load HF weights into our model
        incompatible = my.load_state_dict(hf.state_dict(), strict=strict)
        if verbose:
            # In newer PyTorch, load_state_dict returns IncompatibleKeys(missing_keys, unexpected_keys)
            try:
                missing = incompatible.missing_keys
                unexpected = incompatible.unexpected_keys
            except Exception:
                missing, unexpected = incompatible
            print("=== load_state_dict report ===")
            print("missing keys:", missing)
            print("unexpected keys:", unexpected)

        return cls(model=my, tokenizer=tokenizer, max_length=max_length)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Returns:
            last_hidden_state: [B, T, D]
        """
        device = next(self.parameters()).device

        tokens: Dict[str, torch.Tensor] = self.tokenizer(texts, device=device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask", None)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
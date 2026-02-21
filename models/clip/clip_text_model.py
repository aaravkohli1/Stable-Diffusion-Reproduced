from __future__ import annotations
import torch
import torch.nn as nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))

class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask: [B, T] with 1 for real tokens, 0 for padding (HF style)
        """
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores: [B, heads, T, T]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # causal mask (no looking ahead): mask upper triangle
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float("-inf"))

        # padding mask: don’t attend to PAD tokens in keys
        if attention_mask is not None:
            # attention_mask: 1 = keep, 0 = pad
            key_is_pad = attention_mask == 0  # [B, T]
            attn = attn.masked_fill(key_is_pad[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # [B, heads, T, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, layer_norm_eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        return self.token_embedding(input_ids) + self.position_embedding(pos_ids)

class MyCLIPTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, intermediate_size, num_attention_heads, layer_norm_eps)
            for _ in range(num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embeddings(input_ids)  # [B, T, D]
        for layer in self.encoder.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.final_layer_norm(x)
        return x  # last_hidden_state [B, T, D]

class MyCLIPTextModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.text_model = MyCLIPTextTransformer(**kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        last_hidden_state = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        class Output: pass
        out = Output()
        out.last_hidden_state = last_hidden_state
        return out
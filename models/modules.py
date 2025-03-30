import torch
import math
from torch import nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    """Standard FeedForward block: Linear -> ReLU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        d_model: The number of features in the transformer model's internal representations (also the size of
        embeddings). This controls how much a model can remember and process.
        h: The number of attention heads in the multi-head self-attention mechanism.
        dropout: The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divided by heads"

        self.head_dim = d_model // h  # also called d_k in paper
        assert (
                h * self.head_dim == d_model
        ), "Embed size have to be equal to this multiplication"

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(
            self.head_dim * self.h, d_model, bias=False
        )  # self.head_dim * self.h should be == d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]

        # (batch, h, seq_len, head_dim) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)  # @: matrix multiplication
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        # Input shapes: (batch_eff, seq_len_eff, d_model)
        batch_eff, seq_len_eff, _ = q.shape

        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # Reshape for multi-head: (batch_eff, seq_len_eff, h, head_dim) -> (batch_eff, h, seq_len_eff, head_dim)
        q_multi = q_proj.view(batch_eff, seq_len_eff, self.h, self.head_dim).transpose(1, 2)
        k_multi = k_proj.view(batch_eff, seq_len_eff, self.h, self.head_dim).transpose(1, 2)
        v_multi = v_proj.view(batch_eff, seq_len_eff, self.h, self.head_dim).transpose(1, 2)

        # Calculate attention
        context, self.attention_scores = MultiHeadAttentionBlock.attention(
            q_multi, k_multi, v_multi, mask, self.dropout
        )

        # Combine heads: (batch_eff, h, seq_len_eff, head_dim) -> (batch_eff, seq_len_eff, h, head_dim) -> (
        # batch_eff, seq_len_eff, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_eff, seq_len_eff, self.d_model)

        return self.w_o(context)


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 1000):  # Reduced default max_len slightly
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Don't add batch dimension here, add it dynamically if needed, or apply broadcasting
        self.register_buffer("pe", pe)  # Shape [max_len, d_model]

    def forward(self, x):
        """ Expects x shape [batch_eff, seq_len, d_model] """
        # Add positional encoding using broadcasting
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
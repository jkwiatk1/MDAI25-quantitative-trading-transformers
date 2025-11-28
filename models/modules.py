import torch
import math
from torch import nn


class LayerNormalization(nn.Module):
    """Custom layer normalization with learnable scale and shift parameters."""
    
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    """Feed-forward network: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention mechanism with scaled dot-product attention."""
    
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Args:
            d_model: Model dimension (embedding size)
            h: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by number of heads"

        self.head_dim = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """Scaled dot-product attention."""
        head_dim = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: Query, key, value tensors of shape [batch, seq_len, d_model]
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = q.shape

        # Project and reshape for multi-head attention
        q_proj = self.w_q(q).view(batch_size, seq_len, self.h, self.head_dim).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, seq_len, self.h, self.head_dim).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, seq_len, self.h, self.head_dim).transpose(1, 2)

        # Apply attention
        context, self.attention_scores = MultiHeadAttentionBlock.attention(
            q_proj, k_proj, v_proj, mask, self.dropout
        )

        # Combine heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """Residual connection with layer normalization and dropout."""
    
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Apply sublayer with residual connection: x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))
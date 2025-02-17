import math
import torch
from torch import nn

from models.modules import (
    LayerNormalization,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


class PositionalEncoding(nn.Module):
    """Adds time information to transformer inputs"""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


# class PatchEmbedding(nn.Module):
#     """Przekształca dane wejściowe na sekwencję patchy"""
#     def __init__(self, input_dim, d_model, patch_size):
#         super().__init__()
#         self.proj = nn.Conv1d(input_dim, d_model, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         batch, time, stocks, features = x.shape
#         x = x.permute(0, 3, 1, 2).contiguous().view(batch * stocks, features, time)
#         x = self.proj(x)  # (Batch * Stocks, d_model, num_patches)
#         x = x.permute(0, 2, 1)  # (Batch * Stocks, num_patches, d_model)
#         return x.view(batch, stocks, -1, x.shape[-1])  # (Batch, Stocks, Patches, d_model)
class PatchEmbedding(nn.Module):
    """Encodes time segments (patches) into do d_model"""

    def __init__(self, patch_size, num_features, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = patch_size * num_features
        self.linear = nn.Linear(self.input_dim, d_model)

    def forward(self, x):
        """
        x: (Batch, Stocks, Patches, Patch_Size, Features)
        Return: (Batch, Stocks, Patches, d_model)
        """
        batch, stocks, patches, patch_size, features = x.shape
        assert patch_size * features == self.input_dim, f"Incorrect input size: {patch_size * features} != {self.input_dim}"

        x = x.view(batch, stocks, patches, patch_size * features)  # Combine Patch_Size & Features
        x = self.linear(x)  # Transform to d_model
        return x  # (Batch, Stocks, Patches, d_model)


class IntraPatchAttention(nn.Module):
    """
    Self-Attention inside a single patch.
    It counts the attention inside each time segment separately.
    Each patch is analyzed by MultiheadAttention.
    """
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (Batch, Stocks, Patches, d_model)
        Return: (Batch, Stocks, Patches, d_model)
        """
        batch, stocks, patches, d_model = x.shape
        x = x.view(batch * stocks, patches, d_model)  # (Batch * Stocks, Patches, d_model)
        attn_output, _ = self.attn(x, x, x)
        attn_output = attn_output.view(batch, stocks, patches, d_model)  # (Batch, Stocks, Patches, d_model)
        return self.norm(x.view(batch, stocks, patches, d_model) + self.dropout(attn_output))


class InterPatchAttention(nn.Module):
    """
    Cross-Attention between patches (long term trends).
    Counts attention between different time segments.
    Analyzes the impact of previous patches on future patches.
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (Batch, Stocks, Patches, d_model)
        Return: (Batch, Stocks, Patches, d_model)
        """
        batch, stocks, patches, d_model = x.shape
        x = x.view(batch * stocks, patches, d_model)  # (Batch * Stocks, Patches, d_model)
        attn_output, _ = self.attn(x, x, x)
        attn_output = attn_output.view(batch, stocks, patches, d_model)  # (Batch, Stocks, Patches, d_model)
        return self.norm(x.view(batch, stocks, patches, d_model) + self.dropout(attn_output))

class CrossFormer(nn.Module):
    """CrossFormer with hierarchical attention"""

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        num_stocks,
        patch_size,
        num_features
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbedding(patch_size, num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_stocks, 1, d_model))

        self.intra_blocks = nn.ModuleList([IntraPatchAttention(d_model, nhead, dropout) for _ in range(num_layers)])
        self.inter_blocks = nn.ModuleList([InterPatchAttention(d_model, nhead, dropout) for _ in range(num_layers)])

        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (Batch, Stocks, Patches, Patch_Size, Features)
        """
        batch, stocks, patches, patch_size, features = x.shape

        # Patch Embedding
        x = self.patch_embed(x)  # (Batch, Stocks, Patches, d_model)
        x = x + self.pos_encoder

        # Processing by hierarchical blocks of attitudes
        for intra_attn, inter_attn in zip(self.intra_blocks, self.inter_blocks):
            x = intra_attn(x)  # Attention in patches
            x = inter_attn(x)  # Attention between patches

        # x = x[:, -1, :, :]  # (Batch, 1, Stocks, d_model)
        x = x.view(batch, patches * stocks, self.d_model)

        x = self.projection(x)  # (Batch, 1, Stocks)
        # x = x.permute(0, 2, 1)  # (Batch, Stocks, 1)

        return x  # Poprawny kształt dla predykcji

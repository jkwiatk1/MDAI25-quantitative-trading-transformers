import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm
import math

# --- Core MASTER Modules (Mostly Unchanged Internally) ---

class PositionalEncoding(nn.Module):
    """Injects positional encoding based on the sequence dimension."""
    def __init__(self, d_model, max_len=1000): # Increased max_len default
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Simplified div_term calculation slightly
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Keep pe as buffer, not parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [*, seq_len, d_model] where * means any number of leading dimensions.
        Returns:
            Tensor with positional encodings added.
        """
        # Add positional encoding to the sequence length dimension (assumed to be -2)
        # Slice pe to match the input sequence length
        x = x + self.pe[:x.shape[-2], :]
        return x


class SAttention(nn.Module):
    """Spatial Attention (Across Stocks)."""
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head) # Use d_head for scaling as is common

        self.qkv_layer = nn.Linear(d_model, 3 * d_model) # Combine Q, K, V projections
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            # Use d_ff if provided, otherwise default (e.g., 4*d_model)
            # Note: Original MASTER didn't explicitly define d_ff here, using d_model
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size_eff, n_stocks, d_model]
               (batch_size_eff could be B or B*T depending on reshaping)
        Returns:
            Tensor, shape [batch_size_eff, n_stocks, d_model]
        """
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V
        # qkv: [batch_size_eff, n_stocks, 3 * d_model]
        qkv = self.qkv_layer(x)

        # Split Q, K, V and reshape for multi-head attention
        # q, k, v: [batch_size_eff, n_stocks, n_heads, d_head]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        k = k.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        v = v.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        # Transpose for attention calculation: [batch_size_eff, n_heads, n_stocks, d_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # scores: [batch_size_eff, n_heads, n_stocks, n_stocks]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # context: [batch_size_eff, n_heads, n_stocks, d_head]
        context = torch.matmul(attn_weights, v)

        # Reshape and project output
        # context: [batch_size_eff, n_stocks, n_heads, d_head] -> [batch_size_eff, n_stocks, d_model]
        context = context.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        attention_output = self.out_proj(context)

        # Add residual connection (1st)
        x = residual + self.output_dropout(attention_output)

        # FFN part
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        # Add residual connection (2nd)
        x = residual + x

        return x


class TAttention(nn.Module):
    """Temporal Attention (Across Time Steps)."""
    # Structure is identical to SAttention, just applied differently
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size_eff, seq_len, d_model]
               (batch_size_eff could be B or B*N depending on reshaping)
        Returns:
            Tensor, shape [batch_size_eff, seq_len, d_model]
        """
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_layer(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        k = k.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)
        v = v.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        attention_output = self.out_proj(context)

        x = residual + self.output_dropout(attention_output)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TemporalAttentionAggregation(nn.Module):
    """Final temporal aggregation using attention based on the last time step."""
    def __init__(self, d_model):
        super().__init__()
        # Transformation for query generation (can be identity or a learned proj)
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        # Transformation for key/value (can be identity or a learned proj)
        # Let's keep it simple for now, using the input directly as key/value
        # self.kv_transform = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model) # Scaling factor for attention

    def forward(self, z):
        """
        Args:
            z: Tensor, shape [batch_size_eff, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size_eff, d_model] (aggregated representation)
        """
        # Generate query from the last time step
        query = self.query_transform(z[:, -1, :]).unsqueeze(1) # [B_eff, 1, D]

        # Use input sequence as keys and values (can add kv_transform if needed)
        keys = z # [B_eff, T, D]
        values = z # [B_eff, T, D]

        # Calculate attention scores
        # scores = [B_eff, 1, D] x [B_eff, D, T] -> [B_eff, 1, T]
        scores = torch.matmul(query, keys.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1) # [B_eff, 1, T]

        # Apply attention weights to values
        # output = [B_eff, 1, T] x [B_eff, T, D] -> [B_eff, 1, D]
        output = torch.matmul(attn_weights, values)
        return output.squeeze(1) # [B_eff, D]


# --- Adapted PortfolioMASTER Model ---
class PortfolioMASTER(nn.Module):
    def __init__(self, finance_features_amount, stock_amount, lookback, d_model, d_ff,
                 t_n_heads, s_n_heads, t_dropout, s_dropout, num_encoder_layers: int = 1):
        super().__init__()
        self.d_feat = finance_features_amount
        self.stock_amount = stock_amount
        self.lookback = lookback
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers  # NOT USED NOW

        # Initial feature projection layer
        self.feature_proj = nn.Linear(self.d_feat, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback + 1) # Max len based on lookback

        # Intra-stock temporal attention layer
        self.temporal_attention = TAttention(d_model=d_model, d_ff=d_ff, n_heads=t_n_heads, dropout=t_dropout)

        # Inter-stock spatial attention layer
        self.spatial_attention = SAttention(d_model=d_model, d_ff=d_ff, n_heads=s_n_heads, dropout=s_dropout)

        # Final temporal aggregation layer
        self.temporal_agg = TemporalAttentionAggregation(d_model=d_model)

        # Final linear decoder layer to predict return for each stock
        self.decoder = nn.Linear(d_model, 1) # Output 1 value (return) per stock

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [batch_size, lookback, stock_amount, finance_features_amount]
               (B, T, N, F)
        Returns:
            Output tensor, shape [batch_size, stock_amount, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        assert T == self.lookback, "Input lookback mismatch"
        assert N == self.stock_amount, "Input stock_amount mismatch"
        assert F == self.d_feat, "Input finance_features mismatch"

        # 1. Project features: [B, T, N, F] -> [B, T, N, D] (D=d_model)
        x = self.feature_proj(x)

        # 2. Add Positional Encoding (along Time dimension T)
        # Reshape for PE: [B, T, N, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        x = self.pos_encoder(x)
        # Reshape back: [B*N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        x = x.view(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()

        # 3. Temporal Attention (intra-stock)
        # Reshape for TAttention: [B, T, N, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        x = self.temporal_attention(x)
        # Reshape back: [B*N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        x = x.view(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()

        # 4. Spatial Attention (inter-stock)
        # Reshape for SAttention: [B, T, N, D] -> [B*T, N, D]
        x = x.permute(0, 1, 3, 2).contiguous().view(B * T, self.d_model, N) # Check this reshape carefully
        x = x.permute(0, 2, 1).contiguous() # Now it's [B*T, N, D]
        x = self.spatial_attention(x)
        # Reshape back: [B*T, N, D] -> [B, T, N, D]
        x = x.view(B, T, N, self.d_model)

        # 5. Final Temporal Aggregation
        # Reshape for Aggregation: [B, T, N, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        x = self.temporal_agg(x) # Output: [B*N, D]

        # 6. Decoder
        # Reshape input for decoder: [B*N, D]
        # Apply decoder: [B*N, D] -> [B*N, 1]
        predictions = self.decoder(x)

        # Reshape final output: [B*N, 1] -> [B, N, 1]
        predictions = predictions.view(B, N, 1)

        return predictions
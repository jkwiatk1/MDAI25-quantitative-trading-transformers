from math import ceil
import torch
import torch.nn as nn
from math import sqrt


class DSW_embedding(nn.Module):
    """Dimension-Segment-Wise embedding that segments time series and embeds each segment."""
    
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        """Embed segmented time series.
        
        Args:
            x: Input of shape [batch, ts_len, ts_dim]
        
        Returns:
            Embedded segments of shape [batch, ts_dim, seg_num, d_model]
        """
        batch, ts_len, ts_dim = x.shape
        assert ts_len % self.seg_len == 0, "ts_len must be divisible by seg_len"
        seg_num = ts_len // self.seg_len

        # Segment time dimension: [batch, seg_num, seg_len, ts_dim]
        x_segment = x.view(batch, seg_num, self.seg_len, ts_dim)
        # Rearrange to [batch, ts_dim, seg_num, seg_len]
        x_segment = x_segment.permute(0, 3, 1, 2).contiguous()
        # Flatten to [(batch * ts_dim * seg_num), seg_len]
        x_segment = x_segment.view(batch * ts_dim * seg_num, self.seg_len)

        # Apply linear embedding
        x_embed = self.linear(x_segment)

        # Restore original dimensions: [batch, ts_dim, seg_num, d_model]
        x_embed = x_embed.view(batch, ts_dim, seg_num, -1)

        return x_embed


class FullAttention(nn.Module):
    """Full attention mechanism with scaled dot-product."""
    
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        # Używamy torch.einsum, co jest wydajne i czytelne
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    """Multi-head attention layer with projections."""
    
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    """Two-stage attention: temporal attention followed by cross-dimensional attention."""
    
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.seg_num_expected = seg_num
        self.d_model = d_model
        self.factor = factor

        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(self.seg_num_expected, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        # x shape: [batch_size, ts_d, actual_seg_num, d_model]
        batch, ts_d, actual_seg_num, d_model = x.shape

        if actual_seg_num != self.seg_num_expected:
            raise AssertionError(
                f"Input seg_num {actual_seg_num} != expected seg_num {self.seg_num_expected} for TSA Layer's router.")

        # Cross Time Stage: attention across time segments
        time_in = x.reshape(batch * ts_d, actual_seg_num, d_model)

        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: attention across dimensions
        # Reshape: [(b * ts_d), seg_num, d_model] -> [(b * seg_num), ts_d, d_model]
        dim_send_prep = dim_in.view(batch, ts_d, actual_seg_num, d_model)
        dim_send_prep = dim_send_prep.permute(0, 2, 1, 3).contiguous()
        dim_send = dim_send_prep.view(batch * actual_seg_num, ts_d, d_model)

        # Expand router for batch dimension
        batch_router = self.router.repeat(batch, 1, 1)

        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)  # [(b * seg_num), factor, d_model]
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)  # [(b * seg_num), ts_d, d_model]
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        # Reshape back to [batch, ts_d, seg_num, d_model]
        final_out = dim_enc.view(batch, actual_seg_num, ts_d, d_model)
        final_out = final_out.permute(0, 2, 1, 3).contiguous()

        return final_out


class SegMerging(nn.Module):
    """Merge adjacent segments by concatenating and projecting."""
    
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """Merge segments. Input shape: [batch, ts_d, seg_num, d_model]"""
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            # Pad by repeating last segment
            last_segment = x[:, :, -1:, :]
            padding = last_segment.repeat(1, 1, pad_num, 1)
            x = torch.cat([x, padding], dim=-2)
            seg_num = x.shape[2]

        assert (
            seg_num % self.win_size == 0
        ), "Padded seg_num should be divisible by win_size"
        new_seg_num = seg_num // self.win_size

        # Reshape and merge segments
        x = x.view(batch_size, ts_d, new_seg_num, self.win_size, d_model)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = x.view(batch_size, ts_d, new_seg_num, self.win_size * d_model)

        x = self.norm(x)
        x = self.linear_trans(x)
        return x


class scale_block(nn.Module):
    """Scale block that optionally merges segments and applies two-stage attention."""
    
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, input_seg_num, factor=10):
        super(scale_block, self).__init__()
        self.win_size = win_size
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
            self.output_seg_num = ceil(input_seg_num / win_size)
        else:
            self.merge_layer = None
            self.output_seg_num = input_seg_num

        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(
                    self.output_seg_num, factor, d_model, n_heads, d_ff, dropout
                )
            )

    def forward(self, x):
        """Process through scale block. Input: [batch, ts_d, seg_num, d_model]"""
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """Multi-scale encoder with progressive segment merging."""
    
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()
        current_seg_num = in_seg_num

        # First block with win_size=1 (no merging)
        self.encode_blocks.append(
            scale_block(1, d_model, n_heads, d_ff, block_depth, dropout, current_seg_num, factor)
        )

        # Subsequent blocks with segment merging
        for i in range(1, e_blocks):
            block = scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout, current_seg_num, factor)
            self.encode_blocks.append(block)
            current_seg_num = block.output_seg_num

    def forward(self, x):
        """Process through encoder blocks. Returns list of multi-scale outputs."""
        encode_x = []
        current_x = x
        for block in self.encode_blocks:
            current_x = block(current_x)
            encode_x.append(current_x)
        return encode_x


class PortfolioCrossformer(nn.Module):
    def __init__(self, stock_amount, financial_features, in_len, seg_len, win_size=2, factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=2, dropout=0.1, aggregation_type="avg_pool", device=torch.device("cuda:0")):
        super().__init__()
        self.stock_amount = stock_amount
        self.financial_features = financial_features
        self.data_dim = stock_amount * financial_features
        self.in_len = in_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.d_model = d_model
        self.e_layers = e_layers
        self.aggregation_type = aggregation_type
        self.device = device

        # Calculate padding and segment numbers
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        self.in_seg_num = self.pad_in_len // seg_len

        # Embedding layers
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, self.in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Multi-scale encoder
        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=self.in_seg_num,
            factor=factor,
        )

        # Calculate final segment number after merging
        self.final_seg_num = self.in_seg_num
        if e_layers > 0:
            temp_seg_num = self.in_seg_num
            if win_size > 1:  # Merging zaczyna się od drugiego bloku (i=1)
                for _ in range(e_layers - 1):
                    temp_seg_num = ceil(temp_seg_num / win_size)
            self.final_seg_num = temp_seg_num

        # Agregacja (warunkowo)
        if self.aggregation_type == "avg_pool":
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif self.aggregation_type == "last_segment":
            self.aggregation = None
        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type}")
        self.portfolio_head = nn.Linear(self.data_dim * d_model, self.stock_amount, bias=False)
        # hidden_dim = d_model // 2
        # self.portfolio_head = nn.Sequential(
        #     nn.Linear(self.data_dim * d_model, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, self.stock_amount, bias=False)
        # )

    def forward(self, x):
        # x: [batch_size, lookback, stock_amount, financial_features] == [B, T, N, F]
        B, T, N, F = x.shape
        assert (T == self.in_len and N == self.stock_amount and F == self.financial_features)


        # 1. Reshape & Padding
        x_seq = x.reshape(B, T, N * F);
        if self.in_len_add != 0:
            padding = x_seq[:, :1, :].expand(-1, self.in_len_add, -1)
            x_seq = torch.cat((padding, x_seq), dim=1)  # [B, pad_in_len, data_dim]

        # 2. Embedding + Pos + Norm
        x_embed = self.enc_value_embedding(x_seq)  # [B, data_dim, in_seg_num, D]
        x_embed = x_embed + self.enc_pos_embedding  # Broadcasting
        x_embed = self.pre_norm(x_embed)

        # 3. Encoder
        enc_out_list = self.encoder(x_embed)
        enc_out = enc_out_list[-1]  # [B, data_dim, L_final, D]

        # 4. Aggregation
        if self.aggregation_type == "avg_pool":
            # Reshape for pooling: [B, data_dim, D, L_final] -> [B, D*M, L_final]
            aggregated_features_prep = enc_out.permute(0, 1, 3, 2).contiguous()
            aggregated_features_prep = aggregated_features_prep.view(
                B, self.data_dim * self.d_model, self.final_seg_num
            )
            aggregated_features = self.aggregation(aggregated_features_prep).squeeze(
                -1
            )  # [B, data_dim * D]
        elif self.aggregation_type == "last_segment":
            # Take last segment: [B, data_dim, L_final, D] -> [B, data_dim, D]
            last_segment_features = enc_out[:, :, -1, :]
            # Reshape: [B, data_dim, D] -> [B, data_dim * D]
            aggregated_features = last_segment_features.reshape(
                B, self.data_dim * self.d_model
            )
        else:
            raise ValueError("Should not happen for CrossFormer")

        predicted_logits = self.portfolio_head(aggregated_features)  # [B, N]
        predictions = predicted_logits.unsqueeze(-1)  # [B, N, 1]

        return predictions


def build_CrossFormer(
    # --- Data Shape Parameters ---
    stock_amount: int,  # num_tickers_to_use
    financial_features: int,  # num_tickers_to_use
    in_len: int,  # lookback
    # --- Crossformer Specific Architecture Parameters ---
    seg_len: int,
    win_size: int = 2,
    factor: int = 10,
    aggregation_type: str = "avg_pool",
    e_layers: int = 2,  # num_encoder_layers
    # --- General Transformer Architecture Parameters ---
    d_model: int = 128,
    n_heads: int = 4,  # nhead
    d_ff: int = 256,  # dim_feedforward
    # --- Regularization ---
    dropout: float = 0.1,
    # --- Deployment ---
    device: torch.device = torch.device("cpu"),
) -> PortfolioCrossformer:
    """
    Builds and initializes the PortfolioCrossformer model.

    Args:
        stock_amount (int): Number of different assets/stocks in the portfolio.
                            Determines the output size and part of the internal data_dim.
        financial_features (int): Number of features used to describe each stock
                                  (e.g., close price, volume, RSI). Part of the internal data_dim.
        in_len (int): The lookback period, i.e., the length of the input time series sequence.

        seg_len (int): The length of each segment the input sequence is divided into
                       by the DSW embedding. `in_len` should ideally be divisible by `seg_len`,
                       though the model handles padding if needed. Controls the granularity
                       of the initial time series representation.
        win_size (int): The number of adjacent segments merged in the SegMerging layer
                        at each scale (except the first). Typically 2. Controls the rate
                        of temporal aggregation across encoder layers.
        factor (int): The dimension factor for the router mechanism in the
                      TwoStageAttentionLayer, controlling the bottleneck size for
                      cross-dimension attention.
        e_layers (int): The number of hierarchical encoder blocks (scales). Each block
                        (except the first) includes segment merging and TSA layers.

        d_model (int): The main dimensionality of the embeddings and hidden states
                       throughout the model. Must be divisible by `n_heads`.
        n_heads (int): The number of parallel attention heads in the multi-head
                       attention mechanisms (both time and dimension attention).
        d_ff (int): The dimensionality of the inner hidden layer in the feed-forward
                    networks within the TSA layers. Often 2x or 4x `d_model`.

        dropout (float): Dropout probability applied in various layers for regularization.

        device (torch.device): The device (e.g., 'cuda', 'cpu') to create the model on.

    Returns:
        PortfolioCrossformer: An initialized instance of the model.
    """
    print("-" * 30)
    print(
        f"Building PortfolioCrossformer with aggregation: {aggregation_type} with  parameters:"
    )
    print(
        f"  Data: stock_amount={stock_amount}, financial_features={financial_features}, in_len={in_len}"
    )
    print(
        f"  Crossformer Arch: seg_len={seg_len}, win_size={win_size}, factor={factor}, e_layers={e_layers}"
    )
    print(f"  Transformer Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")

    # --- Parameter Validation (Optional but Recommended) ---
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
    if in_len % seg_len != 0:
        padded_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        print(
            f"Warning: in_len ({in_len}) is not divisible by seg_len ({seg_len}). "
            f"Input will be effectively padded to length {padded_in_len}."
        )

    model = PortfolioCrossformer(
        stock_amount=stock_amount,
        financial_features=financial_features,
        in_len=in_len,  # Passed directly, model handles padding internally if needed
        seg_len=seg_len,
        win_size=win_size,
        factor=factor,
        aggregation_type=aggregation_type,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        device=device  # Pass the device to the model constructor if it uses it internally
        # (e.g., for creating certain tensors directly on the target device, though
        # often just calling .to(device) after creation is sufficient)
    )
    # The .to(device) call is crucial to move all parameters and buffers
    return model.to(device)


if __name__ == "__main__":
    batch_size = 32
    lookback = 96
    stock_amount = 10
    financial_features = 5
    seg_len = 24
    win_size = 2
    factor = 10
    d_model = 128
    d_ff = 256
    n_heads = 4
    e_layers = 2
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if lookback % seg_len != 0:
        print(
            f"Warning: lookback ({lookback}) is not perfectly divisible by seg_len ({seg_len}). Padding will be applied."
        )

    input_data = torch.randn(batch_size, lookback, stock_amount, financial_features).to(
        device
    )

    model = build_CrossFormer(
        stock_amount=stock_amount,
        financial_features=financial_features,
        in_len=lookback,
        seg_len=seg_len,
        win_size=win_size,
        factor=factor,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        device=device,
    ).to(device)

    # Sprawdzenie poprawności obliczenia final_seg_num w modelu
    calculated_final_seg = ceil(lookback / seg_len)
    for _ in range(e_layers - 1):
        if win_size > 1:
            calculated_final_seg = ceil(calculated_final_seg / win_size)
    print(f"Calculated final_seg_num outside model: {calculated_final_seg}")
    print(f"Final_seg_num inside model: {model.final_seg_num}")
    assert (
        model.final_seg_num == calculated_final_seg
    ), "Mismatch in final_seg_num calculation!"

    output_weights = model(input_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_weights.shape}")
    print(f"Output weights (first batch sample): {output_weights[0]}")
    print(f"Sum of weights (first batch sample): {output_weights[0].sum()}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat # Usunięto import

from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math


# --- Klasy pomocnicze (ze zmianami) ---

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # x shape: [batch, ts_len, ts_dim]
        batch, ts_len, ts_dim = x.shape
        # Upewnij się, że ts_len jest podzielne przez seg_len
        assert ts_len % self.seg_len == 0, "ts_len must be divisible by seg_len"
        seg_num = ts_len // self.seg_len

        # x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len) # <--- ZMIANA Z EINOPS
        # 1. Podziel wymiar czasu: [b, seg_num, seg_len, d]
        x_segment = x.view(batch, seg_num, self.seg_len, ts_dim)
        # 2. Zmień kolejność wymiarów: [b, d, seg_num, seg_len]
        x_segment = x_segment.permute(0, 3, 1, 2).contiguous()
        # 3. Połącz wymiary b, d, seg_num: [(b * d * seg_num), seg_len]
        x_segment = x_segment.view(batch * ts_dim * seg_num, self.seg_len)

        x_embed = self.linear(x_segment)  # [(b * d * seg_num), d_model]

        # x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim) # <--- ZMIANA Z EINOPS
        # Przywróć oryginalne wymiary: [b, d, seg_num, d_model]
        x_embed = x_embed.view(batch, ts_dim, seg_num, -1)  # -1 automatycznie dopasuje d_model

        return x_embed


class FullAttention(nn.Module):
    # Bez zmian, nie używa einops
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Używamy torch.einsum, co jest wydajne i czytelne
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    # Bez zmian, nie używa einops
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
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.seg_num = seg_num  # Przechowaj seg_num
        self.d_model = d_model  # Przechowaj d_model
        self.factor = factor  # Przechowaj factor

        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        # x shape: [batch_size, ts_d, seg_num, d_model]
        batch, ts_d, seg_num, d_model = x.shape
        assert seg_num == self.seg_num, f"Input seg_num {seg_num} != configured seg_num {self.seg_num}"
        assert d_model == self.d_model, f"Input d_model {d_model} != configured d_model {self.d_model}"

        # Cross Time Stage
        # time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model') # <--- ZMIANA Z EINOPS
        # Połącz wymiary b i ts_d
        time_in = x.reshape(batch * ts_d, seg_num, d_model)

        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)  # dim_in shape: [(b * ts_d), seg_num, d_model]

        # Cross Dimension Stage
        # dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch) # <--- ZMIANA Z EINOPS
        # 1. Przywróć 4D: [b, ts_d, seg_num, d_model]
        dim_send_prep = dim_in.view(batch, ts_d, seg_num, d_model)
        # 2. Zmień kolejność: [b, seg_num, ts_d, d_model]
        dim_send_prep = dim_send_prep.permute(0, 2, 1, 3).contiguous()
        # 3. Połącz b i seg_num: [(b * seg_num), ts_d, d_model]
        dim_send = dim_send_prep.view(batch * seg_num, ts_d, d_model)

        # batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch) # <--- ZMIANA Z EINOPS
        # 1. Dodaj wymiar batch: [1, seg_num, factor, d_model]
        router_prep = self.router.unsqueeze(0)
        # 2. Rozszerz wymiar batch: [b, seg_num, factor, d_model]
        router_prep = router_prep.expand(batch, -1, -1, -1)  # -1 oznacza zachowanie rozmiaru
        # 3. Połącz wymiary b i seg_num: [(b * seg_num), factor, d_model]
        batch_router = router_prep.reshape(batch * seg_num, self.factor, self.d_model)

        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)  # [(b * seg_num), factor, d_model]
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)  # [(b * seg_num), ts_d, d_model]
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)  # dim_enc shape: [(b * seg_num), ts_d, d_model]

        # final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch) # <--- ZMIANA Z EINOPS
        # 1. Przywróć 4D: [b, seg_num, ts_d, d_model]
        final_out = dim_enc.view(batch, seg_num, ts_d, d_model)
        # 2. Zmień kolejność z powrotem: [b, ts_d, seg_num, d_model]
        final_out = final_out.permute(0, 2, 1, 3).contiguous()

        return final_out


class SegMerging(nn.Module):
    # Bez zmian, nie używa einops
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """ x: B, ts_d, L, d_model """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            # Padowanie w wymiarze segmentów (L)
            # Używamy replikacji ostatnich 'pad_num' segmentów
            padding = x[:, :, -pad_num:, :]
            x = torch.cat([x, padding], dim=-2)
            seg_num = x.shape[2]  # Zaktualizuj seg_num po paddingu

        # Upewnij się, że seg_num jest teraz podzielne przez win_size
        assert seg_num % self.win_size == 0, "Padded seg_num should be divisible by win_size"
        new_seg_num = seg_num // self.win_size

        # Zamiast pętli i listy, użyj reshape i permute
        # 1. Reshape do: [B, ts_d, new_seg_num, win_size, d_model]
        x = x.view(batch_size, ts_d, new_seg_num, self.win_size, d_model)
        # 2. Przenieś win_size obok d_model: [B, ts_d, new_seg_num, d_model, win_size]
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        # 3. Połącz d_model i win_size: [B, ts_d, new_seg_num, win_size * d_model]
        x = x.view(batch_size, ts_d, new_seg_num, self.win_size * d_model)

        x = self.norm(x)
        x = self.linear_trans(x)  # [B, ts_d, new_seg_num, d_model]
        return x


class scale_block(nn.Module):
    # Bez zmian, nie używa einops
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10):
        super(scale_block, self).__init__()
        self.win_size = win_size  # Przechowaj win_size
        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
            # Oblicz seg_num po mergingu dla warstw TSA
            current_seg_num = ceil(seg_num / win_size)
        else:
            self.merge_layer = None
            current_seg_num = seg_num

        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(current_seg_num, factor, d_model, n_heads, d_ff, dropout))

    def forward(self, x):
        # x: [B, ts_d, L, d_model]
        if self.merge_layer is not None:
            x = self.merge_layer(x)  # Kształt L się zmienia
        for layer in self.encode_layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    # Bez zmian, nie używa einops
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num=10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()
        # Pierwszy blok bez mergingu
        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor))
        # Kolejne bloki z mergingiem
        current_seg_num = in_seg_num
        for i in range(1, e_blocks):
            # Przekazujemy seg_num *przed* mergingiem do konstruktora scale_block
            self.encode_blocks.append(
                scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout, current_seg_num, factor))
            # Aktualizujemy current_seg_num dla następnego bloku (po potencjalnym mergingu w tym bloku)
            if win_size > 1:
                current_seg_num = ceil(current_seg_num / win_size)

    def forward(self, x):
        # x: [B, ts_d, L_in, d_model]
        encode_x = []  # Przechowuje wyjścia z różnych skal

        # Przetwarzanie przez bloki
        current_x = x
        for block in self.encode_blocks:
            current_x = block(current_x)  # Wyjście z bloku staje się wejściem do następnego
            encode_x.append(current_x)

        # encode_x to lista tensorów o kształtach:
        # [B, ts_d, L_in, d_model],
        # [B, ts_d, L_in/win, d_model],
        # [B, ts_d, L_in/win^2, d_model], ...
        return encode_x


# --- Zmodyfikowana klasa główna PortfolioCrossformer (ze zmianami) ---

class PortfolioCrossformer(nn.Module):
    def __init__(self, stock_amount, financial_features, in_len, seg_len,
                 win_size=2, factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=2,  # Zmniejszono domyślne e_layers
                 dropout=0.1, aggregation_type='avg_pool',  # Dodano typ agregacji
                 device=torch.device('cuda:0')):
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

        # Padding i obliczenie liczby segmentów
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        self.in_seg_num = self.pad_in_len // seg_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, self.in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)  # Norma na wejściu do enkodera

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1,
                               dropout=dropout, in_seg_num=self.in_seg_num, factor=factor)

        # Obliczanie final_seg_num (bez zmian)
        self.final_seg_num = self.in_seg_num
        if e_layers > 0:
            for _ in range(e_layers - 1):
                if win_size > 1:
                    self.final_seg_num = ceil(self.final_seg_num / win_size)

        # Agregacja (warunkowo)
        if self.aggregation_type == 'avg_pool':
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif self.aggregation_type == 'last_segment':
            self.aggregation = None  # Nie potrzebujemy modułu
        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type}")

        self.portfolio_head = nn.Linear(self.data_dim * d_model, self.stock_amount,
                                        bias=False)
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
        assert T == self.in_len and N == self.stock_amount and F == self.financial_features

        # 1. Reshape i Padding
        x_seq = x.reshape(B, T, N * F)
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
        if self.aggregation_type == 'avg_pool':
            # Reshape for pooling: [B, data_dim, D, L_final] -> [B, D*M, L_final]
            aggregated_features_prep = enc_out.permute(0, 1, 3, 2).contiguous()
            aggregated_features_prep = aggregated_features_prep.view(B, self.data_dim * self.d_model,
                                                                     self.final_seg_num)
            aggregated_features = self.aggregation(aggregated_features_prep).squeeze(-1)  # [B, data_dim * D]
        elif self.aggregation_type == 'last_segment':
            # Take last segment: [B, data_dim, L_final, D] -> [B, data_dim, D]
            last_segment_features = enc_out[:, :, -1, :]
            # Reshape: [B, data_dim, D] -> [B, data_dim * D]
            aggregated_features = last_segment_features.reshape(B, self.data_dim * self.d_model)
        else:
            # Już sprawdzono w __init__
            raise ValueError("Should not happen")

        predicted_logits = self.portfolio_head(aggregated_features)  # [B, N]
        predictions = predicted_logits.unsqueeze(-1)  # [B, N, 1]

        return predictions



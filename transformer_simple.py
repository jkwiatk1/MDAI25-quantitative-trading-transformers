import numpy as np
import torch
import math
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Generowanie sztucznego szeregu czasowego
def generate_data(n_samples=200, noise=0.1):
    x = np.linspace(0, 4 * np.pi, n_samples)
    y = np.sin(x) + noise * np.random.randn(n_samples)
    return y

# 2. Przygotowanie danych treningowych i testowych
def create_sequences(data, lookback):
    sequences = []
    targets = []
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])
        targets.append(data[i + lookback])
    return np.array(sequences), np.array(targets)

# 3. Definicja modelu transformera
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int = 5000) -> None:
        """
        seq_len: maximum length of sentence
        dropout: to make model less overfit
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply the sin/cos to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # become a tensor with size (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.size(1), :]).requires_grad_(
            False
        )  # requires_grad_ -> this tensor will not be learn
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)  # standard deviation
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

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

        self.w_q = nn.Linear(
            d_model, d_model, bias=False
        )  # Wq, TODO check bias = False
        self.w_k = nn.Linear(
            d_model, d_model, bias=False
        )  # Wk, TODO check bias = False
        self.w_v = nn.Linear(
            d_model, d_model, bias=False
        )  # Wv, TODO check bias = False

        self.w_o = nn.Linear(
            self.head_dim * self.h, d_model, bias=False
        )  # Wo, in paper it is d_v * h, d_v == d_k == head_dim,
        # self.head_dim * self.h should be == d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]

        # (batch, h, seq_len, head_dim) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            head_dim
        )  # @: matrix multiplication
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        q_prim = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        k_prim = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        v_prim = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # divide to smaller matrix's (batch, seq_len, d_model) --> (batch, seq_len, h, head_dim),
        # embeddings split into edge parts (batch [0], seq [1] is not  splited), -->
        # -->{transpose}-> (batch, h, seq_len, head_dim)
        q_prim = q_prim.view(
            q_prim.shape[0], q_prim.shape[1], self.h, self.head_dim
        ).transpose(1, 2)
        k_prim = k_prim.view(
            k_prim.shape[0], k_prim.shape[1], self.h, self.head_dim
        ).transpose(1, 2)
        v_prim = v_prim.view(
            v_prim.shape[0], v_prim.shape[1], self.h, self.head_dim
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            q_prim, k_prim, v_prim, mask, self.dropout
        )

        # (batch, h, seq_len, head_dim) --> (batch, seq_len, h, head_dim) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.head_dim)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections_1 = ResidualConnection(dropout)  # Sequential.modules
        self.residual_connections_2 = ResidualConnection(dropout)  # Sequential.modules

    def forward(self, x, src_mask):
        x = self.residual_connections_1(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections_2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        d_model=512,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Create custom encoder layers
        encoder_blocks = []
        for _ in range(num_encoder_layers):
            encoder_self_attention_block = MultiHeadAttentionBlock(
                d_model, nhead, dropout
            )
            feed_forward_block = FeedForwardBlock(d_model, dim_feedforward, dropout)
            encoder_block = EncoderBlock(
                encoder_self_attention_block, feed_forward_block, dropout
            )
            encoder_blocks.append(encoder_block)

        self.transformer_encoder = Encoder(nn.ModuleList(encoder_blocks))
        self.d_model = d_model
        self.projection = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, None)
        x = x.mean(dim=1)
        x = self.projection(x)
        return x


# Ustawienia
lookback = 20
n_samples = 200
test_split = 0.2

# Dane
data = generate_data(n_samples)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Podział na zbiory treningowy i testowy
train_size = int((1 - test_split) * len(data_scaled))
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

train_sequences, train_targets = create_sequences(train_data, lookback)
test_sequences, test_targets = create_sequences(test_data, lookback)

train_sequences = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

# Parametry modelu
d_model = 128
nhead = 2
num_encoder_layers = 1
dim_feedforward = 128
dropout = 0.05
model = TransformerModel(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout).to(device)

# 5. Trenowanie modelu
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 50

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(train_sequences)
    loss = criterion(predictions.squeeze(), train_targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

# 6. Testowanie modelu
model.eval()
with torch.no_grad():
    test_predictions = model(test_sequences).squeeze()
    test_loss = criterion(test_predictions, test_targets)
    print(f"Test Loss: {test_loss.item():.4f}")

# 7. Wizualizacja wyników
test_predictions = test_predictions.cpu().numpy()
test_targets = test_targets.cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(range(len(test_targets)), test_targets, label="True Values", linestyle="dashed")
plt.plot(range(len(test_predictions)), test_predictions, label="Predictions")
plt.legend()
plt.title("Transformer Predictions on Test Set")
plt.show()

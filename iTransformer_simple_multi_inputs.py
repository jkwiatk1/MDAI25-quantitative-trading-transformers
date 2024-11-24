import numpy as np
import torch
import math
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Generating time series with multiple features
def generate_multivariate_data(n_samples=200, num_features=3, noise=0.1):
    x = np.linspace(0, 4 * np.pi, n_samples)
    data = []
    for i in range(num_features):
        y = np.sin(x + i) + noise * np.random.randn(
            n_samples
        )
        data.append(y)
    return np.stack(data, axis=1)  # matrix [n_samples, num_features]


# 2. Train, test data preparation
def create_sequences(data, lookback):
    sequences = []
    targets = []
    for i in range(len(data) - lookback):
        sequences.append(data[i : i + lookback])
        targets.append(data[i + lookback, 0])  # forecast first feature
    return np.array(sequences), np.array(targets)


# 3 iTransformers definition
class DataEmbeddingInverted(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        """
        Data embedding layer for inverted input dimensions.
        input_dim: number of features in the input (number of variables).
        d_model: dimension of the embedding space.
        """
        super(DataEmbeddingInverted, self).__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        x: Tensor [Batch, Features, Sequence Length]
        x_mark: Tensor [Batch, Covariates, Sequence Length] (opcjonalne)
        """
        # Osadzanie wartości
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=1)
        x = self.value_embedding(x)
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
        self.model_type = "iTransformer"

        # self.encoder = nn.Linear(input_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = DataEmbeddingInverted(input_dim, d_model, dropout)

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

    def forward(self, x, x_mark=None):
        """
        x: Tensor [Batch, Sequence Length, Features]
        x_mark: Tensor [Batch, Sequence Length, Covariates] (optional)
        """
        # Inversion of dimensions on [Batch, Features, Sequence Length]
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x_mark = x_mark.permute(0, 2, 1)

        x = self.embedding(x, x_mark)  # [Batch, Features, d_model]

        # x = x.permute(2, 0, 1)  # [Sequence Length, Batch, Features]
        x = self.transformer_encoder(x, None)

        # Back to [Batch, Sequence Length, Features]
        # x = x.permute(0, 2, 1)
        # x = [Batch, Features, Sequence Length] => x = [Batch, Sequence Length, Features]
        x = self.projection(x).permute(0, 2, 1)[:,-1,:]
        # TODO to jest tymczasowo tak chyba nie powinno byc...
        x = x.mean(dim=1)
        return x


# Params
lookback = 20
n_samples = 200
num_features = 3
test_split = 0.2
batch_size = 32

# data generating
data = generate_multivariate_data(n_samples, num_features)
scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(num_features)]
data_scaled = np.column_stack(
    [
        scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        for i, scaler in enumerate(scalers)
    ]
)

# Train test split
train_size = int((1 - test_split) * len(data_scaled))
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

train_sequences, train_targets = create_sequences(train_data, lookback)
test_sequences, test_targets = create_sequences(test_data, lookback)


# train_sequences = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
train_sequences = torch.tensor(train_sequences, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
# test_sequences = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

train_dataset = TensorDataset(train_sequences, train_targets)
test_dataset = TensorDataset(test_sequences, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Model params
input_dim = num_features
d_model = 128
nhead = 2
num_encoder_layers = 1
dim_feedforward = 128
dropout = 0.05
model = TransformerModel(
    input_dim=lookback,
    # input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
).to(device)

# 5. Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 50

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for batch_sequences, batch_targets in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_sequences)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item()
        epoch_loss += loss.item() * batch_sequences.size(0)

    epoch_loss /= len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

# 6. Test
model.eval()
test_loss = 0.0
predictions_list = []
targets_list = []

with torch.no_grad():
    for batch_sequences, batch_targets in test_loader:
        test_predictions = model(batch_sequences).squeeze()
        loss = criterion(test_predictions, batch_targets)
        # test_loss += loss.item()
        test_loss += loss.item() * batch_sequences.size(0)
        predictions_list.append(test_predictions.cpu())
        targets_list.append(batch_targets.cpu())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

test_predictions = torch.cat(predictions_list).numpy()
test_targets = torch.cat(targets_list).numpy()

plt.figure(figsize=(10, 5))
plt.plot(range(len(test_targets)), test_targets, label="True Values", linestyle="dashed")
plt.plot(range(len(test_predictions)), test_predictions, label="Predictions")
plt.legend()
plt.title("Transformer Predictions on Test Set")
plt.show()

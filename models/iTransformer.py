import torch
from torch import nn
from models.modules import (
    LayerNormalization,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


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
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=1)
        x = self.value_embedding(x)
        return self.dropout(x)


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


class iTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        d_model=512,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_features=1,
        columns_amount=1,
    ):
        super(iTransformerModel, self).__init__()
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
        self.num_features = num_features
        # self.projection = nn.Linear(d_model, num_features)
        self.projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_feedforward * columns_amount, num_features)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)

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
        x = self.transformer_encoder(x, None)
        x = self.projection(x)


        # # Back to [Batch, Sequence Length, Features]
        # # x = x.permute(0, 2, 1)
        # # x = [Batch, Features, Sequence Length] => x = [Batch, Sequence Length, Features]
        # x = self.projection(x).permute(0, 2, 1)
        # x = x[:, :, -1]  # [:, -1, :]
        return x

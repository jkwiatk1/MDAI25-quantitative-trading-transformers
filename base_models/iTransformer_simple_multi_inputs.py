import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from models.PortfolioiTransformer import iTransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Generating time series with multiple features
def generate_multivariate_data(n_samples=200, num_features=3, noise=0.1):
    x = np.linspace(0, 4 * np.pi, n_samples)
    data = []
    for i in range(num_features):
        y = np.sin(x + i) + noise * np.random.randn(n_samples)
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
model = iTransformerModel(
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
plt.plot(
    range(len(test_targets)), test_targets, label="True Values", linestyle="dashed"
)
plt.plot(range(len(test_predictions)), test_predictions, label="Predictions")
plt.legend()
plt.title("Transformer Predictions on Test Set")
plt.show()

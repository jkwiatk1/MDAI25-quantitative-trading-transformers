import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from models.PortfolioVanillaTransformer import TransformerModel
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data(n_samples=200, noise=0.1):
    x = np.linspace(0, 4 * np.pi, n_samples)
    y = np.sin(x) + noise * np.random.randn(n_samples)
    return y


# 2. Przygotowanie danych treningowych i testowych
def create_sequences(data, lookback):
    sequences = []
    targets = []
    for i in range(len(data) - lookback):
        sequences.append(data[i : i + lookback])
        targets.append(data[i + lookback])
    return np.array(sequences), np.array(targets)


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

train_sequences = (
    torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
test_sequences = (
    torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

# Parametry modelu
d_model = 128
nhead = 2
num_encoder_layers = 1
dim_feedforward = 128
dropout = 0.05
model = TransformerModel(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
).to(device)

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
plt.plot(
    range(len(test_targets)), test_targets, label="True Values", linestyle="dashed"
)
plt.plot(range(len(test_predictions)), test_predictions, label="Predictions")
plt.legend()
plt.title("Transformer Predictions on Test Set")
plt.show()

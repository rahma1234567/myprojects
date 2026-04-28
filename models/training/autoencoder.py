#This file loads the preprocessed data, creates autoencoder architecture, trains autoencoder morel,
#calculates reconstruction error, flags anomalies and saves the results. 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Load the preprocessed + engineered data
X_train = pd.read_csv("feature_engineering/X_train_final.csv")
X_test = pd.read_csv("feature_engineering/X_test_final.csv")

# Convert to data into tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

#Create DataLoader for batching
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

#Creating an Autoencoder Architecture with 23 features
input_dim = X_train.shape[1]  # 23 features

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Now Train Autoencoder using Adam optimiser
model = Autoencoder(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.6f}")

print("\nTraining completed")

#Calculate the reconstruction error on test set
model.eval()
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    reconstruction_errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

#Flag anomalies using 95th percentile threshold
threshold = np.percentile(reconstruction_errors, 95)
print(f"\nReconstruction Error Threshold (95th percentile): {threshold:.6f}")

X_test = X_test.copy()
X_test["reconstruction_error"] = reconstruction_errors
X_test["is_fraud"] = (reconstruction_errors > threshold).astype(int)

#Evaluate
total = len(X_test)
flagged = X_test["is_fraud"].sum()
print("TOTAL TRANSACTIONS:", total)
print("FLAGGED AS FRAUD:", flagged)
print("FRAUD PERCENTAGE: {:.2f}%".format(flagged / total * 100))
print("\nReconstruction Error Stats:")
print(X_test["reconstruction_error"].describe())

#Save model and results
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("models/training", exist_ok=True)
torch.save(model.state_dict(), "models/saved_models/autoencoder.pth")
X_test.to_csv("models/training/autoencoder_results.csv", index=True, index_label="original_index")
print("\nModel saved and results exported.")
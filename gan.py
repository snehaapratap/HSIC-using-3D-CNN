import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class HyperspectralDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load Data
mat = sio.loadmat('Indian_pines_corrected.mat')  # Adjust the filename if necessary
gt = sio.loadmat('Indian_pines_gt.mat')  # Ground truth file
data = mat['indian_pines_corrected']  # Replace with the correct key from your .mat file
labels = gt['indian_pines_gt']  # Replace with the correct key from your .mat file

# Optional PCA
n_components = 30
reshaped_data = data.reshape(-1, data.shape[2])
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(reshaped_data)
data_pca = data_pca.reshape(data.shape[0], data.shape[1], n_components)

# Create Patches
patch_size = 5
pad = patch_size // 2
padded = np.pad(data_pca, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
X, y = [], []
for i in range(pad, padded.shape[0] - pad):
    for j in range(pad, padded.shape[1] - pad):
        if labels[i - pad, j - pad] != 0:
            patch = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            X.append(patch)
            y.append(labels[i - pad, j - pad] - 1)

X = np.stack(X)
y = np.array(y)

X = np.transpose(X, (0, 3, 1, 2))  # NCHW

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
train_loader = DataLoader(HyperspectralDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(HyperspectralDataset(X_test, y_test), batch_size=64)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(n_components, np.max(y) + 1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
for epoch in range(50):
    model.train()
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        out = model(data)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        out = model(data)
        _, pred = torch.max(out, 1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

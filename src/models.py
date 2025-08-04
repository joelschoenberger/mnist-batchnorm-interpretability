import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """Original 2‑conv MNIST network (no Batch‑Norm)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dp1   = nn.Dropout(0.25)
        self.dp2   = nn.Dropout(0.50)
        self.fc1   = nn.Linear(64 * 12 * 12, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dp2(x)
        return F.log_softmax(self.fc2(x), dim=1)

class CNN_BN(nn.Module):
    """Same topology with Batch‑Norm layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.dp1   = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 12 * 12, 128)
        self.bn3   = nn.BatchNorm1d(128)
        self.dp2   = nn.Dropout(0.50)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = x.flatten(1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dp2(x)
        return F.log_softmax(self.fc2(x), dim=1)

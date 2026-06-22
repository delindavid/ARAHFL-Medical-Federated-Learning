import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Helper Block
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


# =========================
# 1. TinyCNN
# =========================
class TinyCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(TinyCNN, self).__init__()

        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 32)

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# =========================
# 2. SmallCNN
# =========================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SmallCNN, self).__init__()

        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# =========================
# 3. MediumCNN
# =========================
class MediumCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(MediumCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


# =========================
# 4. WideCNN
# =========================
class WideCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(WideCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


# =========================
# 5. LightweightResCNN
# =========================
class LightweightResCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(LightweightResCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        residual = F.relu(self.conv1(x))

        x = F.relu(self.conv2(residual))

        x = x + residual

        x = self.pool(x)

        residual = F.relu(self.conv3(x))

        x = F.relu(self.conv4(residual))

        x = x + residual

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
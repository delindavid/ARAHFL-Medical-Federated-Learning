import torch
import torch.nn as nn
import numpy as np


class Client:
    def __init__(self, model, train_loader, device=None):

        self.model = model
        self.train_loader = train_loader

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

    # -------------------------
    # MixUp
    # -------------------------
    def mixup_data(self, x, y, alpha=0.4):

        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

        batch_size = x.size(0)

        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        y_a = y
        y_b = y[index]

        return mixed_x, y_a, y_b, lam

    # -------------------------
    # Consistency Loss
    # -------------------------
    def consistency_loss(self, x):

        noise = torch.randn_like(x) * 0.05

        pred1 = self.model(x)

        pred2 = self.model(x + noise)

        return torch.mean((pred1 - pred2) ** 2)

    # -------------------------
    # Reliability Score
    # -------------------------
    def compute_reliability(self, avg_loss):

        reliability = 1.0 / (1.0 + avg_loss)

        return float(reliability)

    # -------------------------
    # Training
    # -------------------------
    def train(self, epochs=1):

        self.model.train()

        total_loss = 0

        for _ in range(epochs):

            for data, target in self.train_loader:

                data = data.to(self.device)

                target = target.squeeze().long().to(self.device)

                mixed_x, y_a, y_b, lam = self.mixup_data(
                    data,
                    target
                )

                self.optimizer.zero_grad()

                output = self.model(mixed_x)

                mix_loss = (
                    lam * self.criterion(output, y_a)
                    + (1 - lam) * self.criterion(output, y_b)
                )

                cons_loss = self.consistency_loss(mixed_x)

                loss = mix_loss + 0.1 * cons_loss

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        reliability = self.compute_reliability(avg_loss)

        return avg_loss, reliability

    # -------------------------
    # Get Weights
    # -------------------------
    def get_weights(self):

        return {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    # -------------------------
    # Set Weights
    # -------------------------
    def set_weights(self, weights):

        self.model.load_state_dict(weights)
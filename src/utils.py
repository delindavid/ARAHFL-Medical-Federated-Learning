import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Subset

from medmnist import PathMNIST
from medmnist import INFO
from torchvision import transforms


# ==================================
# Load PathMNIST
# ==================================
def load_pathmnist(size=28):

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = PathMNIST(
        split="train",
        download=True,
        transform=data_transform,
        size=size
    )

    test_dataset = PathMNIST(
        split="test",
        download=True,
        transform=data_transform,
        size=size
    )

    return train_dataset, test_dataset


# ==================================
# Dirichlet Non-IID Partition
# ==================================
def partition_dirichlet(
    dataset,
    num_clients=5,
    alpha=0.5
):

    labels = np.array(dataset.labels).flatten()

    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):

        idx = np.where(labels == cls)[0]

        np.random.shuffle(idx)

        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, num_clients)
        )

        proportions = (
            np.cumsum(proportions) * len(idx)
        ).astype(int)[:-1]

        split_idx = np.split(
            idx,
            proportions
        )

        for client_id in range(num_clients):

            client_indices[client_id].extend(
                split_idx[client_id]
            )

    client_datasets = []

    for indices in client_indices:

        client_datasets.append(
            Subset(dataset, indices)
        )

    return client_datasets


# ==================================
# Corruption Injection
# ==================================
def inject_corruption(
    images,
    noise_level=0.1
):

    noise = (
        torch.randn_like(images)
        * noise_level
    )

    corrupted = images + noise

    corrupted = torch.clamp(
        corrupted,
        0,
        1
    )

    return corrupted


# ==================================
# Accuracy Plot
# ==================================
def plot_accuracy(
    baseline_acc,
    arahfl_acc
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        baseline_acc,
        label="FedAvg"
    )

    plt.plot(
        arahfl_acc,
        label="ARAHFL"
    )

    plt.xlabel("Round")

    plt.ylabel("Accuracy")

    plt.title(
        "Accuracy Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ==================================
# Reliability Plot
# ==================================
def plot_reliability(
    reliability_history
):

    plt.figure(figsize=(8, 5))

    reliability_history = np.array(
        reliability_history
    )

    for client_id in range(
        reliability_history.shape[1]
    ):

        plt.plot(
            reliability_history[:, client_id],
            label=f"Client {client_id}"
        )

    plt.xlabel("Round")

    plt.ylabel("Reliability")

    plt.title(
        "Reliability Evolution"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ==================================
# Aggregation Weight Plot
# ==================================
def plot_weights(
    weight_history
):

    plt.figure(figsize=(8, 5))

    weight_history = np.array(
        weight_history
    )

    for client_id in range(
        weight_history.shape[1]
    ):

        plt.plot(
            weight_history[:, client_id],
            label=f"Client {client_id}"
        )

    plt.xlabel("Round")

    plt.ylabel("Aggregation Weight")

    plt.title(
        "Aggregation Weight Evolution"
    )

    plt.legend()

    plt.grid(True)

    plt.show()
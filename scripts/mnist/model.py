"""Define the model for the two moons toy problem."""
import os

import torch
from torchvision.datasets import MNIST


def load_mnist_data(device, n_nominal, n_failure, n_failure_eval):
    """Load MNIST data."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mnist_dir = os.path.join(current_dir, "../notebooks/data/mnist")
    trainset = MNIST(root=mnist_dir, download=True, train=True)

    # Restrict to just 8s and 9s (sorry 9, you're just a failed 8)
    labels = trainset.targets
    nominal = trainset.data[labels == 8]
    failure = trainset.data[labels == 9]

    nominal = nominal.float() / 255
    failure = failure.float() / 255

    # Reduce the number of failures in the dataset
    failure_perm = torch.randperm(len(failure))
    failure_eval = failure[failure_perm[n_failure : n_failure + n_failure_eval]]
    failure = failure[failure_perm[:n_failure]]
    nominal = nominal[torch.randperm(len(nominal))[:n_nominal]]

    # Flatten
    nominal = nominal.reshape(-1, 28 * 28)
    failure = failure.reshape(-1, 28 * 28)
    failure_eval = failure_eval.reshape(-1, 28 * 28)

    return nominal.to(device), failure.to(device), failure_eval.to(device)

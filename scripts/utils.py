"""Define useful functions"""
import numpy as np
import ot
import torch
from torchmetrics.classification import BinaryF1Score


def kl_divergence(p_dist, q_dist, num_particles=10):
    """KL divergence between two distributions."""
    p_samples, p_logprobs = p_dist.rsample_and_log_prob((num_particles,))
    q_logprobs = q_dist.log_prob(p_samples)

    kl_divergence = (p_logprobs - q_logprobs).mean(dim=0)

    return kl_divergence


def cross_entropy(p_samples, q_dist):
    """Cross entropy between two distributions."""
    q_logprobs = q_dist.log_prob(p_samples)

    cross_entropy = -q_logprobs.mean(dim=0)

    return cross_entropy


def f_score(nominal_samples, failure_samples, nominal_dist, failure_dist):
    """Compute the f score of a likelihood ratio test for failure."""
    # Concatenate the samples
    samples = torch.cat([nominal_samples, failure_samples], dim=0)
    true_labels = torch.cat(
        [
            torch.zeros(nominal_samples.shape[0]),
            torch.ones(failure_samples.shape[0]),
        ],
        dim=0,
    )

    # Compute the likelihood ratio and classify as a failure if is >= 1 (more likely
    # to be from the failure distribution than the nominal distribution)
    likelihood_ratio = failure_dist.log_prob(samples) - nominal_dist.log_prob(samples)
    predicted_labels = (likelihood_ratio >= 0).float()

    # Get the f score
    f_score = BinaryF1Score()(predicted_labels, true_labels)
    return f_score


def sinkhorn(p_samples, q_samples, epsilon=1.0):
    # Uniform weights on the samples
    n = p_samples.shape[0]
    a, b = (np.ones((n,)) / n, np.ones((n,)) / n)
    # EMD loss matrix
    M = ot.dist(
        p_samples.detach().cpu().numpy(),
        q_samples.detach().cpu().numpy(),
    )
    # Solve regularized EMD (sinkhorn)
    sinkhorn_dist = ot.sinkhorn2(a, b, M, epsilon, method="sinkhorn_log")
    return sinkhorn_dist


class RBF(torch.nn.Module):
    # Implementation from Yiftach Beer on GitHub
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(torch.nn.Module):
    # Implementation from Yiftach Beer on GitHub
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def simple_mmd(X, Y):
    Z = torch.vstack([X, Y])
    L2_distances = torch.cdist(Z, Z) ** 2
    K = torch.exp(-0.5 * L2_distances)
    X_size = X.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()
    return XX - 2 * XY + YY


class ContextFreeBase(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, _):
        return self.base()


if __name__ == "__main__":
    # Test sinkhorn and mmd
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    sinkhorns = [
        sinkhorn(
            torch.randn(100, 2),
            torch.randn(100, 2) + torch.tensor([i, 0.0]),
        )
        for i in tqdm(torch.linspace(-2, 2, 100))
    ]

    mmd = [
        MMDLoss()(
            torch.randn(1000, 2),
            torch.randn(1000, 2) + torch.tensor([i, 0.0]),
        )
        for i in tqdm(torch.linspace(-2, 2, 100))
    ]

    plt.plot(torch.linspace(-2, 2, 100), sinkhorns, label="Sinkhorn")
    plt.plot(torch.linspace(-2, 2, 100), mmd, label="MMD")
    plt.legend()
    plt.show()

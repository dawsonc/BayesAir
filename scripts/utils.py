"""Define useful functions"""
import torch


def kl_divergence(p_dist, q_dist, num_particles=10):
    """KL divergence between two distributions."""
    p_samples, p_logprobs = p_dist.rsample_and_log_prob((num_particles,))
    q_logprobs = q_dist.log_prob(p_samples)

    kl_divergence = (p_logprobs - q_logprobs).mean(dim=0)

    return kl_divergence


class ContextFreeBase(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, _):
        return self.base()

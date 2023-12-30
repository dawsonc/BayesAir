import wandb

from .wn_network import train


def train_from_config(config):
    return train(
        config.top_n,
        config.days,
        config.svi_steps,
        config.n_samples,
        config.svi_lr,
        config.plot_every,
    )


def train_sweep():
    wandb.init(project="bayes-air")
    train_from_config(wandb.config)


if __name__ == "__main__":
    train_sweep()

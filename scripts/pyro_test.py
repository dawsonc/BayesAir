"""Run the simulation for a simple two-airport network."""
import pprint

import pyro
import pyro.distributions as dist
import torch


def main():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    def model(obs=None):
        z = pyro.sample("z", dist.Normal(0.0, 2.0))
        for i in pyro.plate("i", 20):
            x = pyro.sample(
                f"x_{i}", dist.Normal(z, 0.1), obs=None if obs is None else obs[i]
            )

    model_graph = pyro.render_model(
        model,
        model_args=(),
        render_params=False,
        render_distributions=False,
    )
    model_graph.render("tmp/two_airport_network", cleanup=True)

    # Print dependencies
    print("Dependencies:")
    pprint.pprint(pyro.infer.inspect.get_dependencies(model, ()))

    # Set up MCMC inference
    num_samples = 200
    warmup_steps = 200
    num_chains = 4
    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        disable_validation=True,
    )
    mcmc.run(torch.randn(20) + 1.0)
    mcmc.summary(prob=0.9)


if __name__ == "__main__":
    main()

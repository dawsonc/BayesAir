import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

if __name__ == "__main__":
    # Define your list of wandb projects
    project_list = [
        # "two-moons-sensitivity",
        "swi-sensitivity",
        # "uav-sensitivity",
    ]

    # Define a display name for each project
    project_display_names = {
        "two-moons-sensitivity": "2D",
        "swi-sensitivity": "SWI",
        "uav-sensitivity": "UAV",
    }

    normalize_by = {
        "two-moons-sensitivity": -2,
        "swi-sensitivity": -1,
        "uav-sensitivity": -22,
    }

    # Define metrics of interest
    metrics = {
        "Failure/ELBO (eval)": "Test set ELBO (nats/dim)",
    }

    # Initialize a dictionary to store the summary statistics for each run
    summary_stats = []

    # Initialize the wandb API
    api = wandb.Api()

    # Loop through each project
    for project in project_list:
        print(project)
        # Get all runs for the current project
        runs = api.runs(project, per_page=400)

        # Extract summary statistics for each run
        for run in runs:
            if run.state == "failed":
                continue

            # Average the metric over the last 50 steps
            window_len = 50
            metric_means = {}
            for metric in metrics:
                metric_means[metric] = run.history(keys=[metric], samples=1000)[
                    -window_len:
                ][metric].mean()

            summary_stats.append(
                {
                    "project": project_display_names[project],
                    "K": run.config["n_calibration_permutations"],
                    "Calibration": True
                    if run.config["calibration_weight"] > 0
                    else False,
                }
                | {
                    name: metric_means[metric] / normalize_by[project]
                    for metric, name in metrics.items()
                }
            )

    stats_df = pd.DataFrame(summary_stats)

    # Melt the dataframe for use with seaborn
    stats_df = stats_df.melt(
        id_vars=["project", "K", "Calibration"],
        value_vars=[
            col
            for col in stats_df.columns
            if col != "project" and col != "K" and col != "Calibration"
        ],
        var_name="metric",
        value_name="value",
    )

    # g = sns.FacetGrid(
    #     stats_df, row="metric", col="project", height=3, aspect=1.5, sharey=False
    # )
    # g.set_titles(template="{col_name}")
    # g.map_dataframe(
    #     sns.lmplot,
    #     data=stats_df,
    #     x="K",
    #     y="value",
    #     hue="Calibration",
    #     order=1,
    #     x_jitter=0.1,
    # )

    # # Label y axes with the name of the metric
    # for ax, metric_name in zip(g.axes[:, 0], stats_df["metric"].unique()):
    #     ax.set_ylabel(metric_name)

    sns.lmplot(
        data=stats_df,
        x="K",
        y="value",
        hue="Calibration",
        col="project",
        row="metric",
        order=1,
        x_jitter=0.1,
        height=3,
        aspect=1.5,
    )
    ax = plt.gca()
    ax.set_ylabel("Test set ELBO (nats/dim)")
    ax.set_xlabel("Number of subsamples $K$")
    ax.set_title("")

    # fig = g.figure
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig("scripts/plotting/plots/sensitivity.png", dpi=300)

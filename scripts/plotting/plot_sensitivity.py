import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

if __name__ == "__main__":
    # Define your list of wandb projects
    project_list = [
        "two-moons-sensitivity",
    ]

    # Define a display name for each project
    project_display_names = {
        "two-moons-sensitivity": "2D",
    }

    project_dimensions = {
        "two-moons-sensitivity": 2,
    }

    # Define metrics of interest
    metrics = {
        "Failure/ELBO (eval)": "ELBO (test set)",
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
            summary_stats.append(
                {
                    "project": project_display_names[project],
                    "K": run.config["n_calibration_permutations"],
                }
                | {name: run.summary.get(metric) for metric, name in metrics.items()}
            )

    stats_df = pd.DataFrame(summary_stats)

    # Melt the dataframe for use with seaborn
    stats_df = stats_df.melt(
        id_vars=["project", "K"],
        value_vars=[col for col in stats_df.columns if col != "project" and col != "K"],
        var_name="metric",
        value_name="value",
    )

    g = sns.FacetGrid(
        stats_df, row="metric", col="project", height=3, aspect=1, sharey=False
    )
    g.set_titles(template="{col_name}")
    g.map_dataframe(
        sns.lineplot,
        data=stats_df,
        x="K",
        y="value",
    )

    fig = g.figure
    fig.tight_layout()
    plt.savefig("scripts/plotting/plots/sensitivity.png", dpi=300)

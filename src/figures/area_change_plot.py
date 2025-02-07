import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")

    # Initialize an empty list to store the resulting DataFrames
    df_lst = []

    # Define the list of perturbation strengths (f1_val) to iterate over
    perturb_strengths = [70, 80, 90, 100]

    for perturb_strength in perturb_strengths:
        area_change_lst = []

        # Define the path to the ground truth filtered DataFrame for the current perturb_strength
        gt_path = (
            Path("results/feature_tables_gt")
            / f"LHCC35_GT_qc_{perturb_strength}.parquet"
        )

        # Load the ground truth DataFrame
        df_gt = pd.read_parquet(gt_path)

        # Define the directory containing perturbed feature tables for the current perturb_strength
        perturbed_dir = Path("results") / f"feature_tables_{perturb_strength}_qc"

        # Iterate over all parquet files in the perturbed directory using tqdm for progress tracking
        for df_path in perturbed_dir.rglob("*.parquet"):
            # Load the perturbed DataFrame
            df_pert = pd.read_parquet(df_path)

            # Compute the area change ratio and append to the list
            area_change_ratio = df_pert.Area.values / df_gt.Area.values
            area_change_lst.append(area_change_ratio)

        # Convert the list of area change ratios to a NumPy array
        area_change = np.array(area_change_lst)

        # Calculate the median area change across all perturbations for each cell
        change_arr = np.median(area_change, axis=0)

        # Create a new DataFrame to store the results
        df = pd.DataFrame(
            {
                "Perturbation": [str(perturb_strength)] * change_arr.size,
                "Median area change": change_arr,
            }
        )

        # Append the resulting DataFrame to the list
        df_lst.append(df)

    df_full = pd.concat(df_lst)
    df_full.to_feather("results/area_change_median_full.feather")
    print(df_full.Perturbation.unique())
    print(df_full.shape)
    print(df_full.groupby("Perturbation").median().round(3))

    # customPalette = sns.color_palette(["#E69F00", "#56B4E9", "#009E73", "#F0E442"])
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    figsize = (2.2, 2.2)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    sns.kdeplot(
        data=df_full,
        x="Median area change",
        hue="Perturbation",
        palette=customPalette,
        cut=0,
        linewidth=2,
    )
    plt.xlabel("Median relative area change")
    plt.ylabel("Density")

    plt.xlim([0.75, 1.25])
    # plt.xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    plt.legend("", frameon=False)
    plt.title("Cell size comparison")
    # fig.tight_layout()
    plt.savefig("figures/area_change_median_kde.pdf", transparent=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.violinplot(
        data=df_full,
        x="Perturbation",
        y="Median area change",
        hue="Perturbation",
        palette=customPalette,
        ax=ax,
        saturation=1,
        # inner=None,
    )
    plt.xlabel("Perturbation")
    plt.ylabel("Median relative area change")

    # plt.xlim([0.75, 1.25])
    # plt.xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    plt.legend("", frameon=False)
    plt.title("Cell size comparison")
    fig.tight_layout()
    plt.savefig("figures/area_change_median_violin.pdf", transparent=True)


if __name__ == "__main__":
    main()

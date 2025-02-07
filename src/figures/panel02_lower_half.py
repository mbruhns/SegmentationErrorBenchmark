import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    plt.style.use("matplotlibrc.txt")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    c2st_df = pd.read_parquet("results/c2st-results.parquet")
    c2st_df.rmv = c2st_df.rmv.map({True: "No", False: "Yes"})
    c2st_df = c2st_df.loc[c2st_df.rmv == "No"]

    corr_df = pd.read_feather("results/corr_mae_full.feather")
    area_df = pd.read_feather("results/area_change_median_full.feather")

    FIG_WIDTH = 17.4 / 2.54  # in inches
    FIG_HEIGHT = 6.2 / 2.54  # in inches

    figsize = (FIG_WIDTH, FIG_HEIGHT)
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=figsize)
    ax = axs.ravel()

    sns.violinplot(
        data=area_df,
        x="Perturbation",
        y="Median area change",
        hue="Perturbation",
        palette=customPalette,
        ax=ax[0],
        saturation=1,
    )

    ax[0].set_xlabel("Perturbation")
    ax[0].set_ylabel("Median relative area change")
    ax[0].legend("", frameon=False)
    ax[0].set_title("Cell size comparison")

    sns.violinplot(
        data=corr_df,
        x="perturbation",
        y="MAE",
        hue="perturbation",
        palette=customPalette,
        ax=ax[1],
        saturation=1,
    )

    ax[1].set_xlabel("Perturbation")
    ax[1].set_ylabel("Mean absolute error")
    ax[1].legend("", frameon=False)
    ax[1].set_title("Feature correlation comparison")

    sns.violinplot(
        data=c2st_df,
        x="perturb_strength",
        y="median_accuracy",
        hue="perturb_strength",
        palette=customPalette,
        ax=ax[2],
        saturation=1,
    )

    ax[2].set_xlabel("Perturbation")
    ax[2].set_ylabel("Median accuracy")
    ax[2].legend("", frameon=False)
    ax[2].set_title("Classifier 2-Sample Test")

    plt.savefig("figures/panel02_lower.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

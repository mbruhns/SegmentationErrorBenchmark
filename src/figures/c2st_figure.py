import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    plt.style.use("matplotlibrc.txt")

    df = pd.read_parquet("results/c2st-results.parquet")
    df.rmv = df.rmv.map({True: "No", False: "Yes"})

    df = df.loc[df.rmv == "No"]

    figsize = (2.2, 2.2)

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    sns.violinplot(
        data=df,
        x="perturb_strength",
        y="mean_accuracy",
        hue="perturb_strength",
        palette=customPalette,
        ax=ax,
        saturation=1,
        inner=None,
    )

    plt.xlabel("Perturbation")
    plt.ylabel("Mean accuracy")

    # sns.despine(top=True, right=True)
    plt.legend("", frameon=False)
    plt.title("Classifier 2-Sample Test")
    fig.tight_layout()
    plt.savefig("figures/c2st_mean_plot.pdf", dpi=300, transparent=True)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    sns.violinplot(
        data=df,
        x="perturb_strength",
        y="median_accuracy",
        hue="perturb_strength",
        palette=customPalette,
        ax=ax,
        saturation=1,
        # inner=None,
    )

    plt.xlabel("Perturbation")
    plt.ylabel("Median accuracy")

    # sns.despine(top=True, right=True)
    plt.legend("", frameon=False)
    plt.title("Classifier 2-Sample Test")
    fig.tight_layout()
    plt.savefig("figures/c2st_median_plot.pdf", dpi=300, transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

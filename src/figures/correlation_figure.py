import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    df_lst = []
    partial_df_path = Path("results/MarkerCorrelation/")

    for df_path in partial_df_path.rglob("*.feather"):
        df_lst.append(pd.read_feather(df_path))

    corr_df = pd.concat(df_lst)
    corr_df.to_feather("results/corr_mae_full.feather")
    print(corr_df)

    figsize = (2.2, 2.2)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    sns.violinplot(
        data=corr_df,
        x="perturbation",
        y="MAE",
        hue="perturbation",
        palette=customPalette,
        ax=ax,
        saturation=1,
        # inner=None,
    )

    plt.xlabel("Perturbation")
    plt.ylabel("Mean absolute error")
    # plt.xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    plt.legend("", frameon=False)
    plt.title("Feature correlation comparison")
    plt.savefig("figures/correlation_mae.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

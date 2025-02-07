import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")
    sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    df_lst = []
    partial_df_path = Path("results/HDBSCAN/")

    for df_path in partial_df_path.rglob("*.feather"):
        df_lst.append(pd.read_feather(df_path))

    hdb_df = pd.concat(df_lst)
    hdb_df.to_csv("results/hdb.csv", index=False)

    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    ax = ax.ravel()
    for ax_id, (f1_val, df_partial) in enumerate(hdb_df.groupby("perturbation")):
        sns.boxplot(
            data=df_partial,
            x="min_cluster_size",
            hue="min_samples",
            y="arand",
            ax=ax[ax_id],
        )
        ax[ax_id].set_title(f1_val)

    plt.savefig("figures/hdb_fig.pdf", bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

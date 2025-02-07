import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils import mad_errorbars_with_scale
from matplotlib.transforms import ScaledTranslation


def main():
    plt.style.use("matplotlibrc.txt")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    err_width = 2

    df_lst = []
    partial_df_path = Path("results/KMeans/")

    for df_path in partial_df_path.rglob("*.feather"):
        df_lst.append(pd.read_feather(df_path))

    kmeans_df = pd.concat(df_lst)

    kmeans_df.to_feather("results/combined_kmeans.feather")

    kmeans_df["variance"] = kmeans_df.variance.replace(
        {"log1p_true": "log1p", "log1p_false": "None"}
    )

    ### ARAND PLOT ###

    # Create the subplot mosaic layout
    fig, axd = plt.subplot_mosaic(
        mosaic="AAB;CCD", layout="constrained", figsize=(5.2, 3)
    )

    for label, ax in axd.items():
        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                ax.transAxes
                + ScaledTranslation(-65 / fig.dpi, 0 / fig.dpi, fig.dpi_scale_trans)
            ),
            fontsize=12,
            va="top",
            fontfamily="Arial",
            fontweight="bold",
        )

    # Filter out the axes we are going to use
    scale_factor = 1

    sns.barplot(
        data=kmeans_df.loc[kmeans_df.variance == "None"],
        x="k",
        y="arand",
        hue="perturbation",
        ax=axd["A"],
        palette=customPalette,
        estimator="median",
        errorbar=mad_errorbars_with_scale(scale_factor),
        err_kws={"linewidth": err_width},
        saturation=1,
    )

    sns.barplot(
        data=kmeans_df,
        x="perturbation",
        y="arand",
        hue="variance",
        ax=axd["B"],
        palette="Set2",
        estimator="median",
        errorbar=mad_errorbars_with_scale(scale_factor),
        err_kws={"linewidth": err_width},
        saturation=1,
    )

    sns.barplot(
        data=kmeans_df.loc[kmeans_df.variance == "log1p"].copy(),
        x="k",
        y="arand",
        hue="perturbation",
        ax=axd["C"],
        palette=customPalette,
        estimator="median",
        errorbar=mad_errorbars_with_scale(scale_factor),
        err_kws={"linewidth": err_width},
        saturation=1,
    )

    sns.barplot(
        data=kmeans_df,
        x="k",
        y="arand",
        ax=axd["D"],
        estimator="median",
        errorbar=mad_errorbars_with_scale(scale_factor),
        err_kws={"linewidth": err_width},
        saturation=1,
        color="gray",
    )

    # axd["A"].legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=4, title="Perturbation strength"
    handles_a, labels_a = axd["A"].get_legend_handles_labels()
    axd["A"].set_xlabel("")
    axd["A"].set_ylabel("ARI")
    axd["A"].get_legend().remove()

    axd["B"].set_xlabel("Perturbation level")
    axd["B"].set_ylabel("")
    handles_b, labels_b = axd["B"].get_legend_handles_labels()
    axd["B"].get_legend().remove()

    axd["C"].set_xlabel("Number of clusters")
    axd["C"].set_ylabel("ARI")
    axd["C"].get_legend().remove()
    # axd["C"].yaxis.set_ticklabels([])

    axd["D"].set_xlabel("Number of clusters")
    axd["D"].set_ylabel("")
    axd["D"].set_ylim([0.5, 1])
    # axd["D"].get_legend().remove()
    # axd["D"].yaxis.set_ticklabels([])
    fig.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="outside upper center",
        ncol=6,
        title="Perturbation strength/preprocessing",
    )
    plt.savefig("figures/kmeans_arand.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

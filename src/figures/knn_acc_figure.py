import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
from pathlib import Path
from utils import mad_errorbars_with_scale


def main():
    plt.style.use("matplotlibrc.txt")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    df_lst = []
    partial_df_path = Path("results/Phenotyping/")

    for df_path in partial_df_path.rglob("phenotyping_partial_*.feather"):
        df_lst.append(pd.read_feather(df_path))

    df_phenotyping = pd.concat(df_lst)
    scale_factor = 1
    err_width = 2

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    df_lst = []
    partial_df_path = Path("results/KNN_ACC/")

    for df_path in partial_df_path.rglob("*.feather"):
        df_lst.append(pd.read_feather(df_path))

    knn_acc_df = pd.concat(df_lst)
    knn_acc_df = knn_acc_df.loc[(knn_acc_df.k.isin([10, 50, 75, 100, 250]))].copy()

    fig, axd = plt.subplot_mosaic(
        mosaic="ABBC;ABBD",  # DAAB;DAAC
        layout="constrained",
        figsize=(5.2, 2.6),
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

    sns.barplot(
        data=df_phenotyping,
        x="Perturbation",
        y="BalancedAccuracy",
        hue="Perturbation",
        estimator="median",
        palette=customPalette,
        ax=axd["A"],
        saturation=1,
        errorbar=mad_errorbars_with_scale(scale_factor),
        err_kws={"linewidth": err_width},
    )

    estimator = "median"
    scale_factor = 1

    sns.barplot(
        data=knn_acc_df,
        x="k",
        y="KNN-Acc",
        hue="perturbation",
        palette=customPalette,
        gap=0,
        saturation=1.0,
        ax=axd["B"],
        width=0.8,
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )
    sns.barplot(
        data=knn_acc_df,
        x="perturbation",
        y="KNN-Acc",
        gap=0.2,
        saturation=1.0,
        ax=axd["C"],
        hue="perturbation",
        palette=customPalette,
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )
    sns.barplot(
        data=knn_acc_df,
        x="k",
        y="KNN-Acc",
        gap=0.3,
        saturation=1.0,
        ax=axd["D"],
        color="gray",
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )

    axd["B"].get_legend().remove()
    axd["C"].get_legend().remove()
    handles, labels = axd["B"].get_legend_handles_labels()

    axd["B"].tick_params(axis="x", labelrotation=45)
    axd["C"].tick_params(axis="x", labelrotation=45)
    axd["D"].tick_params(axis="x", labelrotation=45)
    axd["A"].tick_params(axis="x", labelrotation=45)

    axd["B"].set_xlabel("k")
    axd["C"].set_xlabel("Perturbation")
    axd["D"].set_xlabel("k")

    axd["B"].set_ylabel("kNN-acc")
    axd["C"].set_ylabel("kNN-acc")
    axd["D"].set_ylabel("kNN-acc")

    axd["A"].set_ylabel("Balanced accuracy")
    axd["A"].get_legend().remove()

    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        title="Perturbation strength",
    )
    plt.savefig("figures/barplot_KNN-Acc_median.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

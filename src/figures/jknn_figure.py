import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
from pathlib import Path
from utils import mad_errorbars_with_scale


def main():
    plt.style.use("matplotlibrc.txt")
    df_lst = []
    partial_df_path = Path("results/JKNN/")

    for df_path in partial_df_path.rglob("*.feather"):
        df_lst.append(pd.read_feather(df_path))

    jacc_df = pd.concat(df_lst)

    jacc_df = jacc_df.loc[
        (jacc_df.scaling == "scale_false") & (jacc_df.variance == "log1p_false")
    ].copy()
    jacc_df = jacc_df.loc[(jacc_df.k.isin([10, 50, 75, 100, 250]))].copy()

    # Group by "k" and "perturbation" and calculate the element-wise median
    jacc_df = (
        jacc_df.groupby(["k", "perturbation"])["JKNN"]
        .apply(lambda x: np.median(np.vstack(x), axis=0))
        .reset_index()
    )

    jacc_df = jacc_df.explode("JKNN")

    # jacc_df = jacc_df.sample(n=500_000, random_state=0)
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    err_width = 2

    FIG_WIDTH = 17.4 / 2.54  # in inches

    fig, axd = plt.subplot_mosaic(
        mosaic="AABC", layout="constrained", figsize=(FIG_WIDTH, 2.5)
    )

    # "AAB;AAC"

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

    estimator = "median"
    scale_factor = 1

    sns.barplot(
        data=jacc_df,
        x="k",
        y="JKNN",
        hue="perturbation",
        palette=customPalette,
        gap=0,
        saturation=1.0,
        ax=axd["A"],
        width=0.8,
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )
    sns.barplot(
        data=jacc_df,
        x="perturbation",
        y="JKNN",
        gap=0.2,
        saturation=1.0,
        ax=axd["B"],
        hue="perturbation",
        palette=customPalette,
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )
    sns.barplot(
        data=jacc_df,
        x="k",
        y="JKNN",
        gap=0.2,
        saturation=1.0,
        ax=axd["C"],
        color="gray",
        errorbar=mad_errorbars_with_scale(scale_factor),
        estimator=estimator,
        err_kws={"linewidth": err_width},
    )

    handles, labels = axd["A"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        title="Perturbation strength",
    )

    axd["A"].get_legend().remove()
    axd["B"].get_legend().remove()
    # axd["A"].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, title="Perturbation strength"

    axd["A"].tick_params(axis="x", labelrotation=45)
    axd["B"].tick_params(axis="x", labelrotation=45)
    axd["C"].tick_params(axis="x", labelrotation=45)

    axd["A"].set_xlabel("k")
    axd["B"].set_xlabel("Perturbation")
    axd["C"].set_xlabel("k")

    plt.savefig("figures/barplot_jknn_median.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")
    plot_channels = ["CD45", "HNFalpha", "CD3", "CD11b"]
    fig, ax = plt.subplots(4, 4, figsize=(3.5, 3.74), sharex=True, sharey=False)
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    bw_adjust = 1

    f1_val_lst = [70, 80, 90, 100]

    for j, perturb_strength in enumerate(f1_val_lst):
        df_gt_original = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{perturb_strength}.parquet"
        )
        channels = df_gt_original.columns[:-4].to_list()
        channels.sort()
        df_gt_original = df_gt_original[channels].copy()

        df_gt_lst = []
        df_pert_lst = []

        for df_path in Path(f"results/feature_tables_{perturb_strength}_qc/").rglob(
            "*.parquet"
        ):
            df_pert = pd.read_parquet(df_path)
            df_pert = df_pert[channels].copy()

            df_gt = df_gt_original.copy()

            max_normalizer = pd.concat([df_gt, df_pert]).max(axis=0)

            df_gt = df_gt / max_normalizer
            df_pert = df_pert / max_normalizer

            df_gt_lst.append(df_gt)
            df_pert_lst.append(df_pert)

        df_gt_full = pd.concat(df_gt_lst)
        df_pert_full = pd.concat(df_pert_lst)

        for i, m in enumerate(plot_channels):
            x_ = df_gt_full[m]
            y_ = df_pert_full[m]

            residual = y_ - x_
            lim_threshold = residual.mean() + 2.75 * residual.std()

            sns.kdeplot(
                residual, ax=ax[j, i], color=customPalette[j], bw_adjust=bw_adjust
            )

            # ax[j,i].set_box_aspect(1)
            ax[i, j].tick_params(axis="both", labelsize=8)
            if j == 0:
                ax[j, i].set_title(m, fontsize=9)
            else:
                ax[j, i].set_title("")
            ax[j, i].set_xlabel("")
            ax[j, i].set_ylabel("")
            ax[j, i].set_xlim([-lim_threshold, lim_threshold])
        # ax[j,i].tick_params(axis='x', labelrotation=45)

    fig.autofmt_xdate(rotation=45)
    plt.subplots_adjust(wspace=0.6, bottom=0.16)
    fig.supxlabel("Marker expression residual", fontsize=10)
    fig.supylabel("Density", fontsize=10)
    plt.suptitle("Feature value comparison")
    # fig.tight_layout()
    plt.savefig(f"figures/marker_kde_bw-{bw_adjust}.pdf", transparent=True)


if __name__ == "__main__":
    main()

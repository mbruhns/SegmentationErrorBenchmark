import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    df_lst = []

    partial_df_path = Path("results/Phenotyping/")
    for df_path in partial_df_path.rglob("sankey_partial_*.feather"):
        df_lst.append(pd.read_feather(df_path))

    df = pd.concat(df_lst)

    celltype_to_broad_group = {
        "T cell": "Immune cells",
        "MAIT cell": "Immune cells",
        "CD8+ T cell": "Immune cells",
        "CD4+ T cell": "Immune cells",
        "Treg": "Immune cells",
        "B cell": "Immune cells",
        "NK cell": "Immune cells",
        "Neutrophil Granulocyte": "Immune cells",
        "Kupffer cell": "Immune cells",
        "M2 Macrophage": "Immune cells",
        "Dendritic cell": "Immune cells",
        "Immune cell": "Immune cells",
        "Parenchymal cell": "Parenchymal cells",
        "Hepatocytes/Tumor": "Parenchymal cells",
        "Biliary epithelial cells": "Parenchymal cells",
        "Vascular endothelial cells": "Parenchymal cells",
        "LSEC": "Parenchymal cells",
        "Unknown": "Unknown",
    }

    celltype_to_group = {
        "T cell": "Lymphocytes",
        "MAIT cell": "Lymphocytes",
        "CD8+ T cell": "Lymphocytes",
        "CD4+ T cell": "Lymphocytes",
        "Treg": "Lymphocytes",
        "B cell": "Lymphocytes",
        "NK cell": "Lymphocytes",
        "Neutrophil Granulocyte": "Myeloid Cells",
        "Kupffer cell": "Myeloid Cells",
        "M2 Macrophage": "Myeloid Cells",
        "Dendritic cell": "Myeloid Cells",
        "Immune cell": "Myeloid Cells",  # General immune marker
        "Parenchymal cell": "Parenchymal Cells",
        "Hepatocytes/Tumor": "Parenchymal Cells",
        "Biliary epithelial cells": "Parenchymal Cells",
        "Vascular endothelial cells": "Parenchymal Cells",
        "LSEC": "Parenchymal Cells",
        "Unknown": "Unknown",
    }

    t_cell_subtypes = {
        "T cell": "T cells",
        "MAIT cell": "T cells",
        "CD8+ T cell": "T cells",
        "CD4+ T cell": "T cells",
        "Treg": "T cells",
        "B cell": "B cell",
        "NK cell": "NK cell",
        "Neutrophil Granulocyte": "Neutrophil Granulocyte",
        "Kupffer cell": "Kupffer cell",
        "M2 Macrophage": "M2 Macrophage",
        "Dendritic cell": "Dendritic cell",
        "Immune cell": "Immune cell",
        "Parenchymal cell": "Parenchymal cell",
        "Hepatocytes/Tumor": "Hepatocytes/Tumor",
        "Biliary epithelial cells": "Biliary epithelial cells",
        "Vascular endothelial cells": "Vascular endothelial cells",
        "LSEC": "LSEC",
        "Unknown": "Unknown",
    }

    cell_type_abbrev = {
        "Hepatocytes/Tumor": "Hep/Tumor",
        "M2 Macrophage": "M2 Mac",
        "LSEC": "LSEC",
        "Parenchymal cell": "Parenchymal",
        "Neutrophil Granulocyte": "Neut Gran",
        "CD4+ T cell": "CD4+ T",
        "MAIT cell": "MAIT",
        "Dendritic cell": "DC",
        "T cell": "T cell",
        "Unknown": "Unknown",
        "Treg": "Treg",
        "CD8+ T cell": "CD8+ T",
        "NK cell": "NK",
        "Biliary epithelial cells": "BECs",
        "Immune cell": "Immune",
        "Kupffer cell": "Kupffer",
        "B cell": "B cell",
        "Vascular endothelial cells": "VECs",
    }

    FIG_WIDTH = 8.6 / 2.54  # in inches

    figsize = (FIG_WIDTH, FIG_WIDTH)

    for pert_strength in [70, 80, 90, 100]:
        # for grouping_id, mapping in enumerate([celltype_to_broad_group, celltype_to_group, t_cell_subtypes, None]):
        for grouping_id, mapping in enumerate(
            [celltype_to_broad_group, celltype_to_group, t_cell_subtypes, None]
        ):
            fig, ax = plt.subplots(
                1, 1, figsize=figsize, sharey=True, sharex=True, layout="constrained"
            )
            ctr = 0
            df_partial = df.loc[df.pert == pert_strength].copy()

            if mapping is not None:
                df_partial.y_true = df_partial.y_true.map(mapping)
                df_partial.y_pred = df_partial.y_pred.map(mapping)
            if mapping is None:
                df_partial.replace(cell_type_abbrev, inplace=True)

            le = LabelEncoder()
            le.fit(df_partial.y_true)

            y_true = le.transform(df_partial.y_true)
            y_pred = le.transform(df_partial.y_pred)

            cm = confusion_matrix(y_true, y_pred, normalize="true")
            cm_df = pd.DataFrame(data=cm, columns=le.classes_, index=le.classes_)

            labels = cm_df.map(lambda v: f"{v * 100:.0f}" if v >= 0.01 else "")
            sns.heatmap(
                cm_df,
                annot=labels,
                linewidth=0.5,
                square=True,
                fmt="",
                cbar=False,
                ax=ax,
                vmin=0,
                vmax=1,
                cmap="rocket_r",
                annot_kws={"fontsize": 8},
                xticklabels=1,
                yticklabels=1,
            )
            ax.set_title("")
            ctr += 1

            plt.savefig(
                f"figures/phenotyping_sankey_{pert_strength}_{grouping_id}.pdf",
                transparent=True,
            )


if __name__ == "__main__":
    main()

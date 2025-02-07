import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.style.use("matplotlibrc.txt")
    fig, ax = plt.subplots(4, 4, figsize=(3.8, 3.74), sharex=True, sharey=False)
    sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

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

    res_df = df_pert_full - df_gt_full
    print(res_df)


if __name__ == "__main__":
    main()

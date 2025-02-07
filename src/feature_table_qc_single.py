import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import median_abs_deviation
import re


def is_outlier(M, nmads: int):
    """Detects outliers based on the median absolute deviation."""
    median = np.median(M)
    mad = median_abs_deviation(M)
    return (M < median - nmads * mad) | (M > median + nmads * mad)


def filter_data(df, min_cell_size, max_cell_size, nmads):
    """Applies filtering criteria to remove outliers and size-constrained data."""
    return df.loc[
        (df["Area"] > min_cell_size)
        & (df["Area"] < max_cell_size)
        & ~is_outlier(df["HOECHST 2"], nmads)
    ]


def load_and_filter_data(base_path, min_cell_size, max_cell_size, nmads):
    """Loads and filters all parquet files in the given base path."""
    index_lists = []
    for df_path in Path(base_path).rglob("*.parquet"):
        df = pd.read_parquet(df_path)
        df_filtered = filter_data(df, min_cell_size, max_cell_size, nmads)
        index_lists.append(set(df_filtered.index))
    return index_lists


def save_filtered_data(df_gt, f1_val, shared_ind, base_path):
    """Saves filtered ground truth and corresponding feature tables."""
    filtered_gt = df_gt.loc[df_gt.index.isin(shared_ind)].copy()
    filtered_gt.to_parquet(f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet")

    # Save filtered feature tables
    pattern = r"_(\d+)\.parquet"
    for df_path in Path(base_path).rglob("*.parquet"):
        perturb_id = int(re.findall(pattern, str(df_path))[0])
        df = pd.read_parquet(df_path)
        df_filtered = df.loc[shared_ind].copy()
        assert filtered_gt.shape[0] == df_filtered.shape[0], (
            "Mismatch of selected cells!"
        )
        df_filtered.to_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{perturb_id}_qc.parquet"
        )


def main():
    # Configuration
    min_cell_size = 40
    max_cell_size = 2060
    nmads = 4
    f1_val_lst = [70, 80, 90, 100]

    # Load and filter ground truth data
    df_gt = pd.read_parquet("data/LHCC35_HALONUC_groundtruth.parquet")
    original_size = df_gt.shape[0]
    df_gt = filter_data(df_gt, min_cell_size, max_cell_size, nmads)
    print(f"Lost cells: {df_gt.shape[0] / original_size:.2%}")

    gt_labels = df_gt.Label.to_list()

    pattern = r"_(\d+)\.parquet"

    for f1_val in f1_val_lst:
        # Load and filter perturbation data
        base_path = f"results/feature_tables_{f1_val}/"

        # This is kind of stupid since it saves the same data multiple times
        # but this way it is compatible with the original code.
        df_gt.to_parquet(f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet")

        print(f1_val)
        sample_change_lst = []

        for df_path in Path(base_path).rglob("*.parquet"):
            perturb_id = int(re.findall(pattern, str(df_path))[0])
            df_pert = pd.read_parquet(df_path)
            pert_org_size = df_pert.shape[0]
            df_pert = df_pert.loc[df_pert.Label.isin(gt_labels)]
            pert_filt_size = df_pert.shape[0]

            sample_change_lst.append(pert_filt_size / pert_org_size)

            df_pert.to_parquet(
                f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{perturb_id}_qc.parquet"
            )

        print(sample_change_lst)
        print(f"Min: {np.min(sample_change_lst):.3f}")
        print(f"Median: {np.median(sample_change_lst):.3f}")
        print(f"Max: {np.max(sample_change_lst):.3f}")
        print(32 * "- ")


if __name__ == "__main__":
    main()

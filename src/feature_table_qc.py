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
    df_gt = filter_data(df_gt, min_cell_size, max_cell_size, nmads)
    original_size = df_gt.shape[0]

    for f1_val in f1_val_lst:
        # Load and filter perturbation data
        base_path = f"results/feature_tables_{f1_val}/"
        index_lists = load_and_filter_data(
            base_path, min_cell_size, max_cell_size, nmads
        )
        index_lists.append(set(df_gt.index))
        shared_ind = list(set.intersection(*index_lists))
        print(len(shared_ind))

        # Print remaining data percentage
        print(
            f"{len(shared_ind) / original_size:.2%} of the original data remains after filtering."
        )

        # Check for sample bias in index selection
        sample_shared_ind_ratios = [
            len(shared_ind) / len(index_list) for index_list in index_lists
        ]
        assert min(sample_shared_ind_ratios) >= 0.8, sample_shared_ind_ratios

        # Save filtered data
        save_filtered_data(df_gt, f1_val, shared_ind, base_path)

        # Calculate and validate mean area change
        area_filtered_gt = df_gt.loc[df_gt.index.isin(shared_ind), "Area"].mean()
        area_lst = [
            pd.read_parquet(df_path).loc[shared_ind, "Area"].mean()
            for df_path in Path(base_path).rglob("*.parquet")
        ]
        rel_area_change = np.mean(area_lst) / area_filtered_gt
        assert 0.98 < rel_area_change < 1.02, (
            f"Unsuitable relative area with {rel_area_change:.2%}"
        )
        print(f"Mean area change: {rel_area_change:.2f}\n")


if __name__ == "__main__":
    main()

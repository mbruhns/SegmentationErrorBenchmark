import pandas as pd
import numpy as np
from numba import njit, prange
from itertools import product
from utils import chunk_list

import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler
from cuml.neighbors import NearestNeighbors

import argparse
from tqdm import tqdm


@njit(parallel=True)
def calculate_jaccard_index_numba(indices_a, indices_b):
    assert indices_a.shape == indices_b.shape
    m, n = indices_a.shape

    jacc_arr = np.empty(m)

    for i in prange(m):
        size_intersection = np.intersect1d(indices_a[i], indices_b[i]).size
        jacc_arr[i] = size_intersection / (2 * n - size_intersection)

    return jacc_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    # Warmup for Numba
    _ = calculate_jaccard_index_numba(
        indices_a=np.random.randint(0, 5, (5, 10)),
        indices_b=np.random.randint(0, 5, (5, 10)),
    )

    maximum_id = args.max
    current_id = args.current - 1

    pert_strength_lst = [70, 80, 90, 100]
    number_perturbations = 20
    results = []

    full_lst = list(
        product(
            pert_strength_lst,
            range(number_perturbations),
            ("log1p_true", "log1p_false"),
            ("scale_true", "scale_false"),
        )
    )
    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    n_neighbors_arr = np.array([10, 20, 30, 40, 50, 75, 100, 250])
    metric = "euclidean"

    # Progress bar for the local schedule
    for f1_val, mask_id, log1p, scale in tqdm(
        local_schedule, desc="Processing Local Schedule"
    ):
        df_gt = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet"
        )
        df_pert = pd.read_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{mask_id}_qc.parquet"
        )

        channels = df_gt.columns[:-4].drop(labels="HOECHST 2")
        cuda_gt = cudf.from_dataframe(df_gt[channels], allow_copy=True)
        cuda_pert = cudf.from_dataframe(df_pert[channels], allow_copy=True)

        if log1p == "log1p_true":
            cuda_gt[channels] = cp.log1p(cuda_gt[channels].values)
            cuda_pert[channels] = cp.log1p(cuda_pert[channels].values)

        if scale == "scale_true":
            cuda_gt[channels] = StandardScaler().fit_transform(cuda_gt[channels].values)
            cuda_pert[channels] = StandardScaler().fit_transform(
                cuda_pert[channels].values
            )

        model_gt = NearestNeighbors(
            n_neighbors=n_neighbors_arr.max() + 1,
            algorithm="brute",
            metric=metric,
        )
        model_gt.fit(cuda_gt)
        _, indices_gt = model_gt.kneighbors(cuda_gt)
        nn_indices_gt = indices_gt.values.get()

        model_pert = NearestNeighbors(
            n_neighbors=n_neighbors_arr.max() + 1,
            algorithm="brute",
            metric=metric,
        )
        model_pert.fit(cuda_pert)
        _, indices_pert = model_pert.kneighbors(cuda_pert)
        nn_indices_pert = indices_pert.values.get()

        # Inner progress bar for neighbor counts
        for n_neighbors in tqdm(
            n_neighbors_arr, desc="Calculating Jaccard Index for different k"
        ):
            jacc = calculate_jaccard_index_numba(
                nn_indices_gt[:, :n_neighbors],
                nn_indices_pert[:, :n_neighbors],
            )
            results.append(
                {
                    "k": n_neighbors,
                    "perturbation": f1_val,
                    "scaling": scale,
                    "variance": log1p,
                    "JKNN": jacc,
                    "metric": metric,
                    "mask_id": mask_id,
                }
            )

    # Save results
    jacc_df = pd.DataFrame(results)
    jacc_df.to_feather(f"results/JKNN/knn_jacc_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

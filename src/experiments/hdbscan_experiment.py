import pandas as pd
from itertools import product
from utils import chunk_list

import cudf
import cupy as cp
from cuml.cluster import HDBSCAN

import argparse
from tqdm import tqdm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.metrics import adjusted_mutual_info_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    pert_strength_lst = [70, 80, 90, 100]
    min_samples_lst = [3, 5, 7, 10]
    min_cluster_size_lst = [10, 20, 30, 40, 50]
    number_perturbations = 20
    results = []

    full_lst = list(
        product(
            pert_strength_lst,
            range(number_perturbations),
            min_samples_lst,
            min_cluster_size_lst,
        )
    )
    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    # Progress bar for the local schedule
    for f1_val, mask_id, min_samples, min_cluster_size in tqdm(
        local_schedule, desc="Processing local schedule"
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

        cuda_gt[channels] = cp.log1p(cuda_gt[channels].values)
        cuda_pert[channels] = cp.log1p(cuda_pert[channels].values)

        random_state = 42

        # hdbscan = refHDBSCAN(min_samples=5, min_cluster_size=30, core_dist_n_jobs=-1)
        # HDBSCAN(*, min_cluster_size=5, min_samples=None

        gt_hdb = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            core_dist_n_jobs=-1,
            random_state=random_state,
            output_type="numpy",
        )

        pert_hdb = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            core_dist_n_jobs=-1,
            random_state=random_state,
            output_type="numpy",
        )

        clst_gt = gt_hdb.fit_predict(cuda_gt)
        clst_pert = pert_hdb.fit_predict(cuda_pert)

        arand_score = adjusted_rand_score(clst_gt, clst_pert)
        ami_score = adjusted_mutual_info_score(clst_gt, clst_pert)

        # Todo: Adapt NID for negative values!

        results.append(
            {
                "min_samples": min_samples,
                "min_cluster_size": min_cluster_size,
                "perturbation": f1_val,
                "gt_labels": clst_gt,
                "pert_labels": clst_pert,
                "arand": arand_score,
                "ami": ami_score,
                "mask_id": mask_id,
            }
        )

    # Save results
    kmeans_df = pd.DataFrame(results)
    kmeans_df.to_feather(f"results/HDBSCAN/hdb_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

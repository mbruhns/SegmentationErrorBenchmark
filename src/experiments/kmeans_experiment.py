import pandas as pd
from itertools import product
from utils import chunk_list

import cudf
import cupy as cp
from cuml.cluster import KMeans

import argparse
from tqdm import tqdm
import numpy as np

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics.cluster import adjusted_rand_score

from informations_measures import normalized_information_distance
from sklearn.metrics import adjusted_mutual_info_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    pert_strength_lst = [70, 80, 90, 100]
    n_clusters_lst = [3, 5, 10, 15, 20, 30]
    number_perturbations = 20
    results = []

    full_lst = list(
        product(
            pert_strength_lst,
            range(number_perturbations),
            ("log1p_true", "log1p_false"),
        )
    )
    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    for f1_val, mask_id, log1p in tqdm(
        local_schedule, desc="Processing local schedule"
    ):
        df_gt = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet"
        )
        df_pert = pd.read_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{mask_id}_qc.parquet"
        )

        """
        shared_indices = set(df_gt.Label.to_list()).intersection(df_pert.Label.to_list())
        df_gt = df_gt[df_gt.Label.isin(shared_indices)]
        df_pert = df_pert[df_pert.Label.isin(shared_indices)]
        """

        # For sure not needed, but just to be sure.
        df_gt = df_gt.sort_values("Label")
        df_pert = df_pert.sort_values("Label")

        channels = df_gt.columns[:-4].drop(labels="HOECHST 2")

        cuda_gt = cudf.from_dataframe(df_gt[channels], allow_copy=True)
        cuda_pert = cudf.from_dataframe(df_pert[channels], allow_copy=True)

        if log1p == "log1p_true":
            cuda_gt[channels] = cp.log1p(cuda_gt[channels].values)
            cuda_pert[channels] = cp.log1p(cuda_pert[channels].values)

        cuda_gt[channels].sample(n_clusters_lst[-1], random_state=42)

        max_iter = 1_000
        oversampling_factor = 3.0
        metric = "euclidean"
        # random_state = 42

        for n_clusters in tqdm(
            n_clusters_lst, desc="Clustering for different number of clusters."
        ):
            label_lst_gt = []
            label_lst_pert = []

            for random_state in range(5):
                gt_kmeans = KMeans(
                    n_clusters=n_clusters,
                    output_type="numpy",
                    max_iter=max_iter,
                    # init=initial_centroids.values[:n_clusters],
                    oversampling_factor=oversampling_factor,
                    random_state=random_state,
                )

                pert_kmeans = KMeans(
                    n_clusters=n_clusters,
                    output_type="numpy",
                    max_iter=max_iter,
                    # init=initial_centroids.values[:n_clusters],
                    oversampling_factor=oversampling_factor,
                    random_state=100 - random_state,
                )
                clst_gt = gt_kmeans.fit_predict(cuda_gt)
                clst_pert = pert_kmeans.fit_predict(cuda_pert)
                label_lst_gt.append(clst_gt)
                label_lst_pert.append(clst_pert)

            ari_scores = []
            nid_scores = []
            ami_scores = []
            for a, b in product(label_lst_gt, label_lst_pert):
                ari_scores.append(adjusted_rand_score(a, b))
                nid_scores.append(normalized_information_distance(a, b))
                ami_scores.append(adjusted_mutual_info_score(a, b))

            arand_score = np.max(ari_scores)
            nid_score = np.min(nid_scores)
            ami_score = np.max(ami_scores)

            score_gt = gt_kmeans.score(cuda_gt)
            score_pert = gt_kmeans.score(cuda_pert)

            # arand_score = adjusted_rand_score(clst_gt, clst_pert)
            # ami_score = adjusted_mutual_info_score(clst_gt, clst_pert)
            # nid_score = normalized_information_distance(clst_gt, clst_pert)

            results.append(
                {
                    "k": n_clusters,
                    "perturbation": f1_val,
                    "variance": log1p,
                    "arand": arand_score,
                    "ami": ami_score,
                    "nid": nid_score,
                    "metric": metric,
                    "mask_id": mask_id,
                    "score_gt": score_gt,
                    "score_pert": score_pert,
                }
            )

    # Save results
    kmeans_df = pd.DataFrame(results)
    kmeans_df.to_feather(f"results/KMeans/kmeans_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

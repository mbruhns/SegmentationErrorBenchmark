import pandas as pd
import numpy as np
from itertools import product
from utils import chunk_list

import cudf
import cupy as cp
from cuml.neighbors import NearestNeighbors

import argparse
from tqdm import tqdm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import igraph as ig
import random
from sklearn.metrics.cluster import adjusted_rand_score
from contextlib import contextmanager
from sklearn.utils import check_random_state
from cuml.manifold.umap import fuzzy_simplicial_set

from informations_measures import normalized_information_distance
from sklearn.metrics import adjusted_mutual_info_score


class RNGIgraph:
    """
    Random number generator for ipgraph so global seed is not changed.
    See :func:`igraph.set_random_number_generator` for the requirements.
    """

    def __init__(self, random_state: int = 0) -> None:
        self._rng = check_random_state(random_state)

    def __getattr__(self, attr: str):
        return getattr(self._rng, "normal" if attr == "gauss" else attr)


@contextmanager
def set_igraph_random_state(random_state: int):
    import igraph

    rng = RNGIgraph(random_state)
    try:
        igraph.set_random_number_generator(rng)
        yield None
    finally:
        igraph.set_random_number_generator(random)


def leiden_igraph_fuzzy(
    X_cudf, n_neighbors, metric="euclidean", resolution=1.0, random_state=0
):
    X = X_cudf.values.copy()
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute")
    model.fit(X)
    knn_dist, knn_indices = model.kneighbors(X)

    X_conn = cp.empty((X.shape[0], 1), dtype=np.float32)

    connectivities = fuzzy_simplicial_set(
        X_conn,
        n_neighbors=n_neighbors,
        random_state=42,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dist,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )

    source_array = connectivities.row.get()
    destination_array = connectivities.col.get()
    weight_array = connectivities.data.get().astype(np.float64)

    n_vertices = X.shape[0]
    edges = list(zip(source_array, destination_array))
    g = ig.Graph(n_vertices, edges)
    g["weight"] = weight_array

    with set_igraph_random_state(random_state):
        parts = g.community_leiden(
            objective_function="modularity", weights=weight_array
        )

    return parts.membership


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    pert_strength_lst = [70, 80, 90, 100]
    n_neighbors_lst = [10, 50, 75, 100]
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

    metric = "euclidean"

    # Progress bar for the local schedule
    for f1_val, mask_id, log1p in tqdm(
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

        if log1p == "log1p_true":
            cuda_gt[channels] = cp.log1p(cuda_gt[channels].values)
            cuda_pert[channels] = cp.log1p(cuda_pert[channels].values)

        for n_neighbors in tqdm(
            n_neighbors_lst, desc="Clustering for different neighborhoods."
        ):
            label_lst_gt = []
            label_lst_pert = []

            # TODO!
            for random_state in range(5):
                clst_gt = leiden_igraph_fuzzy(
                    X_cudf=cuda_gt, n_neighbors=n_neighbors, random_state=random_state
                )
                clst_pert = leiden_igraph_fuzzy(
                    X_cudf=cuda_pert,
                    n_neighbors=n_neighbors,
                    random_state=100 - random_state,
                )

                label_lst_gt.append(clst_gt)
                label_lst_pert.append(clst_pert)

                # number_clusters_gt = np.max(clst_gt) + 1
                # number_clusters_pert = np.max(clst_pert) + 1

            ari_scores = []
            nid_scores = []
            ami_scores = []

            label_combinations = list(product(label_lst_gt, label_lst_pert))

            for a, b in label_combinations:
                ari_scores.append(adjusted_rand_score(a, b))
                nid_scores.append(normalized_information_distance(a, b))
                ami_scores.append(adjusted_mutual_info_score(a, b))

            arand_score = np.max(ari_scores)
            nid_score = np.min(nid_scores)
            ami_score = np.max(ami_scores)

            # Get the corresponding number of clusters for the maximum scores
            max_gt_labels, max_pert_labels = label_combinations[np.argmax(arand_score)]

            number_clusters_gt = np.max(max_gt_labels) + 1
            number_clusters_pert = np.max(max_pert_labels) + 1

            results.append(
                {
                    "k": n_neighbors,
                    "perturbation": f1_val,
                    "variance": log1p,
                    "arand": arand_score,
                    "ami": ami_score,
                    "nid": nid_score,
                    "n_gt": number_clusters_gt,
                    "n_pert": number_clusters_pert,
                    "metric": metric,
                    "mask_id": mask_id,
                }
            )

    # Save results
    leiden_df = pd.DataFrame(results)
    leiden_df.to_feather(f"results/Leiden/leiden_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

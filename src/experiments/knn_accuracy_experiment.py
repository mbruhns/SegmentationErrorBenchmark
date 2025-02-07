from itertools import product
from utils import chunk_list

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import argparse
from tqdm import tqdm
from gating import phenotyping, annotate_cell_types, select_phenotype

import cudf
from cuml.neighbors import NearestNeighbors

from scipy.stats import mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    f1_val_lst = [70, 80, 90, 100]
    number_perturbations = 20
    n_neighbors_arr = np.array([10, 20, 30, 40, 50, 75, 100, 250])
    metric = "euclidean"
    results = []

    celltypes = [
        "Immune cell",
        "Parenchymal cell",
        "Unknown",
        "Hepatocytes/Tumor",
        "Biliary epithelial cells",
        "Vascular endothelial cells",
        "LSEC",
        "T cell",
        "Treg",
        "MAIT cell",
        "CD8+ T cell",
        "CD4+ T cell",
        "B cell",
        "NK cell",
        "Neutrophil Granulocyte",
        "Kupffer cell",
        "M2 Macrophage",
        "Dendritic cell",
    ]

    full_lst = list(product(f1_val_lst, range(number_perturbations)))

    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    # Progress bar for the local schedule
    for f1_val, mask_id in tqdm(local_schedule, desc="Processing local schedule"):
        df_gt_raw = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet"
        )
        df_pert_raw = pd.read_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{mask_id}_qc.parquet"
        )

        channels = df_gt_raw.columns[:-4].drop(labels="HOECHST 2")

        df_gt = np.log1p(df_gt_raw[channels])
        df_gt = phenotyping(df_gt)
        df_gt = annotate_cell_types(df_gt)
        assignments_gt = select_phenotype(df_gt[celltypes])

        le = LabelEncoder().fit(assignments_gt.values)

        df_pert = np.log1p(df_pert_raw[channels])
        df_pert = phenotyping(df_pert)
        df_pert = annotate_cell_types(df_pert)
        assignments_pert = select_phenotype(df_pert[celltypes])

        y_true = le.transform(assignments_gt.values)
        y_pred = le.transform(assignments_pert.values)

        cuda_gt = cudf.from_dataframe(df_gt[channels], allow_copy=True)
        cuda_pert = cudf.from_dataframe(df_pert[channels], allow_copy=True)

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
            n_neighbors_arr, desc="Calculating KNN accuracy for different k"
        ):
            dist_true = mode(
                y_true[nn_indices_gt[:, :n_neighbors]], axis=1
            ).mode.flatten()
            dist_pert = mode(
                y_pred[nn_indices_pert[:, :n_neighbors]], axis=1
            ).mode.flatten()
            bal_acc = balanced_accuracy_score(y_true=dist_true, y_pred=dist_pert)

            results.append(
                {
                    "k": n_neighbors,
                    "perturbation": f1_val,
                    "KNN-Acc": bal_acc,
                    "metric": metric,
                    "mask_id": mask_id,
                }
            )
    result_df = pd.DataFrame(results)
    result_df.to_feather(f"results/KNN_ACC/knn_balacc_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

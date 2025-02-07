from itertools import product
from utils import chunk_list

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import argparse
from tqdm import tqdm

from gating import phenotyping, annotate_cell_types, select_phenotype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    f1_val_lst = [70, 80, 90, 100]
    number_perturbations = 20
    results = []
    results_sankey = []

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

    full_lst = list(
        product(
            f1_val_lst,
            range(number_perturbations),
        )
    )

    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    # Progress bar for the local schedule
    for f1_val, mask_id in tqdm(local_schedule, desc="Processing local schedule"):
        df_gt = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet"
        )
        df_pert = pd.read_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{mask_id}_qc.parquet"
        )

        channels = df_gt.columns[:-4].drop(labels="HOECHST 2")

        """
        foo = phenotyping_prob(df)
        foo = annotate_cell_types_score_product(foo)
        assignments = select_phenotype_dataframe_vectorized(foo[celltypes])
        """

        df_gt = np.log1p(df_gt[channels])
        df_gt = phenotyping(df_gt)
        df_gt = annotate_cell_types(df_gt)
        assignments_gt = select_phenotype(df_gt[celltypes])

        le = LabelEncoder().fit(assignments_gt.values)

        df_pert = np.log1p(df_pert[channels])
        df_pert = phenotyping(df_pert)
        df_pert = annotate_cell_types(df_pert)
        assignments_pert = select_phenotype(df_pert[celltypes])

        y_true = le.transform(assignments_gt.values)
        y_pred = le.transform(assignments_pert.values)

        bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred).round(3)

        result_df = pd.DataFrame(columns=["Perturbation", "BalancedAccuracy"])
        result_df["Perturbation"] = [f1_val]
        result_df["BalancedAccuracy"] = [bal_acc]

        results.append(result_df)

        df_sankey = pd.DataFrame(columns=["y_true", "y_pred", "pert"])
        df_sankey["y_true"] = assignments_gt.values
        df_sankey["y_pred"] = assignments_pert.values
        df_sankey["pert"] = f1_val
        results_sankey.append(df_sankey)

    df_pheno = pd.concat(results)
    df_sankey = pd.concat(results_sankey)

    df_pheno.to_feather(f"results/Phenotyping/phenotyping_partial_{current_id}.feather")
    df_sankey.to_feather(f"results/Phenotyping/sankey_partial_{current_id}.feather")


if __name__ == "__main__":
    main()

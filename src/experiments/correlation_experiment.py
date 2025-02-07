import pandas as pd
from itertools import product
from utils import chunk_list

import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    maximum_id = args.max
    current_id = args.current - 1

    pert_strength_lst = [70, 80, 90, 100]
    number_perturbations = 20
    results = []

    full_lst = list(product(pert_strength_lst, range(number_perturbations)))
    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    # Progress bar for the local schedule
    for f1_val, mask_id in tqdm(local_schedule, desc="Processing Local Schedule"):
        df_gt = pd.read_parquet(
            f"results/feature_tables_gt/LHCC35_GT_qc_{f1_val}.parquet"
        )
        df_pert = pd.read_parquet(
            f"results/feature_tables_{f1_val}_qc/LHCC35_pert{f1_val}_{mask_id}_qc.parquet"
        )

        channels = df_gt.columns[:-4].drop(labels="HOECHST 2")

        corr_gt = df_gt[channels].corr(method="pearson")
        corr_pert = df_pert[channels].corr(method="pearson")

        mae_val = (corr_gt - corr_pert).abs().mean().mean()
        print(mae_val)

        results.append({"perturbation": f1_val, "mask_id": mask_id, "MAE": mae_val})

    # Save results
    mae_df = pd.DataFrame(results)
    mae_df.to_feather(
        f"results/MarkerCorrelation/marker_corr_partial_{current_id}.feather"
    )


if __name__ == "__main__":
    main()

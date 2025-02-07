import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path


def main():
    # DataFrame to store results
    results_df_lst = []
    perturb_strengths = [70, 80, 90, 100]

    for perturb_strength in perturb_strengths:
        # Load ground truth once per perturbation strength
        gt_path = (
            Path("results/feature_tables_gt")
            / f"LHCC35_GT_qc_{perturb_strength}.parquet"
        )

        df_gt = pd.read_parquet(gt_path)

        perturbed_dir = Path("results") / f"feature_tables_{perturb_strength}_qc"

        # Iterate over all parquet files for the given perturbation strength
        for df_path in perturbed_dir.rglob("*.parquet"):
            df_pert = pd.read_parquet(df_path)

            for rmv in [True, False]:
                # Determine channels based on the "rmv" condition
                channels = df_gt.columns[:-4].tolist()
                if rmv and "HOECHST 2" in channels:
                    channels.remove("HOECHST 2")

                # Prepare data
                df_gt["Label"] = 0
                df_pert["Label"] = 1
                df_combined = pd.concat([df_gt, df_pert])

                X = df_combined[channels]
                y = df_combined["Label"]

                # Train the model and compute cross-validation scores
                clf = CatBoostClassifier(verbose=0, task_type="GPU", devices="0")
                scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
                mean_acc = scores.mean()
                med_acc = np.median(scores)

                # Append results to the results DataFrame
                results_df_lst.append(
                    {
                        "perturb_strength": perturb_strength,
                        "file": df_path.name,
                        "rmv": rmv,
                        "mean_accuracy": mean_acc,
                        "median_accuracy": med_acc,
                    }
                )

    pd.DataFrame(results_df_lst).to_parquet("results/c2st-results.parquet")


if __name__ == "__main__":
    main()

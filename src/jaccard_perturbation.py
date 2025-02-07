import numpy as np
from sklearn.metrics import jaccard_score
from pathlib import Path
import scipy as sc
from scipy.sparse import load_npz
import pandas as pd
from tqdm import tqdm


def main():
    sparse_array = sc.sparse.load_npz("data/LHCC35_Segmentation_GT_COO.npz")
    mask_gt = sparse_array.toarray().astype("int32")
    result_lst = []

    for f1_val in tqdm([70, 80, 90, 100]):
        perturbed_path = Path(f"results/perturbed_masks/f1_{f1_val}")

        for mask_path in perturbed_path.rglob("*.npz"):
            result = {}
            pert_mask = load_npz(mask_path).todense().astype("uint32", copy=False)
            jaccard_val = jaccard_score(
                mask_gt.ravel(), pert_mask.ravel(), average=None
            )

            result["jacc"] = jaccard_val
            result["jacc_min"] = np.min(jaccard_val)
            result["jacc_median"] = np.median(jaccard_val)
            result["TPR"] = (jaccard_val > 0.5).sum() / jaccard_val.size
            result["f1"] = f1_val
            result["mask_id"] = mask_path.stem.split("_")[-1]

            result_lst.append(result)

    result_df = pd.DataFrame(result_lst)
    result_df.to_csv("results/jaccard_scores_perturbations.csv", index=False)


if __name__ == "__main__":
    main()

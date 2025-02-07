import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from skimage.measure import regionprops_table
from itertools import product
import argparse
from utils import chunk_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    # Slurm has 1-indexing, but this is only important for the current ID.
    maximum_id = args.max
    current_id = args.current - 1

    properties = ["intensity_mean", "centroid", "area", "label"]
    pert_strength_lst = [70, 80, 90, 100]
    number_perturbations = 20

    full_data = np.load("data/full_image_stack.npz")
    channel_lst = full_data["channels"]
    img_arr = full_data["image"]

    full_lst = list(product(pert_strength_lst, range(number_perturbations)))
    global_schedule = list(chunk_list(full_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    for f1_val, mask_id in local_schedule:
        mask_path = (
            f"results/perturbed_masks/f1_{f1_val}/LHCC35_pert{f1_val}_{mask_id}.npz"
        )

        pert_mask = load_npz(mask_path).todense().astype("uint32", copy=False)

        # Setting the index explicitly is important to get the same cell ids over all sets!
        df = pd.DataFrame(
            regionprops_table(
                label_image=pert_mask, intensity_image=img_arr, properties=properties
            ),
            index=np.unique(pert_mask)[1:],
        )

        col_rename_dict = dict(zip(df.columns[:-1], channel_lst))
        col_rename_dict["area"] = "Area"
        col_rename_dict["label"] = "Label"
        col_rename_dict["centroid-0"] = "XMean"
        col_rename_dict["centroid-1"] = "YMean"
        df.rename(mapper=col_rename_dict, axis=1, inplace=True)

        df.to_parquet(
            f"results/feature_tables_{f1_val}/LHCC35_pert{f1_val}_{mask_id}.parquet"
        )

        del pert_mask
        del df


if __name__ == "__main__":
    main()

import argparse
from itertools import product, islice

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numba import njit, prange
from numba import boolean as nb_boolean
from scipy.sparse import csr_array
import scipy as sc
from scipy import stats
from skimage.morphology import binary_opening, octagon
from tqdm import tqdm
from sklearn.metrics import jaccard_score

from numba_f1 import f1_score
import csv


from skimage.segmentation import clear_border, relabel_sequential


def chunk_list(data, num_buckets):
    avg_chunk_size = len(data) // num_buckets
    remainder = len(data) % num_buckets
    iterator = iter(data)

    for _ in range(num_buckets):
        chunk_size = avg_chunk_size + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        chunk = list(islice(iterator, chunk_size))
        if chunk:
            yield chunk


@njit
def count_ids(mask, max_val):
    id_counts = np.zeros(max_val + 1, dtype=np.int64)

    # Count the occurrences of each ID
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            id_ = mask[i, j]
            id_counts[id_] += 1

    return id_counts


def area_change(mask_gt, mask_pert):
    max_val = mask_gt.max()
    counts_gt = count_ids(mask=mask_gt, max_val=max_val)
    counts_pert = count_ids(mask=mask_pert, max_val=max_val)

    # Todo: I really hope this removes the background comparison!
    idx = np.nonzero(counts_pert)[1:]
    return counts_pert[idx] / counts_gt[idx]


def construct_index_lookup(image, device):
    csr_matrix = csr_array(image)
    row_indices, col_indices = csr_matrix.nonzero()

    sorted_indices = np.argsort(csr_matrix.data)
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]

    shifted_data = np.roll(csr_matrix.data[sorted_indices], 1)

    unique_value_splits = (
        np.where((csr_matrix.data[sorted_indices] != shifted_data)[1:])[0] + 1
    )
    grouped_row_indices = np.split(sorted_row_indices, unique_value_splits)
    grouped_col_indices = np.split(sorted_col_indices, unique_value_splits)

    index_dict = {
        value: (torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device))
        for value, (rows, cols) in enumerate(
            zip(grouped_row_indices, grouped_col_indices), start=1
        )
    }

    return index_dict


def padded_perturb_torch(
    masks,
    device,
    index_dictionary=None,
    scale=(0.9, 1.25),
    translate_px=(-5, 5),
    rotate=(-5, 5),
    shear=(-10, 10),
    pad_size=20,
    footprint_size=3,
):
    transform = A.Compose(
        [
            A.Affine(
                scale=scale,
                translate_px=translate_px,
                rotate=rotate,
                shear=shear,
                interpolation=1,
                mask_interpolation=0,
                cval=0,
                cval_mask=0,
                mode=0,
                fit_output=False,
                keep_ratio=False,
                rotate_method="largest_box",
                always_apply=True,
                p=1,
            ),
            ToTensorV2(),
        ]
    )

    masks_tensor = torch.nn.functional.pad(
        torch.from_numpy(masks),
        (pad_size, pad_size, pad_size, pad_size),
        mode="constant",
    ).to(device)
    perturbed_masks_tensor = torch.zeros_like(masks_tensor)
    rng = np.random.default_rng(seed=None)
    mask_id_order = rng.permutation(np.arange(1, masks.max() + 1))

    if index_dictionary is None:
        index_dictionary = construct_index_lookup(
            masks_tensor.cpu().numpy(), device=device
        )

    footprint = octagon(footprint_size, footprint_size)
    for mask_id in tqdm(mask_id_order):
        x_coord, y_coord = index_dictionary[mask_id]

        x_min, x_max = x_coord.min(), x_coord.max() + 1
        y_min, y_max = y_coord.min(), y_coord.max() + 1

        x_min -= pad_size
        x_max += pad_size
        y_min -= pad_size
        y_max += pad_size

        # Set all pixels to zero except the mask_id ones
        temp = (masks_tensor[x_min:x_max, y_min:y_max] == mask_id).float()

        # Move tensor to CPU for Albumentations
        temp = np.array(temp.cpu())  # Move tensor to CPU for Albumentations
        trans_img = transform(image=temp)["image"]  # .to(device)

        trans_img = binary_opening(trans_img[0], footprint=footprint)
        trans_img = torch.from_numpy(trans_img).to(device)

        trans_img = torch.where(trans_img == 1, mask_id, torch.tensor(0, device=device))

        mask_to_update = (trans_img != 0).to(device)

        # Update only the masked areas
        perturbed_masks_tensor[x_min:x_max, y_min:y_max] = torch.where(
            mask_to_update, trans_img, perturbed_masks_tensor[x_min:x_max, y_min:y_max]
        )

    perturbed_masks_tensor = perturbed_masks_tensor[
        pad_size:-pad_size, pad_size:-pad_size
    ]
    return perturbed_masks_tensor.cpu().numpy()


@njit()
def pad_numba(A, pad_size=1):
    # A = A.astype(np.uint32)
    arr = np.zeros(
        (A.shape[0] + 2 * pad_size, A.shape[1] + 2 * pad_size), dtype=A.dtype
    )
    arr[pad_size:-pad_size, pad_size:-pad_size] = A
    return arr


@njit()
def numba_setdiff(arr1, arr2):
    # Returns unique values of arr1 that are not in arr1.
    filtered = [element for element in arr1 if element not in arr2]
    return np.array(list(set(filtered)), dtype=arr1.dtype)


@njit
def nb_isin(a, b):
    out = np.zeros_like(a, dtype=nb_boolean)
    b = set(b)

    for index, x in np.ndenumerate(a):
        if x in b:
            out[index] = True
    return out


@njit
def cell_adjacency_numba(label_mask, mode="star"):
    new_lmod = label_mask.copy()
    row_num, col_num = new_lmod.shape
    cell_num = np.max(new_lmod)
    adjacency_matrix = np.zeros((cell_num, cell_num), dtype="int8")

    new_lmod_border = pad_numba(new_lmod, pad_size=1)
    nn_arr = np.zeros_like(label_mask, dtype="int8")

    # start looping the mask and produce the cell-cell contact matrix
    for i in range(1, row_num + 1):
        for j in range(1, col_num + 1):
            cur_val = new_lmod_border[i, j]
            if cur_val != 0:
                temp_matrix = np.array(
                    [
                        new_lmod_border[i - 1, j],
                        new_lmod_border[i + 1, j],
                        new_lmod_border[i, j - 1],
                        new_lmod_border[i, j + 1],
                    ]
                )

                unique_nn = np.unique(temp_matrix)

                # Return the unique values in ar1 that are not in ar2.
                if unique_nn.size > 1:
                    ids = numba_setdiff(arr1=unique_nn, arr2=[0, cur_val])

                    if ids.size > 0:
                        nn_arr[i - 1, j - 1] = 1
                        adjacency_matrix[cur_val - 1, ids - 1] = 1

    return adjacency_matrix, nn_arr


@njit(parallel=True, fastmath=True)
def separation_border_inplace(image):
    x, y = image.shape

    for i in prange(1, x - 1):
        for j in range(1, y - 1):
            cur_val = image[i, j]
            if cur_val == 0:
                continue

            # Define neighbors
            neighbor_top = image[i - 1, j]
            neighbor_bottom = image[i + 1, j]
            neighbor_left = image[i, j - 1]
            neighbor_right = image[i, j + 1]
            neighbor_top_left = image[i - 1, j - 1]
            neighbor_top_right = image[i - 1, j + 1]
            neighbor_bottom_left = image[i + 1, j - 1]
            neighbor_bottom_right = image[i + 1, j + 1]

            # Check all 8 neighbors
            if (
                (neighbor_top != 0 and neighbor_top != cur_val)
                or (neighbor_bottom != 0 and neighbor_bottom != cur_val)
                or (neighbor_left != 0 and neighbor_left != cur_val)
                or (neighbor_right != 0 and neighbor_right != cur_val)
                or (neighbor_top_left != 0 and neighbor_top_left != cur_val)
                or (neighbor_top_right != 0 and neighbor_top_right != cur_val)
                or (neighbor_bottom_left != 0 and neighbor_bottom_left != cur_val)
                or (neighbor_bottom_right != 0 and neighbor_bottom_right != cur_val)
            ):
                image[i, j] = 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max", help="Encodes the number of jobs running.", type=int)
    parser.add_argument("--current", help="Identity of job.", type=int)
    args = parser.parse_args()

    # Function warmup
    _ = pad_numba(A=np.ones((5, 5), dtype=np.uint32), pad_size=3)
    _ = numba_setdiff(arr1=np.array([1, 2, 3, 4, 5]), arr2=np.array([0, 1]))
    _ = cell_adjacency_numba(
        label_mask=np.random.randint(5, size=(10, 10), dtype=np.uint32)
    )
    # _ = separation_border(mask=np.random.randint(5, size=(10, 10), dtype=np.uint32))
    _ = f1_score(np.array([0, 1, 0, 1, 1]), np.array([1, 1, 0, 1, 0]))
    _ = count_ids(mask=np.random.randint(0, 5, (10, 10)), max_val=4)

    # scale_min_lst = np.arange(0.80, 1.01, 0.01)
    # scale_max_lst = np.arange(1, 1.16, 0.01)
    # translate_lst = [0, 1, 2, 3, 4, 5, 6]
    # rotate_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # shear_lst = [0, 1, 2, 3, 4, 5, 6, 7]

    # scale_min_lst = np.arange(0.80, 1.01, 0.01)
    # scale_max_lst = np.arange(1, 1.16, 0.01)
    # translate_lst = [1, 2, 3, 4, 5, 6]
    # rotate_lst = [1, 2, 3, 4, 5, 6, 7, 8]
    # shear_lst = [1, 2, 3, 4, 5, 6, 7]

    # scale_min_lst = np.arange(0.80, 1.01, 0.01)
    # scale_max_lst = np.arange(1, 1.16, 0.02)
    # translate_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # rotate_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # shear_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    scale_min_lst = np.arange(0.80, 1.01, 0.01)
    scale_max_lst = np.arange(1.01, 1.16, 0.02)  # np.arange(1, 1.16, 0.02)
    translate_lst = [6, 7, 8, 9]
    rotate_lst = [6, 7, 8, 9]
    shear_lst = [6, 7, 8, 9]

    pad_size = 20
    footprint_size = 3

    # Slurm has 1-indexing, but this is only important for the current ID.
    maximum_id = args.max
    current_id = args.current - 1

    csv_file = f"results/jaccard_perturbation/jaccard_output_{current_id}_strong.csv"

    # Define the header
    header = [
        "f1",
        "min_change",
        "max_change",
        "med_change",
        "mean_change",
        "std_change",
        "mad_change",
        "skew_change",
        "kurt_change",
        "F1_50",
        "F1_60",
        "F1_70",
        "F1_80",
        "F1_90",
        "scale_min",
        "scale_max",
        "translate_min",
        "translate_max",
        "rotate_min",
        "rotate_max",
        "shear_min",
        "shear_max",
        "pad_size",
        "footprint_size",
    ]

    # Initialize the CSV file with headers
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    config_lst = [
        {
            "scale": (sc_min, sc_max),
            "translate_px": (-tr, tr),
            "rotate": (-rot, rot),
            "shear": (-shear, shear),
            "pad_size": pad_size,
            "footprint_size": footprint_size,
        }
        for sc_min, sc_max, tr, rot, shear in product(
            scale_min_lst, scale_max_lst, translate_lst, rotate_lst, shear_lst
        )
    ]

    # global_schedule contains all lists of configurations. Each job will work on one list.
    global_schedule = list(chunk_list(data=config_lst, num_buckets=maximum_id))
    local_schedule = global_schedule[current_id]

    sparse_array = sc.sparse.load_npz("data/LHCC35_Segmentation_GT_COO.npz")
    mask_gt = sparse_array.toarray().astype("int32")

    # Todo: Define slice object and apply it to mask_gt to get a smaller subselection of the mask
    mask_slice = slice(3000, 16000)
    mask_gt = mask_gt[mask_slice, mask_slice].copy()

    mask_gt = clear_border(mask_gt)
    mask_gt, _, _ = relabel_sequential(mask_gt)

    bin_gt_mask = (mask_gt != 0).astype("uint8")

    for config_dct in local_schedule:
        perturbed_mask = padded_perturb_torch(
            masks=mask_gt,
            index_dictionary=None,
            device="cpu",
            **config_dct,
        )
        # perturbed_mask = separation_border(perturbed_mask)
        # print(perturbed_mask.sum())
        separation_border_inplace(perturbed_mask)

        bin_pert_mask = (perturbed_mask != 0).astype("uint8")

        f1 = f1_score(bin_gt_mask.ravel(), bin_pert_mask.ravel())
        jaccard_arr = jaccard_score(
            mask_gt.ravel(), perturbed_mask.ravel(), average=None
        )

        tp_50 = (jaccard_arr >= 0.5).sum()
        fn_50 = (jaccard_arr == 0).sum()
        fp_50 = (jaccard_arr < 0.5).sum() - fn_50
        f1_50 = 2 * tp_50 / (2 * tp_50 + fp_50 + fn_50)

        tp_60 = (jaccard_arr >= 0.6).sum()
        fn_60 = (jaccard_arr == 0).sum()
        fp_60 = (jaccard_arr < 0.6).sum() - fn_60
        f1_60 = 2 * tp_60 / (2 * tp_60 + fp_60 + fn_60)

        tp_70 = (jaccard_arr >= 0.7).sum()
        fn_70 = (jaccard_arr == 0).sum()
        fp_70 = (jaccard_arr < 0.7).sum() - fn_70
        f1_70 = 2 * tp_70 / (2 * tp_70 + fp_70 + fn_70)

        tp_80 = (jaccard_arr >= 0.8).sum()
        fn_80 = (jaccard_arr == 0).sum()
        fp_80 = (jaccard_arr < 0.8).sum() - fn_80
        f1_80 = 2 * tp_80 / (2 * tp_80 + fp_80 + fn_80)

        tp_90 = (jaccard_arr >= 0.9).sum()
        fn_90 = (jaccard_arr == 0).sum()
        fp_90 = (jaccard_arr < 0.9).sum() - fn_90
        f1_90 = 2 * tp_90 / (2 * tp_90 + fp_90 + fn_90)

        change_arr = area_change(mask_gt, perturbed_mask)
        del perturbed_mask
        min_change = np.min(change_arr)
        max_change = np.max(change_arr)
        med_change = np.median(change_arr)
        mean_change = np.mean(change_arr)
        std_change = np.std(change_arr)
        mad_change = stats.median_abs_deviation(change_arr)
        skew_change = stats.skew(change_arr)
        kurt_change = stats.kurtosis(change_arr)

        # Round each value to three decimal places
        rounded_values = [
            np.round(f1, 3),
            np.round(min_change, 3),
            np.round(max_change, 3),
            np.round(med_change, 3),
            np.round(mean_change, 3),
            np.round(std_change, 3),
            np.round(mad_change, 3),
            np.round(skew_change, 3),
            np.round(kurt_change, 3),
            ####
            np.round(f1_50, 3),
            np.round(f1_60, 3),
            np.round(f1_70, 3),
            np.round(f1_80, 3),
            np.round(f1_90, 3),
            ####
            config_dct["scale"][0],
            config_dct["scale"][1],
            config_dct["translate_px"][0],
            config_dct["translate_px"][1],
            config_dct["rotate"][0],
            config_dct["rotate"][1],
            config_dct["shear"][0],
            config_dct["shear"][1],
            config_dct["pad_size"],
            config_dct["footprint_size"],
        ]

        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(rounded_values)


if __name__ == "__main__":
    main()

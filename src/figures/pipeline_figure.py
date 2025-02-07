import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import load_npz


def main():
    plt.style.use("matplotlibrc.txt")

    mask_gt = (
        load_npz("data/LHCC35_Segmentation_GT_COO.npz")
        .todense()
        .astype("uint32", copy=False)
    )

    mask_70 = (
        load_npz("results/perturbed_masks/f1_70/LHCC35_pert70_0.npz")
        .todense()
        .astype("uint32", copy=False)
    )

    for selected_mask in [12_000, 12_001, 12_002, 12_003]:
        # selected_mask = 32_202  # 70_002, 41_902
        idx = np.where(mask_gt == selected_mask)

        min_x = idx[0].min() - 1
        min_y = idx[1].min() - 1

        max_x = idx[0].max() + 2
        max_y = idx[1].max() + 2

        gt_partial = mask_gt[min_x:max_x, min_y:max_y].copy()

        # Get unique elements and create a mapping from original values to new values
        unique_elements = np.unique(gt_partial)
        value_map = {val: idx for idx, val in enumerate(unique_elements)}

        # Replace each value in the original array with its corresponding value from the mapping
        gt_partial = np.vectorize(value_map.get)(gt_partial)

        gt_partial_bg = np.where(
            gt_partial == value_map[selected_mask], value_map[selected_mask], 0
        )

        idx_pert = np.where(mask_70 == selected_mask)

        min_x_pert = idx_pert[0].min() - 1
        min_y_pert = idx_pert[1].min() - 1

        max_x_pert = idx_pert[0].max() + 2
        max_y_pert = idx_pert[1].max() + 2

        pert_partial = mask_70[min_x_pert:max_x_pert, min_y_pert:max_y_pert].copy()
        pert_partial = np.where(pert_partial == selected_mask, selected_mask, 0)

        fig, ax = plt.subplots(1, 4, figsize=(4, 3.5))
        ax = ax.ravel()

        ax[0].imshow(np.where(mask_gt[:, :19152], 1, 0), cmap="gray")
        ax[0].set_title("Full image")

        rect = patches.Rectangle(
            (min_x, min_y),
            width=2500,
            height=2500,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax[0].add_patch(rect)

        ax[1].imshow(gt_partial, cmap="gray")
        ax[1].set_title("Selected \n mask")

        ax[2].imshow(gt_partial_bg, cmap="gray")
        ax[2].set_title("Binarized \n selection")

        # Shape difference hickhack
        ax[3].imshow(pert_partial[:, :-1], cmap="gray")
        ax[3].set_title("Perturbed \n mask")

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.axis("off")

        plt.savefig(
            f"figures/pipeline_example_{selected_mask}.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )


if __name__ == "__main__":
    main()

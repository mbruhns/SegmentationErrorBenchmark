import numpy as np
from scipy.sparse import load_npz
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


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
    mask_80 = (
        load_npz("results/perturbed_masks/f1_80/LHCC35_pert80_0.npz")
        .todense()
        .astype("uint32", copy=False)
    )

    mask_90 = (
        load_npz("results/perturbed_masks/f1_90/LHCC35_pert90_0.npz")
        .todense()
        .astype("uint32", copy=False)
    )
    mask_100 = (
        load_npz("results/perturbed_masks/f1_100/LHCC35_pert100_0.npz")
        .todense()
        .astype("uint32", copy=False)
    )

    # customPalette = sns.color_palette(["#648FFF", "#785EF0", "#DC267F", "#FE6100"])
    customPalette = sns.color_palette(["#E69F00", "#56B4E9", "#009E73", "#F0E442"])
    customPalette = sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    tile_width = 150
    x_start = 9200

    for x_start in [4200, 6000, 9200]:
        x_end = x_start + tile_width
        y_start = x_start
        y_end = x_end

        fig, ax = plt.subplots(2, 2, figsize=(3.7, 3.7))
        ax = ax.ravel()

        # Define colors and legend handles
        color_b1 = np.array([255, 255, 255]) / 255
        legend_handles = [mpatches.Patch(color=color_b1, label="Ground truth")]

        # Find ground truth boundaries
        b1 = find_boundaries(mask_gt[x_start:x_end, y_start:y_end], mode="inner")

        # Define perturbation levels and corresponding masks
        perturbations = [(70, mask_70), (80, mask_80), (90, mask_90), (100, mask_100)]

        # Iterate over perturbation levels and masks
        for ax_id, (pert_level, mask) in enumerate(perturbations):
            # Find boundaries and create RGB image
            b2 = find_boundaries(mask[x_start:x_end, y_start:y_end], mode="inner")
            rgb_image = np.zeros((b1.shape[0], b1.shape[1], 3), dtype=np.uint8)

            # Set colors for boundaries
            color_b2 = np.array(customPalette[ax_id]) * 255
            rgb_image[b1 == 1] = color_b1 * 255
            rgb_image[b2 == 1] = color_b2

            # Plotting
            ax[ax_id].imshow(rgb_image)
            ax[ax_id].axis("off")
            legend_handles.append(
                mpatches.Patch(color=color_b2 / 255, label=f"F1={pert_level}")
            )

        # Add legend
        # legend = fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.17, 0.55))
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncols=5,
        )
        for handle in legend.legend_handles:
            handle.set_edgecolor("gray")

        # plt.suptitle("Segmentation shape comparison", fontsize=20)
        fig.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05
        )
        plt.savefig(f"figures/perturb_shape_comparison_{x_start}.pdf", transparent=True)
        plt.close()


if __name__ == "__main__":
    main()

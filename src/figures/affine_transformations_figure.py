import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


def get_image():
    delta = 0.01
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    return Z


def do_plot(ax, Z, transform, trans_title, center_extent=False):
    extent = [-3, 3, -3, 3] if center_extent else [-2, 4, -3, 2]
    im = ax.imshow(
        Z,
        interpolation="none",
        origin="lower",
        extent=extent,
        clip_on=False,
        aspect="equal",
    )
    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # Draw a white rectangle showing the "intended" extent
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "w", transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Put a small title to distinguish them
    ax.set_title(trans_title, pad=4, fontsize=8)

    ax.set_xticklabels([])
    ax.set_yticklabels([])


def main():
    fig, axd = plt.subplot_mosaic(
        mosaic="...XX...;AABBCCDD",
        figsize=(6.6 / 2.54, 2),
        layout="tight",
        gridspec_kw={
            "width_ratios": [1] * 8,  # all 8 columns same width
            "height_ratios": [1, 1],  # top & bottom row same height
        },
    )

    # Generate your image
    Z = get_image()

    # Define your 5 transformations
    transform_lst = [
        (Affine2D().scale(1, 1), "Original", True),
        (Affine2D().translate(0.75, -1), "Translation", True),
        (Affine2D().scale(0.5, 0.5), "Scaling", True),
        (Affine2D().rotate_deg(30), "Rotation", True),
        (Affine2D().skew_deg(30, 15), "Shearing", True),
    ]

    # Apply transformations to each of the 5 axes in the mosaic
    axes = list(axd.values())
    for ax, (transform, label, center_it) in zip(axes, transform_lst):
        do_plot(ax, Z, transform, trans_title=label, center_extent=center_it)

    plt.subplots_adjust(wspace=2, hspace=0.5)
    plt.savefig("figures/affine_transformations.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

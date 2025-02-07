import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


def main():
    plt.style.use("matplotlibrc.txt")
    sns.color_palette(["#4477AA", "#EE6677", "#228833", "#CCBB44"])

    fig, axd = plt.subplot_mosaic(
        mosaic="AAABBB;AAABBB;CCDDEE", layout="constrained", figsize=(6, 6)
    )

    for label, ax in axd.items():
        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                ax.transAxes
                + ScaledTranslation(-65 / fig.dpi, 0 / fig.dpi, fig.dpi_scale_trans)
            ),
            fontsize=12,
            va="top",
            fontfamily="Arial",
            fontweight="bold",
        )

    image = plt.imread("figures/cell_shape_perturbation.png")
    axd["A"].imshow(image, aspect="equal")
    axd["A"].axis("off")

    plt.savefig("figures/panel02_mpl.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()

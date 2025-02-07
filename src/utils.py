from itertools import islice
from scipy.stats import median_abs_deviation
import numpy as np


# Define a custom function for MAD error bars with a scaling factor
def mad_errorbars(values, scale=1):
    median = np.median(values)
    mad = median_abs_deviation(values, scale=scale)
    lower = median - scale * mad
    upper = median + scale * mad
    return lower, upper


# Define a wrapper for Seaborn to pass the scale
def mad_errorbars_with_scale(scale):
    def func(values):
        return mad_errorbars(values, scale=scale)

    return func


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


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def force_aspect(ax, aspect=1.0):
    """
    Stolen from here: https://stackoverflow.com/a/7968690/4100721
    Sometimes works to produce plots with the desired aspect ratio.
    """
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


# -----------------------------------------------------------------------------


def plot_spectrogram_label(spectrogram, true_label, pred_label,
                           grayzones=None, vmin=None, vmax=None):
    """
    Take a spectrogram, the true label, and the predicted label, and combine
    them into a plot for manual inspection. If grayzones are provided, then
    these will be included in the plot.

    Args:
        spectrogram: 2D numpy array containing the spectrogram
        true_label: 1D numpy array containing the true labels for injections
        pred_label: 1D numpy array containing the predicted labels
        grayzones: 1D numpy array containing the parts that are ignored by
            the loss function during training
        vmin: Minimum value of the spectrogram
        vmax: Maximum valie of the spectrogram
    """

    # Fix the size of figure as a whole
    plt.gcf().set_size_inches(18, 6, forward=True)

    # Define a grid to fix the height ratios of the subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # Plot the spectrogram
    ax1.imshow(spectrogram, origin="lower", cmap="Greys_r",
               interpolation="none", vmin=vmin, vmax=vmax)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    force_aspect(ax1, aspect=3.5)

    # Plot the vector indicating where there was an injection
    ax2.plot(true_label, lw=2, c='C2', label='True')
    ax2.plot(pred_label, lw=2, c='C3', label='Pred')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')

    # Plot the gray zones
    start = None
    in_zone = False
    for i in range(len(grayzones)):
        if not in_zone:
            if grayzones[i] == 0:
                start = i
                in_zone = True
        if in_zone:
            if grayzones[i] == 1:
                ax2.axvspan(start, i, alpha=0.5, color='Orange')
                in_zone = False

    ax2.set_xlim(0, len(true_label))
    ax2.set_ylim(-0.2, 1.2)
    plt.tight_layout()
    plt.gcf().subplots_adjust(hspace=0.05)
    plt.show()


# -----------------------------------------------------------------------------


def create_weights(label, start_size=4, end_size=2):
    """
    Create the weights ('grayzones') for a given label.

    Args:
        label: A vector labeling the pixels that contain an injection
        start_size: Number of pixels to ignore at the start of an injection
        end_size: Number of pixels to ignore at the end of an injections

    Returns: A vector that is 0 for the pixels that should be ignored and 1
        for all other pixels.
    """

    a = np.logical_xor(label, np.roll(label, 1))
    b = np.cumsum(a) % 2

    if start_size == 0:
        c = np.zeros(label.shape)
    else:
        c = np.convolve(a * b, np.hstack((np.zeros(start_size - 1),
                                          np.ones(start_size))),
                        mode="same")

    if end_size == 0:
        d = np.zeros(label.shape)
    else:
        d = np.convolve(a * np.logical_not(b),
                        np.hstack((np.ones(end_size), np.zeros(end_size - 1))),
                        mode="same")

    return np.logical_not(np.logical_or(c, d)).astype('int')

import fnmatch
from os import path
import os
import sys
import numpy as np
import numba
from gala import evaluate as ev, imio, viz
from matplotlib import pyplot as plt
from skimage.util import regular_seeds
from skimage import morphology as morph
import argparse

if len(sys.argv) == 1:
    MY_PATH = input('Directory to search for hdf files?%n\
                    Press "enter" to escape.')
    if MY_PATH != '':
        MY_PATH = path.join(path.expanduser(MY_PATH))
        print(MY_PATH)
        os.chdir(MY_PATH)
        FULL_PATH = path.join(MY_PATH, 'research_project_files/Cremi_Data')
        for _, _, filenames in os.walk(FULL_PATH):
            for filename in fnmatch.fnmatchcase(str(filenames), '*.hdf'):
                if filename in '+':
                    pass
                RAW, GT = imio.read_cremi(filename, datasets=
                                          ['volumes/raw',
                                           'volumes/labels/neuron_ids'])
            break
else:
    try:
        RAW, GT = imio.read_cremi(sys.argv[1], datasets=
                                  ['volumes/raw', 'volumes/labels/neuron_ids'])
    except FileNotFoundError:
        print("File not found. \n")
        sys.exit(0)

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['font.size'] = 8
RAW = np.max(RAW) - RAW
RAW = RAW[RAW.shape[0]//2]
GT = GT[GT.shape[0]//2]
SEEDS = regular_seeds(RAW.shape, np.random.randint(1100, 2100))
AUTOMATED_SEG = morph.watershed(RAW, SEEDS, compactness=0.001)


def view_all(gt, automated_seg, num_elem=4, axis=None):
    """Generates an interactive figure highlighting the VI error.

    Parameters
    ----------
    gt: nd-array with shape M*N.
        This corresponds to the 'ground truth'.
    auto: nd-array with same shape as gt. This
        corresponds to the automated segmentation.
    num_elem: Int, optional.
        This parameter determines the number of comps
        shown upon click. Set to output '4' by default.

    Returns
    -------
    A panel with four images - the bottom right corresponds to the
    components that are the worst false merges in the automated
    segmentation that corresponds to the components clicked in
    the first window."""

    if gt.shape != automated_seg.shape:
        return "Input arrays are not of the same shape."
    elif (type(gt) or type(automated_seg)) != np.ndarray:
        return "Input arrays not of valid type."
    cont = ev.contingency_table(automated_seg, gt)
    ii1, err1, ii2, err2 = ev.sorted_vi_components(automated_seg, gt)
    idxs2 = np.argsort(ii2)
    err_unsorted = err2[idxs2]
    err_img = err_unsorted[automated_seg]
    idxs1 = np.argsort(ii1)
    err_unsorted = err1[idxs1]
    err_img_1 = err_unsorted[gt]
    plt.interactive = False
    if axis is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                               figsize=(9, 9))
        plt.setp(ax.flat, aspect=1.0, adjustable='box-forced')
    else:
        for i in range(len(axis)):
            ax = axis[i]
    ax[0, 0].imshow(RAW, cmap='gray')
    viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[0, 0])
    ax[0, 1].imshow(RAW, cmap='gray')
    axes_image_1 = viz.imshow_rand(err_img_1, alpha=0.4, axis=ax[0, 1])
    ax[1, 0].imshow(RAW, cmap='gray')
    viz.imshow_rand(err_img, alpha=0.4, axis=ax[1, 0])
    ax[1, 1].imshow(RAW, cmap='gray')
    axes_image = viz.imshow_rand(gt, alpha=0.4, axis=ax[1, 1])
    ax[0, 0].set_title("Automated seg: click to show worst splits.")
    ax[0, 1].set_title("Worst merge comps in gt, colored by VI error.")
    ax[1, 0].set_title("Ground truth: click to show worst merges.")
    ax[1, 1].set_title("Worst split comps in the gt, colored by VI error.")

    @numba.jit
    def drawer(seg, comps, limit=True):
        a_seg = np.zeros_like(seg.astype('float64'))
        factor = (seg.max() // num_elem)
        lim = 0.0
        for i, (j, k, z) in enumerate(comps):
            lim += k
            if z < 0.02:
                continue
            a_seg += (seg == j) * ((i + 1) * factor)
            if limit:
                if lim >= 0.95:
                    break
        return a_seg

    @numba.jit
    def _onpress(event):
        vint = np.vectorize(int)
        if event.inaxes == ax[1, 0]:
            if event.button != 1:
                return
            x, y = vint(event.xdata), vint(event.ydata)
            comps = ev.split_components(gt[y, x], cont, axis=1, num_elems=None)
            new_seg = drawer(automated_seg, comps)
            axes_image.set_array(new_seg)
            fig.canvas.draw()
        if event.inaxes == ax[0, 0]:
            if event.button != 1:
                return
            x, y = vint(event.xdata), vint(event.ydata)
            comps = ev.split_components(automated_seg[y, x], cont, axis=0,
                                        num_elems=None)
            new_seg_1 = drawer(gt, comps, limit=False)
            axes_image_1.set_array(new_seg_1)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _onpress)
    plt.ioff()
    plt.show()


view_all(GT, AUTOMATED_SEG)

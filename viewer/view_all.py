#!/usr/bin/python3
"""Script that visualises the split-VI error from an input hdf file.

Input file must contain two hdf groups - a raw group, and a group containing
labelled neuron ids.
"""
# Builtins
from os import path
import os
import re
import sys
# Libraries
import numpy as np
from numba import jit
from gala import evaluate as ev, imio, viz
from matplotlib import pyplot as plt
from skimage.segmentation import join_segmentations
from skimage.measure import label
from skimage.util import regular_seeds
from skimage import morphology as morph

import click
import prompt_toolkit

if len(sys.argv) == 1:
    MY_PATH = input("""Directory to search for hdf files? Press "enter" to"""
                    """ escape.\n""")
    if MY_PATH != '':
        MY_PATH = path.join(path.expanduser(MY_PATH))
        if not path.exists(MY_PATH):
            raise FileNotFoundError
        os.chdir(MY_PATH)
        FULL_PATH = path.join(MY_PATH, 'research_project_files/Cremi_Data')
        REGEX = re.compile(r'^(\w)+_(\w)+_(\d)+(\.hdf)$', flags=re.I | re.M)
        for _, _, filenames in os.walk(FULL_PATH):
            for filename in filenames:
                if re.search(REGEX, filename):
                    RAW, GT = imio.read_cremi(filename, datasets=["""volumes/raw""", """volumes/
                                         labels/neuron_ids"""])
            break
    else:
        print("Program exited. \n")
        sys.exit(0)
if len(sys.argv) == 2:
    try:
        RAW, GT = imio.read_cremi(sys.argv[1], datasets=
                                  ["""volumes/raw""",
                                   """volumes/labels/neuron_ids"""])
    except FileNotFoundError:
        print("File/s not found. \n")
        sys.exit(0)
RAW = 1 - RAW/np.max(RAW)
RAW = RAW[RAW.shape[0]//2]
GT = GT[GT.shape[0]//2]
GT = label(GT)
SEEDS = regular_seeds(RAW.shape, np.random.randint(300, 350))
AUTOMATED_SEG = morph.watershed(RAW, SEEDS, compactness=0.001)


def view_all_join(gt, automated_seg, num_elem=6, axis=None):
    """Generate an interactive figure highlighting the VI error.

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
    A window with six panels - the top middle image corresponds to the
    components that are the worst false merges in the automated
    segmentation, which share significant area with the clicked-upon segment.
    Likewise, the top middle image shows the worst false splits.
    """
    if gt.shape != automated_seg.shape:
        return "Input arrays are not of the same shape."
    elif (type(gt) or type(automated_seg)) != np.ndarray:
        return "Input arrays not of valid type."
    vint = np.vectorize(int)
    # Compute the join seg of the automatic seg and the ground truth.
    joint_seg = join_segmentations(automated_seg, gt)
    # Contingency table for merges
    cont_table_m = ev.contingency_table(automated_seg, joint_seg)
    # Contingency table for splits
    cont_table_s = ev.contingency_table(joint_seg, gt)
    # Sort the VI according to the largest false merge components.
    merge_idxs_m, merge_errs_m = ev.sorted_vi_components(joint_seg, automated_seg)[0:2] #merges
    #Sort the VI according to the largest false split components.
    split_idxs_s, split_errs_s = ev.sorted_vi_components(joint_seg, gt)[0:2] #split
    #Find the indices of these largest false merge components, and largest false splits, in descending order.
    merge_idxs_sorted, split_idxs_sorted = np.argsort(merge_idxs_m), np.argsort(split_idxs_s)
    #Sort the errors according to the indices.
    merge_unsorted, split_unsorted = merge_errs_m[merge_idxs_sorted], split_errs_s[split_idxs_sorted]
    # Color both the seg and gt according to the intensity of the split VI error.
    merge_err_img, split_err_img = merge_unsorted[automated_seg], split_unsorted[gt]


    if axis is None:
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        plt.setp(ax.flat, adjustable='box-forced')
    else:
        fig, ax = plt.subplots(nrows=len(axis)//2, ncols=len(axis)//2,
                               sharex=True, sharey=True)
        for i in range(len(axis)//2):
            ax[0, i] = ax[i]
        for i in range(0, (len(axis)//2)):
            ax[1, i] = ax[i+2]

    ax[0, 0].imshow(RAW)
    viz.imshow_magma(merge_err_img, alpha=0.4, axis=ax[0, 0])
    ax[0, 1].imshow(RAW)
    axes_image_1 = viz.imshow_rand(joint_seg, alpha=0.4, axis=ax[0, 1])
    ax[0, 2].imshow(RAW)
    viz.imshow_rand(gt, alpha=0.4, axis=ax[0, 2])
    ax[1, 0].imshow(RAW)
    viz.imshow_magma(split_err_img, alpha=0.4, axis=ax[1, 0])
    ax[1, 1].imshow(RAW)
    axes_image = viz.imshow_rand(joint_seg, alpha=0.4, axis=ax[1, 1])
    ax[1, 2].imshow(RAW)
    viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[1, 2])
    ax[0, 0].set_title("Worst merge comps colored by VI error: click to show them on second panel.")
    ax[0, 1].set_title("Worst merge comps.")
    ax[0, 2].set_title("Ground Truth.")
    ax[1, 0].set_title("Worst split comps colored by VI error: click to show them on second panel.")
    ax[1, 1].set_title("Worst split comps.")
    ax[1, 2].set_title("Automated Seg.")

    @jit
    def drawer(seg, comps, limit=True):
        """Dynamically redraw the worst split/merge comps."""
        a_seg = np.zeros_like(seg.astype('float64'))
        factor = (seg.max() // num_elem)
        lim = 0.0
        for i, (j, k, z) in enumerate(comps):
            lim += k
            if z < 0.01:
                continue
            a_seg += (seg == j) * ((i + 1) * factor)
            if limit:
                if lim >= 0.98:
                    break
        return a_seg

    @jit
    def _onpress(event):
        """Matplotlib 'onpress' event handler."""

        if not (event.inaxes == ax[1, 0] or event.inaxes == ax[0, 0]
                or event.inaxes == ax[0, 2] or event.inaxes == ax[1, 2]):
            fig.text(0.5, 0.5, s="Must click on left or right axes to show comps!",
                     ha="center")
            fig.canvas.draw_idle()
        if event.inaxes == ax[0, 0] or event.inaxes == ax[0, 2]:
            if event.button != 1:
                return
            for txt in fig.texts:
                txt.set_visible(False)
            fig.canvas.draw()
            x, y = vint(event.xdata), vint(event.ydata)
            # Find the indices of the false merge bodies overlapping with the coordinates of the mouse click.
            worst_merge_comps_m = ev.split_components(automated_seg[y, x], num_elems=None, cont=cont_table_m, axis=0)
            new_seg_m = drawer(joint_seg, worst_merge_comps_m, limit=False)
            axes_image_1.set_array(new_seg_m)

            fig.canvas.draw()

        if event.inaxes == ax[1, 0] or event.inaxes == ax[1, 2]:
            if event.button != 1:
                return
            for txt in fig.texts:
                txt.set_visible(False)
            fig.canvas.draw()
            x, y = vint(event.xdata), vint(event.ydata)
            # Find the indices of the false split bodies overlapping with the coordinates of the mouse click.
            worst_split_comps_s = ev.split_components(gt[y, x], num_elems=None, cont=cont_table_s, axis=1)
            new_seg_s = drawer(joint_seg, worst_split_comps_s)
            axes_image.set_array(new_seg_s)

            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _onpress)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    view_all(GT, AUTOMATED_SEG)

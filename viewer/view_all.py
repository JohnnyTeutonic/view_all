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
SEEDS = regular_seeds(RAW.shape, np.random.randint(200, 225))
AUTOMATED_SEG = morph.watershed(RAW, SEEDS, compactness=0.001)


#@click.command
#@click.option('--raw', default='volumes/raw', help='raw hdf file.')
#@click.option('--label', default='volumes/labels/neuron_ids',
#              prompt='segmentation file', help='labeled hdf file.')
#@click.option('--input_file', prompt='what is the input file',
#              help='The input image file.')
def view_all(gt, automated_seg, num_elem=6, axis=None):
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
    A panel with four images - the bottom right corresponds to the
    components that are the worst false merges in the automated
    segmentation that corresponds to the components clicked in
    the first window.
    """
    if gt.shape != automated_seg.shape:
        return "Input arrays are not of the same shape."
    elif (type(gt) or type(automated_seg)) != np.ndarray:
        return "Input arrays not of valid type."
    vint = np.vectorize(int)
    cont = ev.contingency_table(automated_seg, gt)
    ii1, err1, ii2, err2 = ev.sorted_vi_components(automated_seg, gt)
    joint_seg = join_segmentations(automated_seg, gt)
    merge_idxs_m, merge_errs_m = ev.sorted_vi_components(joint_seg, automated_seg)[0:2]
    split_idxs_s, split_errs_s = ev.sorted_vi_components(joint_seg, gt)[0:2]
    idxs1, idxs2 = np.argsort(ii1), np.argsort(ii2)
    err_unsorted, err_unsorted_2 = err1[idxs1], err2[idxs2]
    err_img, err_img_1 = err_unsorted[gt], err_unsorted_2[automated_seg]
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
    viz.imshow_magma(err_img_1, alpha=0.4, axis=ax[0, 0])
    ax[0, 1].imshow(RAW)
    axes_image_1 = viz.imshow_rand(gt, alpha=0.4, axis=ax[0, 1])
    ax[0, 2].imshow(RAW)
    viz.imshow_rand(gt, alpha=0.4, axis=ax[0, 2])
    ax[1, 0].imshow(RAW)
    viz.imshow_magma(err_img, alpha=0.4, axis=ax[1, 0])
    ax[1, 1].imshow(RAW)
    axes_image = viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[1, 1])
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
            fig.text(0.5, 0.5, s="Must click on left axes to show comps!",
                     ha="center")
            fig.canvas.draw_idle()
        if event.inaxes == ax[1, 0] or event.inaxes == ax[1, 2]:
            if event.button != 1:
                return
            for txt in fig.texts:
                txt.set_visible(False)
            fig.canvas.draw()
            x, y = vint(event.xdata), vint(event.ydata)
            comps = ev.split_components(gt[y, x], cont, axis=1, num_elems=None)
            new_seg = drawer(automated_seg, comps)
            axes_image.set_array(new_seg)
            fig.canvas.draw()
        if event.inaxes == ax[0, 0] or event.inaxes == ax[0, 2]:
            if event.button != 1:
                return
            for txt in fig.texts:
                txt.set_visible(False)
            fig.canvas.draw()
            x, y = vint(event.xdata), vint(event.ydata)
            comps = ev.split_components(automated_seg[y, x], cont, axis=0,
                                        num_elems=None)
            new_seg_1 = drawer(gt, comps, limit=False)
            axes_image_1.set_array(new_seg_1)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _onpress)
    plt.ioff()
    plt.show()


#cont_table_m = ev.contingency_table(best_seg_bpm, joint_seg)
#worst_merge_comps_m = ev.split_components(merge_idxs_m[0], num_elems=10, cont=cont_table_m.T, axis=1)
#worst_merge_array_m = np.array(worst_merge_comps_m[0:3], dtype=np.int64)

#cont_table_s = ev.contingency_table(joint_seg, gt_raw_testing)
#worst_split_comps_s = ev.split_components(split_idxs_s[0], num_elems=10, cont=cont_table_s.T, axis=0)
#worst_split_array_s = np.array(worst_split_comps_s[0:3], dtype=np.int64)

if __name__ == '__main__':
    view_all(GT, AUTOMATED_SEG)

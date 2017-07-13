from gala import evaluate as ev, imio, viz
from matplotlib import pyplot as plt
import numpy as np
from skimage.util import regular_seeds
from skimage import morphology as morph, measure
from skimage.segmentation import mark_boundaries
import sys
import argparse


try:
    raw, gt = imio.read_cremi(sys.argv[1],  datasets=
                              ['volumes/raw', 'volumes/labels/neuron_ids'])
except FileNotFoundError:
    print("File not found. \n")
    sys.exit(0)

plt.rcParams['image.cmap'] = 'gray'
raw = np.max(raw) - raw
raw = raw[raw.shape[0]//2]
gt = gt[gt.shape[0]//2]
seeds = regular_seeds(raw.shape, np.random.randint(1100, 2100))
automated_seg = morph.watershed(raw, seeds, compactness=0.001)


def view_all(gt, automated_seg, num_elem=4):
    """Generate an interactive figure with four panels - showing level of
    split-VI error H(Y|X), H(X|Y), the worst false split comps, and worst
    false merge comps, respectively. Clicking on the first two panels
    results in highlighting these worst comps in the other two panels.
    Parameters

    gt: nd-array with shape M*N.
        This corresponds to the 'ground truth'.
    auto: nd-array with same shape as gt. This
        corresponds to the automated segmentation.
    num_elem: Int, optional.
        This parameter determines the number of comps
        shown upon click. Set to output '4' by default.
    returns:
        A panel with four images - the bottom right corresponds to the
        components that are the worst false splits in the automated
        segmentation that corresponds to the components clicked in
        the first window, and the top right shows the worst merges found in the
        ground truth relative to the automated seg.
    """
    # %matplotlib auto
    if gt.shape != automated_seg.shape:
        return "Input arrays are not of the same shape."
    elif (type(gt) or type(automated_seg)) != np.ndarray:
        return "Input arrays not of valid type."
    else:
        cont = ev.contingency_table(automated_seg, gt)
        ii1, err1, ii2, err2 = ev.sorted_vi_components(automated_seg, gt)[0:4]
        idxs = np.argsort(ii2)
        err_unsorted = err2[idxs]
        err_img = err_unsorted[automated_seg]
        idxs1 = np.argsort(ii1)
        err_unsorted1 = err1[idxs1]
        err_img1 = err_unsorted1[gt]
        plt.interactive = False
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                               figsize=(10, 10))
        plt.setp(ax.flat, aspect=1.0, adjustable='box-forced')
        ax[0, 0].imshow(raw)
        viz.imshow_magma(err_img1, alpha=0.4, axis=ax[0, 0])
        ax[0, 1].imshow(raw)
        axes_image_1 = viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[0, 1])
        ax[1, 0].imshow(raw)
        viz.imshow_magma(err_img, alpha=0.4, axis=ax[1, 0])
        ax[1, 1].imshow(raw)
        axes_image = viz.imshow_rand(gt, alpha=0.4, axis=ax[1, 1])
        ax[0, 0].set_title("Automated seg: Colored by level of VI error, \
                           H(X|Y). ")
        ax[0, 1].set_title("Worst merge comps found in the automated seg. ")
        ax[1, 0].set_title("Ground truth: Colored by level of VI error, \
                           H(Y|X).")
        ax[1, 1].set_title("Worst split comps found in the automated seg. ")

        def onpress_m(event):
            if event.button != 1:
                return
            x, y = int(event.xdata), int(event.ydata)
            comps = ev.split_components(automated_seg[y, x], cont, axis=0,
                                        num_elems=num_elem)
            new_seg = np.zeros_like(gt.astype('float64'))
            factor = (gt.max() // num_elem)
            lim = 0.0
            for i, (j, k, z) in enumerate(comps):
                lim += k
                if z < 0.02:
                    continue
                new_seg += (gt == j) * ((i + 1) * factor)
            axes_image_1.set_array(new_seg)
            fig.canvas.draw()

        def onpress_s(event):
            if event.button != 1:
                return
            x, y = int(event.xdata), int(event.ydata)
            comps = ev.split_components(gt[y, x], cont, axis=1, num_elems=None)
            new_seg = np.zeros_like(automated_seg)
            factor = (automated_seg.max() // num_elem)
            lim = 0.0
            for i, (j, k, z) in enumerate(comps):
                lim += k
                if z < 0.02:
                    continue
                new_seg += (automated_seg == j) * ((i + 1) * factor)
                if lim >= 0.95:
                    break
            axes_image.set_array(new_seg)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onpress_m)
        fig.canvas.mpl_connect('button_press_event', onpress_s)
        plt.ioff()
        plt.show()


view_all(gt, automated_seg)

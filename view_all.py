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
plt.rcParams['font.size'] = 8
raw = np.max(raw) - raw
raw = raw[raw.shape[0]//2]
gt = gt[gt.shape[0]//2]
seeds = regular_seeds(raw.shape, np.random.randint(1100, 2100))
automated_seg = morph.watershed(raw, seeds, compactness=0.001)


def view_all(gt, automated_seg, num_elem=4):
        """Generates a click-able image of the auto seg - upon click of
        the auto seg, shows the largest comps of the gt that corresponds
        to the worst false merges made in the automatic seg at the approx.
        same location of the click.
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
        components that are the worst false merges in the automated
        segmentation that corresponds to the components clicked in
        the first window.
        """
        %matplotlib auto
        if gt.shape != automated_seg.shape:
            return "Input arrays are not of the same shape."
        elif (type(gt) or type(automated_seg)) != np.ndarray:
            return "Input arrays not of valid type."
        else:
            cont = ev.contingency_table(automated_seg, gt)
            ii2, err2 = ev.sorted_vi_components(automated_seg, gt)[2:4]
            idxs = np.argsort(ii2)
            err_unsorted = err2[idxs]
            err_img = err_unsorted[automated_seg]
            plt.interactive = False
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,9))
            plt.setp(ax.flat, aspect=1.0, adjustable='box-forced')
            ax[0,0].imshow(raw, cmap='gray')
            viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[0, 0])
            ax[0,1].imshow(raw, cmap='gray')
            axes_image_1 = viz.imshow_rand(automated_seg, alpha=0.4, axis=ax[0,1])
            ax[1,0].imshow(raw, cmap='gray')
            viz.imshow_rand(gt, alpha=0.4, axis=ax[1, 0])
            ax[1,1].imshow(raw, cmap='gray')
            axes_image = viz.imshow_rand(gt, alpha=0.4, axis=ax[1, 1])
            ax[0, 0].set_title("Automated seg: click to show worst splits.")
            ax[0, 1].set_title("Worst merge comps in gt, colored by VI error.")
            ax[1, 0].set_title("Ground truth with random cm: click to show worst merges.")
            ax[1, 1].set_title("Worst split comps in the gt, colored by VI error.")

            def _onpress(event):
                if event.inaxes == ax[1, 0]:
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
                elif event.inaxes == ax[0, 0]:
                    if event.button != 1:
                        return
                    x, y = int(event.xdata), int(event.ydata)
                    comps = ev.split_components(automated_seg[y, x], cont, axis=0, num_elems=num_elem)
                    new_seg = np.zeros_like(gt.astype('float64'))
                    factor = (gt.max() // num_elem)
                    lim = 0.0
                    for i, (j,k, z) in enumerate(comps):
                        lim +=k
                        if z < 0.02:
                            continue
                        new_seg += (gt == j) * ((i + 1) * factor)
                    axes_image_1.set_array(new_seg)
                    fig.canvas.draw()
            
            fig.canvas.mpl_connect('button_press_event', _onpress)
            plt.ioff()
            plt.show()

view_all(gt, automated_seg)

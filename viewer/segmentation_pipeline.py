#!/usr/bin/python3
"""Script that automates the segmentation pipeline, employs the watershed algorithm on three-dimensional data
and uses the GALA algorithm for generating of a stack of automatic segmentations."""
# built-ins
import os
import datetime
import tempfile
import multiprocessing as mp
# libraries
from gala import evaluate as ev, imio, viz, morpho, agglo, classify, features
import line_profiler as lp
from skimage.segmentation import join_segmentations
from skimage.util import regular_seeds
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

# Go to directory containing CREMI samples.
os.chdir("/home/johnnyt/Documents/research_project_files/")
raw, gt = imio.read_cremi("Cremi_Data/sample_B_20160501.hdf", datasets=['volumes/raw', 'volumes/labels/neuron_ids'])
# Read in the boundary probability map for the volume_seg_bpm function
bpm = imio.read_h5_stack('raw_slice_1_Probabilities.h5', group='bpm_raw_b')
train_slice = (slice(15, 20), slice(0, 480), slice(0, 480))
test_slice = (slice(15, 20), slice(480, 960), slice(480, 960))
gt_training = gt[train_slice]
gt_testing = gt[test_slice]
raw_training = 1 - raw[train_slice] / 255
raw_testing = raw[test_slice]


def volume_seg_bpm():
    """Perform a segmentation on a volume using the GALA algorithm and a boundary probability map."""

    membrane_prob = bpm[..., 2]
    raw_bpm_testing = membrane_prob[test_slice]
    ws_larger_seeds = regular_seeds(raw_training[0].shape, n_points=500)
    ws_larger_seeds = np.broadcast_to(ws_larger_seeds, raw_training.shape)
    # Training step using membrane probability map
    ws_larger_train = morpho.watershed_sequence(membrane_prob[train_slice], ws_larger_seeds, n_jobs=-1)
    ws_larger_testing = morpho.watershed_sequence(raw_bpm_testing, ws_larger_seeds, n_jobs=-1)
    fm = features.moments.Manager()
    fh = features.histogram.Manager()
    fc = features.base.Composite(children=[fm, fh])
    # Construct a Region adjacency Graph of the training data
    g_train_larger = agglo.Rag(ws_larger_train, bpm[train_slice], feature_manager=fc)
    (X, Y, W, Merges) = g_train_larger.learn_agglomerate(gt_training, fc, classifier='logistic')[0]
    Y = Y[:, 0]
    rf_log_large = classify.get_classifier('logistic').fit(X, Y)
    learned_policy_large = agglo.classifier_probability(fc, rf_log_large)
    # Construct a Region adjacency Graph of the testing data
    g_test_large_bpm = agglo.Rag(ws_larger_testing, bpm[test_slice], feature_manager=fc,
                                 merge_priority_function=learned_policy_large)
    # Agglomerate until threshold set to infinity.
    g_test_large_bpm.agglomerate(np.inf)
    # Create a stack of segmentations at varying confidence level thresholds.
    seg_stack_large_bpm = [g_test_large_bpm.get_segmentation(t) for t in np.arange(0, 1, 0.01)]
    # Find the split VI for each of the segmentations
    split_vi_score_bpm = [ev.split_vi(seg_stack_large_bpm[t], gt_testing) for t in range(len(seg_stack_large_bpm))]
    split_vi_array_bpm = np.array(split_vi_score_bpm)
    # Identify the index of the lowest scoring segmentation
    best_seg_ind_bpm = np.argmin(split_vi_array_bpm.sum(axis=1))
    best_seg_bpm = seg_stack_large_bpm[best_seg_ind_bpm]
    joint_seg = join_segmentations(best_seg_bpm, gt_testing)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    ax[0, 0].set_title('Automatic Segmentation')
    ax[0, 1].set_title('Join Segmentation')
    ax[0, 2].set_title('Ground Truth')
    ax[1, 2].set_title('Raw Data')
    viz.imshow_rand(best_seg_bpm[3, ...], axis=ax[0, 0])
    ax[0, 0].axis('off')
    viz.imshow_rand(joint_seg[3, ...], axis=ax[0, 1])
    ax[0, 1].axis('off')
    viz.imshow_rand(gt_testing[3, ...], axis=ax[0, 2])
    ax[0, 2].axis('off')
    ax[1, 0].plot(split_vi_array_bpm[:, 1], split_vi_array_bpm[:, 0])
    ax[1, 0].set_xlabel('False merges')
    ax[1, 0].set_ylabel('False splits')
    ax[1, 0].set_title("Split VI")
    ax[1, 1].set_title("Cumulative VI")
    ax[1, 1].plot(split_vi_array_bpm.sum(axis=1))
    ax[1, 2].imshow(raw_testing[3,...])
    plt.show()


def volume_seg_non_bpm():
    """Perform a segmentation on a volume using the GALA algorithm and the raw data as input into the watershed map."""
   
    ws_larger_seeds = regular_seeds(raw_training[0].shape, n_points=300)
    ws_larger_seeds = np.broadcast_to(ws_larger_seeds, raw_training.shape)
    # Training step using output from watershedding step
    ws_larger_testing = morpho.watershed_sequence(raw_training, ws_larger_seeds, n_jobs=-1)
    fm = features.moments.Manager()
    fh = features.histogram.Manager()
    fc = features.base.Composite(children=[fm, fh])
    # Construct a Region adjacency Graph of the training data
    g_train_larger = agglo.Rag(ws_larger_testing, raw_training, feature_manager=fc)
    (X2, y2, w2, merges2) = g_train_larger.learn_agglomerate(gt_training, fc, classifier='logistic')[0]
    y2 = y2[:, 0]
    rf_log_large = classify.get_classifier('logistic').fit(X2, y2)
    learned_policy_large = agglo.classifier_probability(fc, rf_log_large)
    # Construct a Region adjacency Graph of the testing data
    g_test_large = agglo.Rag(ws_larger_testing, raw_training, feature_manager=fc,
                             merge_priority_function=learned_policy_large)
    # Agglomerate until threshold set to infinity.
    g_test_large.agglomerate(np.inf)
    # Create a stack of segmentations at varying confidence level thresholds.
    seg_stack_large = [g_test_large.get_segmentation(t) for t in np.arange(0, 1, 0.01)]
    # Find the split VI for each of the segmentations
    split_vi_score = [ev.split_vi(seg_stack_large[t], gt_testing) for t in range(len(seg_stack_large))]
    split_vi_array = np.array(split_vi_score)
    # Identify the index of the lowest scoring segmentation
    best_seg_ind = np.argmin(split_vi_array.sum(axis=1))
    best_seg = seg_stack_large[best_seg_ind]
    joint_seg = join_segmentations(best_seg, gt_testing)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    ax[0, 0].set_title('Automatic Segmentation')
    ax[0, 1].set_title('Join Segmentation')
    ax[0, 2].set_title('Ground Truth')
    viz.imshow_rand(best_seg[3, ...], axis=ax[0, 0])
    ax[0, 0].axis('off')
    viz.imshow_rand(joint_seg[3, ...], axis=ax[0, 1])
    ax[0, 1].axis('off')
    viz.imshow_rand(gt_training[3, ...], axis=ax[0, 2])
    ax[0, 2].axis('off')
    ax[1, 0].plot(split_vi_array[:, 1], split_vi_array[:, 0])
    ax[1, 0].set_xlabel('False merges')
    ax[1, 0].set_ylabel('False splits')
    ax[1, 0].set_title("Split VI")
    ax[1, 1].set_title("Cumulative VI")
    ax[1, 1].plot(split_vi_array.sum(axis=1))
    ax[1, 2].imshow(raw_testing[3, ...])
    plt.show()

def write_out():
    '''Write out the worst false split comps, worst false merge comps and sparse versions of the ground truth
    and automated segmentation.'''
    spacing = [4, 4, 40]
    date = datetime.datetime.now()
    short_date = date.date()

    # Merges
    # Sort the worst false merge components in relation to the join seg.
    merge_idxs_m, merge_errs_m = ev.sorted_vi_components(joint_seg, best_seg_bpm)[0:2]
    cont_table_m = ev.contingency_table(best_seg_bpm, joint_seg)
    # Return the indices of the false merge components in the automatic segmentation overlapping the most with the
    # selected join segmentation components..
    worst_merge_comps_m = ev.split_components(merge_idxs_m[0], num_elems=10, cont=cont_table_m.T, axis=1)
    worst_merge_array_m = np.array(worst_merge_comps_m[0:3], dtype=np.int64)
    extracted_worst_merge_comps_m = imio.extract_segments(ids=worst_merge_array_m[:, 0], seg=joint_seg)

    #Splits
    # Sort the worst false split components in relation to the join seg.
    split_idxs_s, split_errs_s = ev.sorted_vi_components(joint_seg, gt_testing)[0:2]
    cont_table_s = ev.contingency_table(joint_seg, gt_testing)
    # Return the indices of the false split components in the join seg overlapping the most with the
    # selected ground truth components.
    worst_split_comps_s = ev.split_components(split_idxs_s[0], num_elems=10, cont=cont_table_s.T, axis=0)
    worst_split_array_s = np.array(worst_split_comps_s[0:3], dtype=np.int64)
    extracted_worst_split_comps_s = imio.extract_segments(ids=worst_split_array_s[:, 0], seg=joint_seg)

    # Return the sorted indices of the components with the largest area (highest number of labels) in the ground truth
    target_segs_gt = np.argsort(np.bincount(gt_testing.astype(int).ravel()))[-10:]
    # Return the sorted indices of the components with the largest area (highest number of labels) in the automatic seg
    target_segs_auto = np.argsort(np.bincount(best_seg_largest_bpm.astype(int).ravel()))[-10:]
    # Extract these components and write them out into vtk files
    sparse_largest_gt = imio.extract_segments(gt_testing, ids=target_segs_gt)
    sparse_largest_auto = imio.extract_segments(best_seg_largest_bpm, ids=target_segs_auto)
    imio.write_vtk(sparse_largest_auto, fn=f'sparse_largest_auto_{short_date}.vtk', spacing=spacing)
    imio.write_vtk(sparse_largest_gt, fn=f'sparse_largest_gt_{short_date}.vtk', spacing=spacing)
    imio.write_vtk(raw_testing, fn=f'raw_testing{short_date}.vtk', spacing=spacing)
    imio.write_vtk(extracted_worst_merge_comps_m, fn=f'worst_merge_comps_m_{short_date}.vtk', spacing=spacing)
    imio.write_vtk(extracted_worst_split_comps_s, fn=f'worst_split_comps_s_{short_date}.vtk', spacing=spacing)

if __name__ == '__main__':
    volume_seg_bpm()

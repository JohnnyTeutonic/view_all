import os
import datetime
import tempfile
from gala import evaluate as ev, imio, viz, morpho, agglo, classify, features
from skimage.segmentation import join_segmentations
from skimage.color import label2rgb
from skimage.util import regular_seeds
from skimage import io
import numpy as np
import numexpr
from matplotlib import pyplot as plt
os.chdir('/home/johnnyt/Documents/research_project_files')
raw, gt = imio.read_cremi("Cremi_Data/sample_B_20160501.hdf", datasets=['volumes/raw', 'volumes/labels/neuron_ids'])
bpm = imio.read_h5_stack('raw_slice_1_Probabilities.h5', group='bpm_raw_b')
membrane_prob = bpm[..., 2]
train_slice = (slice(0, 15), slice(0, 480), slice(0, 480))
test_slice = (slice(0, 15), slice(480, 960), slice(480, 960))
gt_larger_2 = gt[train_slice]
raw_larger_2 = numexpr.evaluate('1-raw[train_slice]/255')
ws_larger_seeds_2 = regular_seeds(raw_larger_2[0].shape, n_points=700)
ws_larger_seeds_2 = np.broadcast_to(ws_larger_seeds_2, raw_larger_2.shape)
ws_larger_water = morpho.watershed_sequence(membrane_prob[train_slice], ws_larger_seeds_2, n_jobs=-1)
raw_larger_testing_2 = membrane_prob[test_slice]
gt_larger_testing_2 = gt[test_slice]
gg = np.argsort(np.bincount(gt_larger_2.astype(int).ravel()))[-10:]
sparse_large = imio.extract_segments(gt_larger_2, ids = gg)
ws_larger_testing_2 = morpho.watershed_sequence(raw_larger_testing_2, ws_larger_seeds_2, n_jobs=-1)
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])
g_train_larger_2 = agglo.Rag(ws_larger_water, bpm[train_slice], feature_manager=fc)
(X2, y2, w2, merges2) = g_train_larger_2.learn_agglomerate(gt_larger_2, fc, classifier='logistic')[0]
y2 = y2[:, 0]
rf_log_large_2 = classify.get_classifier('logistic').fit(X2,y2)
learned_policy_large_2 = agglo.classifier_probability(fc, rf_log_large_2)
g_test_large_2 = agglo.Rag(ws_larger_testing_2, bpm[test_slice], feature_manager=fc, merge_priority_function=learned_policy_large_2)
g_test_large_2.agglomerate(np.inf)
seg_stack_large_2 = [g_test_large_2.get_segmentation(t) for t in np.arange(0,1, 0.01)]
split_vi_score_large_2 = [ev.split_vi(seg_stack_large_2[t], gt_larger_testing_2) for t in range(len(seg_stack_large_2))]
split_vi_array_2 = np.array(split_vi_score_large_2)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(split_vi_array_2[:, 1], split_vi_array_2[:, 0])
target_segs = np.argsort(np.bincount(seg_stack_large_2[19].astype(int).ravel()))[-10:]
plt.plot(split_vi_array_new.sum(axis=1))
fig, ax = plt.subplots()
ax.plot(split_vi_array_new[:, 1], split_vi_array_new[:, 0])
ax.set_aspect(1)
fig1, ax1 = plt.subplots()
ax1.imshow(raw_larger_2[0], alpha=1, cmap='gray')
viz.imshow_rand(ws_larger_water[0], alpha=0.4, axis=ax)
sorted_seg_indices = np.argsort(split_vi_array_2.sum(axis=1))
best_seg_id = np.argmin(sorted_seg)
best_seg = seg_stack_large_2[best_seg_id]
target_segs = np.argsort(np.bincount(seg_stack_large_2[19].astype(int).ravel()))[-10:]
extracted_segs = imio.extract_segments(seg_stack_large_2[19], ids=target_segs)
joint_seg = join_segmentations(best_seg, gt_larger_testing_2)
imio.write_h5_stack(npy_vol=split_vi_array_2, compression='lzf', fn='stack_of_segs_20_10.h5')
imio.write_vtk(extracted_seg, fn='extraced_seg_5_10.vtk',spacing=[4, 4, 40])
imio.write_vtk(gt_larger_2, fn='extraced_gt_5_10.vtk',spacing=[4, 4, 40])
imio.write_vtk(img_as_ubyte(raw_larger_2), fn='extraced_raw_5_10.vtk', spacing=[4, 4, 40])

def write_out_info(stack_of_segs):
    """Write out vtk files of worst merge and worst split comps, and stack of agglomerated segmentations."""
    date = datetime.datetime.now()
    short_date = date.date()
    merge_idxs_m, merge_errs_m = ev.sorted_vi_components(joint_seg, best_seg_bpm)[2:4]
    cont_table_m = ev.contingency_table(joint_seg, best_seg_bpm)
    worst_merge_comps_m = ev.split_components(merge_idxs_m[0], num_elems=10, cont=cont_table_m.T, axis=0)
    worst_merge_array_m = np.array(worst_merge_comps_m[0:3], dtype=np.int64)
    worst_merge_array_m[:, 0]
    np.cumsum(np.array(worst_merge_comps_m)[:, 1])
    extracted_worst_merge_comps_m = imio.extract_segments(ids=worst_merge_array_m[:, 0], seg=joint_seg)
    imio.write_vtk(extracted_worst_merge_comps_m, fn=f'worst_merge_comps_m_{short_date}.vtk',spacing=[4, 4, 40])
    split_idxs_s, split_errs_s = ev.sorted_vi_components(joint_seg, gt_raw_testing)[0:2]
    cont_table_s = ev.contingency_table(joint_seg, gt_raw_testing)
    worst_split_comps_s = ev.split_components(split_idxs_s[0], num_elems=10, cont=cont_table_s.T, axis=0)
    worst_split_array_s = np.array(worst_split_comps_s[0:3], dtype=np.int64)
    worst_split_array_s[:, 0]
    np.cumsum(np.array(worst_split_comps_s)[:, 1])
    extracted_worst_split_comps_s = imio.extract_segments(ids=worst_split_array_s[:, 0], seg=joint_seg)
    imio.write_vtk(extracted_worst_split_comps_s, fn=f'worst_split_comps_s_{short_date}.vtk',spacing=[4, 4, 40])

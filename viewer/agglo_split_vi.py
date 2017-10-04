from gala import evaluate as ev, imio, viz, morpho, agglo, classify, features
from skimage.util import regular_seeds
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
raw, gt = imio.read_cremi("Cremi_Data/sample_B_20160501.hdf", datasets=['volumes/raw', 'volumes/labels/neuron_ids'])
gt_larger_2 = gt[0:15, 0:120, 0:120]
raw_larger_2 = raw[0:15, 0:120, 0:120]
raw_larger_testing_2 = raw[0:15, 125:245, 125:245]
gt_larger_testing_2 = gt[0:15, 125:245, 125:245]
ws_larger_seeds_2 = regular_seeds(raw_larger_2[0].shape, n_points=700)
ws_larger_seeds_2 = np.broadcast_to(ws_larger_seeds_2, raw_larger_2.shape)
ws_larger_water = morpho.watershed_sequence(raw_larger_2, ws_larger_seeds_2, n_jobs=-1)
ws_larger_testing_2 = morpho.watershed_sequence(raw_larger_testing_2, ws_larger_seeds_2, n_jobs=-1)
gg = np.argsort(np.bincount(gt_larger_2.astype(int).ravel()))[-10:]
sparse_large = imio.extract_segments(raw_larger_2, ids = gg)
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])
g_train_larger_2 = agglo.Rag(ws_larger_water, raw_larger_2, feature_manager=fc)
(X2, Y2, W2, merges2) = g_train_larger_2.learn_agglomerate(gt_larger_2, fc, classifier='logistic')[0]
Y2 = Y2[:, 0]
rf_log_large_2 = classify.get_classifier('logistic').fit(X2, Y2)
learned_policy_large_2 = agglo.classifier_probability(fc, rf_log_large_2)
g_test_large_2 = agglo.Rag(ws_larger_testing_2, raw_larger_testing_2, feature_manager=fc, merge_priority_function=learned_policy_large_2)
g_test_large_2.agglomerate(np.inf)
seg_stack_large_2 = [g_test_large_2.get_segmentation(t) for t in np.arange(0,1, 0.01)]
split_vi_score_large_2 = [ev.split_vi(seg_stack_large_2[t], gt_larger_testing_2) for t in range(len(seg_stack_large_2))]
split_vi_array_2 = np.array(split_vi_score_large_2)
cont_a = [ev.contingency_table(seg_stack_large_2[index], gt_larger_2) for index,_ in enumerate(seg_stack_large_2)]
cont_stack = [ev.split_vi(cont_a[i]) for i,_ in enumerate(cont_a)]
split_vi_array_new = np.array(cont_stack)
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax[0].plot(split_vi_array_2[:, 1], split_vi_array_2[:, 0])
ax[1].plot(split_vi_array_new[:, 1], split_vi_array_new[:, 0])
ax[0].set_title("Split-VI without CT")
ax[1].set_title("Split-VI using CT")

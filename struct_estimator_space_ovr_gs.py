"""
HCP: sparse region-network estimator

"One of the greatest challenges left for systems
neuroscience is to understand the normal and dysfunctional
operations of the cerebral cortex by relating local and global
patterns of activity [...]."

Buzsaki 2007 Nature
"""
print __doc__
"""
Notes:
- turn eta_g multiplier for all netowrks and for all regions into
a model HYPERPARAMETER -> both these could be transformed into a
contjoint reg/net ratio
- test max_it=100 versus max_it=500
- 2-class versus 18-class problems
- test known inseparable 2-class problem
"""
import spams

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from matplotlib import pylab as plt
from nilearn.image import concat_imgs, resample_img
import joblib
import time
import scipy.sparse as ssp
from nilearn.image import index_img
from nilearn import datasets
from scipy.stats import zscore

FORCE_TWO_CLASSES = False
# REG_PEN = 'l1'
# REG_PEN = 'l2'
# REG_PEN = 'sparse-group-lasso-l2'
# REG_PEN = 'sparse-group-lasso-linf'
# REG_PEN = 'tree-l0'
# REG_PEN = 'tree-l2'
# REG_PEN = 'trace-norm'
MY_MAX_IT = 100
MY_DATA_RATIO = 100
N_JOBS = 5
LAMBDA_GRID = np.linspace(0.1, 1.0, 10)

RES_NAME = 'srne_benchmark_space_ovr_gs'
if FORCE_TWO_CLASSES:
    RES_NAME += '_2cl'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

REGS = ['tree-l2']

# REGS = ['tree-l0', 'tree-l2',
#     'sparse-group-lasso-linf', 'sparse-group-lasso-l2',
#     'l1', 'l2',
#     'trace-norm']

on_server = op.exists(r'/storage')

##############################################################################
# load+preprocess data
##############################################################################

# load the mask
mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()
print('Mask shape is %i/%i/%i' % nifti_masker.mask_img_.get_data().shape)
print('Mask voxels: %i' % mask_nvox)

# load atlas rois
# atlas_path = 'resources/aal.nii'  # HACK !
crad = datasets.fetch_craddock_2012_atlas()
atlas_nii = index_img(crad['scorr_mean'], 19) #42)

r_atlas_nii = resample_img(
    img=atlas_nii,
    target_affine=nifti_masker.mask_img_.get_affine(),
    target_shape=nifti_masker.mask_img_.shape,
    interpolation='nearest'
)
r_atlas_nii.to_filename('debug_ratlas.nii.gz')
atlas_labels = nifti_masker.transform(r_atlas_nii)[0, :]
atlas_labels += 1

# impose continuity and 1-based indexing in label numbers (for SPAMs)
atlas2 = np.zeros_like(atlas_labels)
for i_label, label in enumerate(np.unique(atlas_labels)):
    atlas2[np.where(atlas_labels == label)[0]] = i_label + 1  # one-based indexing
atlas_labels = atlas2

n_regions = len(np.unique(atlas_labels))

assert n_regions == 196

# load HCP task data
print('Loading data...')
if on_server:
    X_task, labels = joblib.load('/storage/workspace/danilo/prni2015/preload_HT_3mm')
else:
    X_task, labels = joblib.load('/git/prni2015/preload_HT_3mm')

labels = np.array(labels)
if FORCE_TWO_CLASSES:
    inds1 = labels == 0  # TOM
    inds2 = labels == 1
    # inds1 = labels == 2  # TOM
    # inds2 = labels == 3
    # inds1 = labels == 4  # object grasp/orientation
    # inds2 = labels == 5
    inds = np.logical_or(inds1, inds2)

    X_task = X_task[inds]
    Y = labels[inds].astype(np.float64)
    Y[Y==2] = -1
    Y[Y==3] = 1
else:
    Y = np.float64(labels)

print('Done (%.2f MB)!' % (X_task.nbytes / 1e6))

# subsample input data
from sklearn.cross_validation import StratifiedShuffleSplit
if MY_DATA_RATIO != 100:
    ratio = MY_DATA_RATIO / 100.
    print('DATA SUBSELECTION at %.2f!!!' % ratio)
    folder = StratifiedShuffleSplit(Y, n_iter=10, train_size=ratio,
                                    random_state=42)
    inds_train, _ = iter(folder).next()
    X_task = X_task[inds_train]
    Y = Y[inds_train]

# run SPAMs
from nilearn.decoding import SpaceNetClassifier

folder = StratifiedShuffleSplit(Y, n_iter=10, test_size=0.1,
                                random_state=42)
inds_train, inds_test = iter(folder).next()
X_train = X_task[inds_train]
# all estimators will have the same training set due to random_state
Y_train = Y[inds_train]
X_test = X_task[inds_test]
Y_test = Y[inds_test]

X_train_nii = nifti_masker.inverse_transform(X_train)
X_test_nii = nifti_masker.inverse_transform(X_test)

clf = SpaceNetClassifier(
    penalty='tv-l1', loss='logistic',
    max_iter=100, n_jobs=1,
    verbose=True
)

from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

param_grid = {'estimator__l1_ratios': LAMBDA_GRID}
# param_grid = {'estimator__lambda1': [0.1]}

# start time
start_time = time.time()

clf_ovr = OneVsRestClassifier(clf, n_jobs=1)
clf_ovr_gs = GridSearchCV(clf_ovr, param_grid, n_jobs=N_JOBS, cv=3)
clf_ovr_gs.fit(X_train_nii, Y_train)

train_acc = clf_ovr_gs.score(X_train_nii, Y_train)
print('Train-Accuracy: %.2f' % train_acc)

test_acc = clf_ovr_gs.score(X_test_nii, Y_test)
print('Test-Accuracy: %.2f' % test_acc)
y_pred = clf_ovr_gs.predict(X_test_nii)
test_prfs = precision_recall_fscore_support(y_pred, Y_test)

clf_ovr_gs.train_acc = train_acc
clf_ovr_gs.test_acc = test_acc
clf_ovr_gs.test_prfs = test_prfs

# stop time
dur = time.time() - start_time
total_mins = dur / 60
hs, mins = divmod(total_mins, 60)
print('-' * 80)
print("Elapsed time: %i hours and %i minutes" % (hs, mins))

out_fname = 'l1%.2f_maxit%i' % (clf.l1_ratios, clf.max_iter)
out_path = op.join(WRITE_DIR, out_fname)
joblib.dump(clf, out_path, compress=9)


STOP

# test_acc = gs.best_estimator_.score(X_test, Y_test)
# print('Accuracy: %.2f' % test_acc)
# 
# 
# out_fname = 'test_TOMaudio_vs_video_%s_lambda%.3f.nii.gz' % (
#     clf.regul, clf.lambda1
# )
# nifti_masker.inverse_transform(clf.W_.T).to_filename(out_fname)

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    n_comp = len(components)
    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%i' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            plot_stat_map(nii_z_fwhm, bg_img='colin.nii', threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))

clf = joblib.load('')

# dump_comps(nifti_masker, 'trace_dataratio%i_maxit100', coef_per_class, threshold=0.0)



# rsync -vza dbzdok@drago:/storage/workspace/danilo/srne/srne_benchmark/* /git/srne/srne_benchmark
# rsync -vza dbzdok@drago:/storage/workspace/danilo/srne/srne_benchmark_space/* /git/srne/srne_benchmark_space

import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import re
%matplotlib qt

for reg in [REGS[0]]:
    anal_str = '%s_dataratio%i_maxit%i' % (reg, MY_DATA_RATIO, MY_MAX_IT)
    tar_dump_file = '%s/%s' % (WRITE_DIR, anal_str)
    if not op.exists(tar_dump_file):
        print('SKIPPED: %s' % tar_dump_file)
        # continue
    clf_ovr_gs = joblib.load(tar_dump_file)
    plt.close('all')
    plt.figure()
    means = []
    stds = []
    lbds = []
    for grid_str in clf_ovr_gs.grid_scores_:
        mean, std, lbd = re.findall("\d+.\d+", str(grid_str))
        mean, std, lbd = np.float(mean), np.float(std), np.float(lbd)
        means.append(mean)
        stds.append(std)
        lbds.append(lbd)

    plt.errorbar(lbds, y=means, yerr=stds, color='r', linewidth=2)
    plt.xlabel('$\lambda$')
    plt.ylabel('accuracy (mean)')
    plt.ylim(0.0, 1.0)
    plt.xticks(lbds)
    plt.title('GridSearch: ' + anal_str)
    plt.savefig(tar_dump_file + '_gs.png')

for reg in [REGS[0]]:
    anal_str = '%s_dataratio%i_maxit%i' % (reg, MY_DATA_RATIO, MY_MAX_IT)
    tar_dump_file = '%s/%s' % (WRITE_DIR, anal_str)
    if not op.exists(tar_dump_file):
        print('SKIPPED: %s' % tar_dump_file)
        continue
    clf_ovr_gs = joblib.load(tar_dump_file)
    # weights = clf_ovr_gs.best_estimator_.estimators_[0].W_.T

    n_est = len(clf_ovr_gs.best_estimator_.estimators_)
    coef_per_class = [est.W_ for est in clf_ovr_gs.best_estimator_.estimators_]
    coef_per_class = np.squeeze(coef_per_class)
    dump_comps(
        nifti_masker,
        anal_str + '_weights',
        coef_per_class,
        threshold=0.0)

plt.close('all')
contrasts_names = [
    'REWARD-PUNISH', 'PUNISH-REWARD', 'SHAPES-FACES', 'FACES-SHAPES',
    'RANDOM-TOM', 'TOM-RANDOM',

    'MATH-STORY', 'STORY-MATH',
    'T-AVG', 'F-H', 'H-F',
    'MATCH-REL', 'REL-MATCH',

    'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG',
    '2BK-0BK'
]

from nilearn import plotting
from scipy.stats import zscore
for i_cont, cont_name in enumerate(contrasts_names):
    out_fname = 'plots/tree-l2_weights_%s' % cont_name
    coef = coef_per_class[i_cont, :]
    weight_nii = nifti_masker.inverse_transform(
        coef)
    plotting.plot_stat_map(weight_nii, cut_coords=(0, 0, 0),
                           title='', bg_img='colin.nii',
                           colorbar=True, draw_cross=False,
                           black_bg=True)
    plt.savefig(out_fname + '_raw.png',
                dpi=200, transparent=True)
    
    coef_z = zscore(coef)
    weight_nii = nifti_masker.inverse_transform(
        coef_z)
    plotting.plot_stat_map(weight_nii, cut_coords=(0, 0, 0),
                           title='', bg_img='colin.nii',
                           colorbar=True, draw_cross=False,
                           black_bg=True)
    plt.savefig(out_fname + '_zmap.png',
                dpi=200, transparent=True)
    
    
    